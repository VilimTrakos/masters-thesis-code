#!/usr/bin/env python3
"""
Inferencija: voltage klasifikator + fault detektor + multi-strategija detekcija skokova.

Dva koraka:
  1. Voltage klasifikator -> predikcija napona + pouzdanost
  2. Fault detektor -> normal / fan_imbalance / movement / voltage_spike
"""

import argparse
import json
import os
import time
import traceback

import numpy as np
import lightgbm as lgb
import paho.mqtt.client as mqtt


class VoltageFaultPredictor:
    """
    Dvodijelni klasifikator:
      Faza 1: voltage klasifikator (3v-18v)
      Faza 2: fault detektor (normal / fan_imbalance / movement) + voltage_spike pravilo
    """

    def __init__(self, model_dir):
        # ucitavanje konfiguracije voltage klasifikatora
        putanja_modela = os.path.join(model_dir, 'model_config.json')
        with open(putanja_modela) as datoteka:
            self.voltage_cfg = json.load(datoteka)

        # ucitavanje konfiguracije fault detektora
        putanja_kvara = os.path.join(model_dir, 'fault_config.json')
        with open(putanja_kvara) as datoteka:
            self.fault_cfg = json.load(datoteka)

        self.voltage_classes = self.voltage_cfg['classes']
        self.fault_classes = self.fault_cfg['classes']  # ['normal', 'fan_imbalance', 'movement']
        self.window_size = self.voltage_cfg['window_size']
        self.raw_features = self.voltage_cfg['raw_features']
        self.ratio_defs = self.voltage_cfg.get('ratio_defs', [])

        # skalari voltage klasifikatora
        self.scaler_mean = np.array(self.voltage_cfg['scaler_mean'], dtype=np.float32)
        self.scaler_scale = np.array(self.voltage_cfg['scaler_scale'], dtype=np.float32)
        self.log_scaler_mean = np.array(self.voltage_cfg['log_scaler_mean'], dtype=np.float32)
        self.log_scaler_scale = np.array(self.voltage_cfg['log_scaler_scale'], dtype=np.float32)

        # skalari fault detektora
        self.fault_scaler_mean = np.array(self.fault_cfg['scaler_mean'], dtype=np.float32)
        self.fault_scaler_scale = np.array(self.fault_cfg['scaler_scale'], dtype=np.float32)
        self.fault_log_scaler_mean = np.array(self.fault_cfg['log_scaler_mean'], dtype=np.float32)
        self.fault_log_scaler_scale = np.array(self.fault_cfg['log_scaler_scale'], dtype=np.float32)

        # ucitavanje LightGBM modela
        voltage_model_file = self.voltage_cfg.get('model_file', 'fan_lightgbm.txt')
        voltage_model_path = os.path.join(model_dir, voltage_model_file)
        self.voltage_model = lgb.Booster(model_file=voltage_model_path)

        fault_model_file = self.fault_cfg.get('model_file', 'fault_lightgbm.txt')
        fault_model_path = os.path.join(model_dir, fault_model_file)
        self.fault_model = lgb.Booster(model_file=fault_model_path)

        # parametri PCA + Mahalanobis detekcije anomalija za voltage klasifikator
        anomaly_cfg = self.voltage_cfg['anomaly_detection']
        self.pca_components = np.array(anomaly_cfg['pca_components'], dtype=np.float32)
        self.pca_mean = np.array(anomaly_cfg['pca_mean'], dtype=np.float32)
        self.class_means_pca = {
            klasa: np.array(vrijednost, dtype=np.float32)
            for klasa, vrijednost in anomaly_cfg['class_means_pca'].items()
        }
        self.precision_matrix = np.array(anomaly_cfg['precision_matrix'], dtype=np.float32)
        self.class_thresholds = {k: float(v) for k, v in anomaly_cfg['class_thresholds'].items()}
        self.min_confidence = float(anomaly_cfg['min_confidence'])

        self.buffer = []

        # klizajuci spremnik osnove za detekciju skokova ovisnu o kontekstu.
        # Drzi sirove vektore znacajki nedavnih uzoraka. Outliere ce ionako
        # odbaciti sama MAD detekcija pa nije problem ako su unutra spremljeni
        # i uzorci snimljeni tijekom skoka ili kvara.
        self.baseline_buffer = []
        self.baseline_max_size = 50

        # pracenje resetiranja spremnika nakon skoka.
        # Nakon sto skok obrise spremnik, kratak cooldown sprjecava da se
        # spremnik odmah opet brise. To omogucuje da se Strategija C detekcije
        # nakon ponovnog punjenja (stara osnova vs novi napon) prikazu kao
        # skokovi bez ulaska u petlju ponovljenih reset-ova.
        self._sample_count = 0
        self._last_spike_clear = -100

    def _scale(self, window_np, scaler_mean, scaler_scale):
        """StandardScaler transformacija: (x - mean) / scale."""
        return (window_np - scaler_mean) / (scaler_scale + 1e-12)

    def _minmax_normalize(self, scaled_window):
        """Min-max normalizacija po prozoru, svaka znacajka ide na [0, 1]."""
        minimum = scaled_window.min(axis=0, keepdims=True)
        maksimum = scaled_window.max(axis=0, keepdims=True)
        return (scaled_window - minimum) / (maksimum - minimum + 1e-8)

    def _log_transform(self, window_np):
        """Log1p transformacija s ocuvanjem predznaka."""
        return np.sign(window_np) * np.log1p(np.abs(window_np))

    def _window_stats(self, scaled):
        """(W, F) -> (5*F,) ravni vektor: [prosjek, std, min, max, nagib]"""
        prosjek = scaled.mean(axis=0)
        std = scaled.std(axis=0)
        minimum = scaled.min(axis=0)
        maksimum = scaled.max(axis=0)
        velicina_prozora = scaled.shape[0]
        vrijeme = np.arange(velicina_prozora, dtype=np.float32)
        srednje_vrijeme = vrijeme.mean()
        # nagib linearne regresije po metodi najmanjih kvadrata: 
        # sum((t - t_mean) * (y - y_mean)) / sum((t - t_mean)^2)
        devijacija_vremena = vrijeme - srednje_vrijeme
        varijanca_vremena = (devijacija_vremena ** 2).sum()
        devijacija_prozora = scaled - prosjek
        povezanost_znacajki = (devijacija_prozora * devijacija_vremena[:, None]).sum(axis=0)
        nagibi = povezanost_znacajki / (varijanca_vremena + 1e-12)
        return np.concatenate([prosjek, std, minimum, maksimum, nagibi]).astype(np.float32)

    def _build_multiview(self, window_np, scaler_mean, scaler_scale,
                         log_scaler_mean, log_scaler_scale):
        """
        Gradi multi-view vektor znacajki iz sirovog prozora (W, F).

        Pogledi:
          1. Raw: StandardScaler -> statistike -> 240 znacajki
          2. MinMax: po prozoru [0,1] -> statistike -> 240 znacajki
          3. Log: log1p -> poseban skalar -> statistike -> 240 znacajki
          4. Diff: prva razlika -> statistike -> 240 znacajki

        Vraca: (960,) ravni vektor znacajki
        """
        # Pogled 1: sirovi (samo skaliranje)
        skalirani = self._scale(window_np, scaler_mean, scaler_scale)
        statistike_sirovih = self._window_stats(skalirani)

        # Pogled 2: MinMax normalizacija
        normirani = self._minmax_normalize(skalirani)
        statistike_minmax = self._window_stats(normirani)

        # Pogled 3: Log transformacija (na sirovim podacima, poseban skalar)
        logaritmirani = self._log_transform(window_np)
        log_skalirani = self._scale(logaritmirani, log_scaler_mean, log_scaler_scale)
        statistike_loga = self._window_stats(log_skalirani)

        # Pogled 4: Prva razlika (na skaliranim podacima)
        razlika = np.diff(skalirani, axis=0)
        statistike_razlike = self._window_stats(razlika)

        return np.concatenate([statistike_sirovih, statistike_minmax, statistike_loga, statistike_razlike])

    # znacajke relevantne za detekciju skoka (ukljucuje v_rms za naponske tranzicije)
    SPIKE_FEATURE_INDICES = {
        'v_rms_x': 4, 'v_rms_y': 5, 'v_rms_z': 6, 'v_rms_mag': 7,
        'a_rms_x': 8, 'a_rms_y': 9, 'a_rms_z': 10, 'a_rms_mag': 11,
        'peak_peak_x': 12, 'peak_peak_y': 13, 'peak_peak_z': 14,
        'crest_x': 15, 'crest_y': 16, 'crest_z': 17,
        'kurtosis_x': 24, 'kurtosis_y': 25, 'kurtosis_z': 26,
        'impulse_x': 30, 'impulse_y': 31, 'impulse_z': 32,
    }

    def _detect_voltage_spike(self, window_np):
        """
        Detekcija naponskog skoka s vise strategija.

        Pratimo skok-relevantne znacajke iz SPIKE_FEATURE_INDICES (v_rms, a_rms,
        peak_peak, crest_factor, kurtosis, impulse_factor). Strategije A/B/C
        traze nagla kratka odstupanja (1-2 uzorka koja odskacu), dok Strategija
        D hvata postupne tranzicije (3+ uzoraka prelazi na novu razinu).

        Cetiri komplementarne strategije:

          A. Outlier unutar prozora (MAD-bazirano)
             - Za svaku znacajku, racuna se median i MAD prozora
             - Trazi uzorke s modificiranim z-score > 5.0
             - Ako 1-2 uzorka odskacu u barem max(3, n/2) znacajki -> SKOK

          B. Iznenadni multi-feature skok
             - Za svaki par susjednih uzoraka, broji znacajke s >250% promjene
             - Ako jedan korak pokaze skok u barem max(4, n/2) znacajki -> SKOK

          C. Devijacija od klizajuce osnove (samo ako baseline ima dovoljno uzoraka)
             - Iz prozora uzme vrijednost s najvecom devijacijom od medijana osnove
             - Racuna modificirani z-score koristeci MAD osnove (prag 6.0)
             - Ako bar max(4, n*0.6) znacajki snazno odstupa -> SKOK

          D. V_rms rampa unutar prozora (postupna tranzicija)
             - Usporedi median v_rms-a prve polovice prozora vs druge polovice
             - Gating: oba mediana moraju biti > 0.2 (preskace mali napon)
             - Ako relativna promjena izmedu polovica > 25% -> RAMPA

        Kombinacija:
          - Iz A/B/C se barem 2 moraju sloziti za detekciju skoka
          - D moze sama pokrenuti detekciju (jer rampe nisu vidljive A/B/C)

        Vraca:
            is_spike (bool)
            confidence (float): 0.0 ako nije skok, inace u [0.7, 1.0] ovisno
                                o broju strategija koje su se okinule
            detail (dict): dijagnostika po strategiji
        """
        feature_indices = list(self.SPIKE_FEATURE_INDICES.values())
        n_features = len(feature_indices)

        # ---- Strategija A: detekcija outliera unutar prozora preko MAD ----
        # z-score prag 5.0, dovoljan udio glasova 0.5
        outlier_glasovi = np.zeros(window_np.shape[0], dtype=np.float32)
        for idx in feature_indices:
            stupac = window_np[:, idx].astype(np.float64)
            median_stupca = np.median(stupac)
            mad = np.median(np.abs(stupac - median_stupca)) + 1e-9
            mod_z = np.abs(stupac - median_stupca) / (1.4826 * mad)
            outlier_glasovi += (mod_z > 5.0).astype(np.float32)

        prag_glasova = max(3, int(n_features * 0.5))
        outlier_maska = outlier_glasovi >= prag_glasova
        broj_outliera = int(outlier_maska.sum())
        max_glasova = int(outlier_glasovi.max())

        # kratki outlier: tocno 1-2 uzorka znacajno odstupaju (ne cijeli prozor)
        is_brief_outlier = (1 <= broj_outliera <= 2)

        # ---- Strategija B: detekcija iznenadnog multi-feature skoka ----
        # prag relativne promjene 2.5 (>250%), dovoljan udio 0.5
        max_jump_features = 0
        jump_step = -1
        for t in range(1, window_np.shape[0]):
            broj_skocenih_znacajki = 0
            for idx in feature_indices:
                v_prev = float(window_np[t-1, idx])
                v_curr = float(window_np[t, idx])
                nazivnik = abs(v_prev) + 1e-6
                if nazivnik > 1e-3:
                    rel_promjena = abs(v_curr - v_prev) / nazivnik
                    if rel_promjena > 2.5:  # >250% relativne promjene
                        broj_skocenih_znacajki += 1
            if broj_skocenih_znacajki > max_jump_features:
                max_jump_features = broj_skocenih_znacajki
                jump_step = t

        is_sudden_jump = max_jump_features >= max(4, int(n_features * 0.5))

        # ---- Strategija C: z-score prema klizajucoj osnovi (ako ima dovoljno povijesti) ----
        # mod_z prag 6.0, dovoljan udio 0.6
        baseline_score = 0
        baseline_used = False
        if len(self.baseline_buffer) >= 20:
            baseline_used = True
            osnova = np.array(self.baseline_buffer[-50:], dtype=np.float64)
            for idx in feature_indices:
                vrijednosti_osnove = osnova[:, idx]
                median_osnove = np.median(vrijednosti_osnove)
                mad_osnove = np.median(np.abs(vrijednosti_osnove - median_osnove)) + 1e-9
                stupac_prozora = window_np[:, idx].astype(np.float64)
                najveca_devijacija = stupac_prozora[np.argmax(np.abs(stupac_prozora - median_osnove))]
                mod_z = abs(najveca_devijacija - median_osnove) / (1.4826 * mad_osnove)
                if mod_z > 6.0:
                    baseline_score += 1

        is_baseline_outlier = baseline_used and baseline_score >= max(4, int(n_features * 0.6))

        # ---- Strategija D: prijelaz v_rms unutar prozora (hvata postupne tranzicije) ----
        v_rms_stupac = window_np[:, 7].astype(np.float64)  # indeks v_rms_magnitude
        polovica = window_np.shape[0] // 2
        prvi_median = float(np.median(v_rms_stupac[:polovica]))
        drugi_median = float(np.median(v_rms_stupac[polovica:]))
        if prvi_median > 0.2 and drugi_median > 0.2:
            ramp_ratio = abs(drugi_median - prvi_median) / max(prvi_median, drugi_median)
            is_v_rms_ramp = ramp_ratio > 0.25
        else:
            is_v_rms_ramp = False

        # ---- Kombiniranje strategija ----
        # A/B/C trebaju da se barem 2 slozu za robusnu detekciju skoka.
        # Strategija D (naponski prijelaz) moze sama pokrenuti - hvata postupne tranzicije.
        n_abc = sum([is_brief_outlier, is_sudden_jump, is_baseline_outlier])
        n_triggered = n_abc + int(is_v_rms_ramp)
        is_spike = (n_abc >= 2) or is_v_rms_ramp

        confidence = min(0.5 + 0.2 * n_triggered, 1.0) if is_spike else 0.0

        detail = {
            'within_window_outliers': broj_outliera,
            'max_outlier_votes': max_glasova,
            'max_jump_features': max_jump_features,
            'jump_step': jump_step,
            'baseline_outlier_features': baseline_score,
            'baseline_used': baseline_used,
            'v_rms_ramp_ratio': float(ramp_ratio) if prvi_median > 0.2 and drugi_median > 0.2 else 0.0,
            'triggered': {
                'brief_outlier': is_brief_outlier,
                'sudden_jump': is_sudden_jump,
                'baseline_outlier': is_baseline_outlier,
                'v_rms_ramp': is_v_rms_ramp,
            },
            'n_features_checked': n_features,
        }

        return is_spike, confidence, detail

    def _check_anomaly(self, voltage_features_flat, voltage_class, voltage_conf):
        """
        Detekcija anomalije za predvidenu naponsku klasu.
        Projektira 960-D vektor u PCA prostor, racuna Mahalanobis udaljenost
        do sredine predvidene klase, usporeduje s pragom te klase.

        Vraca (is_anomaly, mahalanobis_dist, threshold).
        """
        # projekcija u PCA prostor: (x - mean) @ components^T
        pca_x = (voltage_features_flat - self.pca_mean) @ self.pca_components.T
        razlika = pca_x - self.class_means_pca[voltage_class]
        udaljenost = float(np.sqrt(razlika @ self.precision_matrix @ razlika))
        prag = self.class_thresholds[voltage_class]

        # anomalija ako je predaleko od centra klase ILI ako je pouzdanost preniska
        is_anomaly = (udaljenost > prag) or (voltage_conf < self.min_confidence)
        return is_anomaly, udaljenost, prag

    def predict(self, window_np):
        # Faza 1: klasifikacija napona
        voltage_features = self._build_multiview(
            window_np,
            self.scaler_mean, self.scaler_scale,
            self.log_scaler_mean, self.log_scaler_scale
        ).reshape(1, -1)
        voltage_probs = self.voltage_model.predict(voltage_features)[0]
        voltage_idx = int(np.argmax(voltage_probs))
        voltage_conf = float(voltage_probs[voltage_idx])
        voltage_class = self.voltage_classes[voltage_idx]

        # Faza 1b: detekcija anomalije nad predvidenom naponskom klasom
        is_anomaly, mahalanobis_dist, anomaly_threshold = self._check_anomaly(
            voltage_features[0], voltage_class, voltage_conf
        )

        # Faza 2a: prvo radimo ML detekciju kvara
        fault_features = self._build_multiview(
            window_np,
            self.fault_scaler_mean, self.fault_scaler_scale,
            self.fault_log_scaler_mean, self.fault_log_scaler_scale
        ).reshape(1, -1)
        fault_probs = self.fault_model.predict(fault_features)[0]
        ml_idx = int(np.argmax(fault_probs))
        ml_conf = float(fault_probs[ml_idx])
        ml_class = self.fault_classes[ml_idx]

        # Faza 2b: multi-strategija detekcija skoka
        is_spike, spike_conf, spike_detail = self._detect_voltage_spike(window_np)

        # Odluka:
        # Ako ML predvida poznat kvar (movement/imbalance) s conf > 0.75,
        # vjerujemo ML modelu - ima eksplicitne trening podatke za to.
        # Inace, ako se okine detekcija skoka, oznaci kao voltage_spike.
        # Inace, koristi ML predikciju (najcesce 'normal').
        ml_strong_fault = (ml_class in ('fan_imbalance', 'movement') and ml_conf > 0.75)

        if is_spike and not ml_strong_fault:
            fault_status = 'voltage_spike'
            fault_conf = float(spike_conf)
        else:
            fault_status = ml_class
            fault_conf = ml_conf

        detail = {
            'voltage_probs': voltage_probs,
            'fault_probs': fault_probs,
            'ml_class': ml_class,
            'ml_conf': ml_conf,
            'is_voltage_spike': is_spike,
            'spike_confidence': float(spike_conf),
            'spike_detail': spike_detail,
            'spike_overridden_by_ml': bool(is_spike and ml_strong_fault),
            'is_anomaly': is_anomaly,
            'mahalanobis_dist': mahalanobis_dist,
            'anomaly_threshold': anomaly_threshold,
        }

        return voltage_class, voltage_conf, fault_status, fault_conf, detail

    def add_sample(self, raw_features_dict):
        """
        Dodaj jedan sirov MQTT uzorak. Vraca predikciju kad se spremnik napuni,
        inace None.
        """
        self._sample_count += 1
        sirove_vrijednosti = [float(raw_features_dict.get(kljuc, 0.0)) for kljuc in self.raw_features]
        for _, num_key, den_key in self.ratio_defs:
            num = float(raw_features_dict.get(num_key, 0.0))
            den = float(raw_features_dict.get(den_key, 0.0))
            sirove_vrijednosti.append(num / (den + 1e-12))

        self.buffer.append(sirove_vrijednosti)

        # azuriranje klizajuceg spremnika osnove (koristi Strategija C u detektoru skoka)
        self.baseline_buffer.append(sirove_vrijednosti)
        if len(self.baseline_buffer) > self.baseline_max_size:
            self.baseline_buffer.pop(0)

        if len(self.buffer) < self.window_size:
            return None

        prozor = np.array(self.buffer[-self.window_size:], dtype=np.float32)
        rezultat = self.predict(prozor)

        voltage, v_conf, fault, f_conf, detail = rezultat
        # Na naponski skok: potpuno cisti spremnik tako da se sustav ponovno napuni
        # cistim post-tranzicijskim uzorcima. Cooldown sprjecava rekurzivno
        # brisanje - post-rebuffer Strategija C detekcije (stara osnova vs novi napon)
        # se prikazuju kao skokovi ali ne pokrecu ponovni reset.
        if fault == 'voltage_spike' and (self._sample_count - self._last_spike_clear) > 20:
            self.buffer = []
            self._last_spike_clear = self._sample_count

        return rezultat


# -- CLI -------------------------------------------------------------------

def run_live_test(args):
    predictor = VoltageFaultPredictor(args.model_dir)
    print(f"Ucitan predictor (voltage + fault + multi-strategija skok) iz {args.model_dir}")
    print(f"Klase napona: {predictor.voltage_classes}")
    print(f"Klase kvarova: {predictor.fault_classes}")
    print(f"Velicina prozora: {predictor.window_size}")
    print(f"Detekcija skokova: 4 strategije (within-window MAD, sudden jump, rolling baseline, v_rms ramp)")
    print(f"Pracene znacajke skoka: {len(predictor.SPIKE_FEATURE_INDICES)} "
          f"(v_rms, a_rms, peak_peak, crest, kurtosis, impulse)")

    expected_voltage = args.expected_voltage
    expected_fault = args.expected_fault
    max_samples = args.samples

    brojaci = {
        'total': 0,
        'voltage_correct': 0,
        'fault_correct': 0,
        'spikes': 0,
        'fan_imbalance': 0,
        'movement': 0,
    }

    def process_sample(data, sample_num):
        try:
            rezultat = predictor.add_sample(data)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Sample #{sample_num}  ERROR: {e}")
            traceback.print_exc()
            return

        if rezultat is None:
            print(f"[{time.strftime('%H:%M:%S')}] Sample #{sample_num}  "
                  f"buffering {len(predictor.buffer)}/{predictor.window_size}")
            return

        voltage, v_conf, fault, f_conf, detail = rezultat
        brojaci['total'] += 1
        if expected_voltage and voltage == expected_voltage:
            brojaci['voltage_correct'] += 1
        if expected_fault and fault == expected_fault:
            brojaci['fault_correct'] += 1

        if fault == 'voltage_spike':
            brojaci['spikes'] += 1
        elif fault == 'fan_imbalance':
            brojaci['fan_imbalance'] += 1
        elif fault == 'movement':
            brojaci['movement'] += 1

        n = brojaci['total']
        G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; RST = "\033[0m"

        v_ok = voltage == expected_voltage if expected_voltage else True
        f_ok = fault == expected_fault if expected_fault else True

        v_acc = f"{brojaci['voltage_correct']/n:.1%}" if expected_voltage else "n/a"
        f_acc = f"{brojaci['fault_correct']/n:.1%}" if expected_fault else "n/a"

        spike_info = ""
        if detail['is_voltage_spike']:
            sd = detail['spike_detail']
            okidaci = [k for k, v in sd['triggered'].items() if v]
            spike_info = (f"  {Y}SPIKE{RST} "
                          f"[{', '.join(okidaci)}] "
                          f"outliers={sd['within_window_outliers']}, "
                          f"jump={sd['max_jump_features']}/{sd['n_features_checked']}")

        anomaly_info = ""
        if detail.get('is_anomaly'):
            anomaly_info = (f"  {Y}ANOMALY{RST} "
                            f"dist={detail['mahalanobis_dist']:.2f}>"
                            f"thr={detail['anomaly_threshold']:.2f}")

        print(f"\n[{time.strftime('%H:%M:%S')}] Sample #{sample_num}  "
              f"v_rms={data.get('v_rms_magnitude','?')}")
        print(f"Voltage: {G if v_ok else R}{voltage:5s}{RST} conf={v_conf:.3f} acc={v_acc}{anomaly_info}")
        print(f"Fault: {G if f_ok else R}{fault:15s}{RST} conf={f_conf:.3f} acc={f_acc}{spike_info}")

    # -- Replay nacin -----------------------------------------------------
    if args.replay:
        print(f"\nReplay iz {args.replay} ...")
        with open(args.replay) as datoteka:
            zapisi = [json.loads(redak) for redak in datoteka]

        for i, zapis in enumerate(zapisi):
            if max_samples and brojaci['total'] >= max_samples:
                break
            data = zapis
            if 'data' in data and isinstance(data['data'], dict):
                data = data['data']
            process_sample(data, i + 1)

    # -- MQTT nacin -------------------------------------------------------
    else:
        sample_num = [0]

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                client.subscribe(args.topic)
                print(f"\nSpojen na {args.broker}:{args.port}, "
                      f"pretplacen na {args.topic}")
            else:
                print(f"Spajanje neuspjesno: rc={rc}")

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload)
                data = payload.get('data', payload)
                if not isinstance(data, dict):
                    return

                sample_num[0] += 1
                process_sample(data, sample_num[0])

                if max_samples and brojaci['total'] >= max_samples:
                    client.disconnect()
            except Exception as e:
                print(f"Error: {e}")

        print(f"\nSpajanje na MQTT {args.broker}:{args.port} ...")
        if expected_voltage:
            print(f"Ocekivani napon: {expected_voltage}")
        if expected_fault:
            print(f"Ocekivani kvar: {expected_fault}")

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(args.broker, args.port, 60)

        try:
            client.loop_forever()
        except KeyboardInterrupt:
            pass

    # -- Sazetak ----------------------------------------------------------
    n = brojaci['total']
    if n > 0:
        print(f"\n{'='*60}")
        print(f"SUMMARY -- {n} predictions")
        print(f"{'='*60}")
        if expected_voltage:
            print(f"Tocnost napona: {brojaci['voltage_correct']}/{n} "
                  f"({brojaci['voltage_correct']/n:.1%})")
        if expected_fault:
            print(f"Tocnost kvarova: {brojaci['fault_correct']}/{n} "
                  f"({brojaci['fault_correct']/n:.1%})")
        print(f"Razrada kvarova:")
        print(f"Voltage skokovi: {brojaci['spikes']}/{n} ({brojaci['spikes']/n:.1%})")
        print(f"Fan imbalance: {brojaci['fan_imbalance']}/{n} ({brojaci['fan_imbalance']/n:.1%})")
        print(f"Movement: {brojaci['movement']}/{n} ({brojaci['movement']/n:.1%})")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Voltage + fault inferencija')
    parser.add_argument('--model-dir', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Putanja do model_implementation direktorija')
    parser.add_argument('--expected-voltage', type=str, default=None,
                        help='Ocekivana klasa napona za pracenje tocnosti')
    parser.add_argument('--expected-fault', type=str, default=None,
                        help='Ocekivana klasa kvara za pracenje tocnosti')
    parser.add_argument('--samples', type=int, default=50,
                        help='Broj predikcija za prikupiti')
    parser.add_argument('--replay', type=str, default=None,
                        help='Replay iz spremljene NDJSON datoteke umjesto MQTT-a')
    parser.add_argument('--broker', default='192.168.1.100', help='MQTT broker')
    parser.add_argument('--port', type=int, default=1883, help='MQTT port')
    parser.add_argument('--topic', default='sensor/mpb/1', help='MQTT topic')

    args = parser.parse_args()
    run_live_test(args)


if __name__ == '__main__':
    main()
