#!/usr/bin/env python3
"""
Klasifikator napona s jednim LightGBM modelom i Mahalanobis detekcijom anomalija.

Multi-view arhitektura: Raw + MinMax + Log + Diff -> 960 znacajki,
plus Mahalanobis udaljenost po klasi za detekciju anomalija.

6 klasa: 3v, 6v, 9v, 12v, 15v, 18v  (+zastavica anomalije)
"""

import json, os, sys, time
import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb


WINDOW_SIZE    = 10
TRAIN_RATIO    = 0.8
SEED           = 42
N_AUGMENT      = 5

LGB_ROUNDS     = 600
LGB_EARLY_STOP = 50
N_PCA          = 50          # komponente za Mahalanobis
ANOMALY_PERCENTILE = 99      # percentil udaljenosti za prag anomalije
MIN_CONFIDENCE = 0.4         # ispod ovoga -> anomalija bez obzira na udaljenost

ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(ROOT), 'training_data')
MODEL_DIR  = ROOT

rng = np.random.RandomState(SEED)

FILES = OrderedDict([
    ('3v',  ['3v_home_data.json']),
    ('6v',  ['6v_home_data.json']),
    ('9v',  ['9v_home_data.json']),
    ('12v', ['12v_home_data.json']),
    ('15v', ['15v_home_data.json']),
    ('18v', ['18v_home_data.json']),
])

CLASSES = list(FILES.keys())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# -- Znacajke (iste 48 kao u prethodnim verzijama) -------------------------
RAW_FEATURES = [
    'dom_peak_frequency', 'dom_peak_amplitude',
    'fund_peak_frequency', 'fund_peak_amplitude',
    'v_rms_x', 'v_rms_y', 'v_rms_z', 'v_rms_magnitude',
    'a_rms_x', 'a_rms_y', 'a_rms_z', 'a_rms_magnitude',
    'peak_peak_x', 'peak_peak_y', 'peak_peak_z',
    'crest_fct_x', 'crest_fct_y', 'crest_fct_z',
    'variance_x', 'variance_y', 'variance_z',
    'skewness_x', 'skewness_y', 'skewness_z',
    'kurtosis_x', 'kurtosis_y', 'kurtosis_z',
    'shape_fct_x', 'shape_fct_y', 'shape_fct_z',
    'impulse_factor_x', 'impulse_factor_y', 'impulse_factor_z',
]

RATIO_DEFS = [
    ('v_rms_x/v_rms_mag', 'v_rms_x', 'v_rms_magnitude'),
    ('v_rms_y/v_rms_mag', 'v_rms_y', 'v_rms_magnitude'),
    ('v_rms_z/v_rms_mag', 'v_rms_z', 'v_rms_magnitude'),
    ('a_rms_x/a_rms_mag', 'a_rms_x', 'a_rms_magnitude'),
    ('a_rms_y/a_rms_mag', 'a_rms_y', 'a_rms_magnitude'),
    ('a_rms_z/a_rms_mag', 'a_rms_z', 'a_rms_magnitude'),
    ('v_mag/a_mag', 'v_rms_magnitude', 'a_rms_magnitude'),
    ('dom_amp/fund_amp', 'dom_peak_amplitude', 'fund_peak_amplitude'),
    ('dom_freq/fund_freq', 'dom_peak_frequency', 'fund_peak_frequency'),
    ('pp_x/pp_y', 'peak_peak_x', 'peak_peak_y'),
    ('pp_x/pp_z', 'peak_peak_x', 'peak_peak_z'),
    ('pp_y/pp_z', 'peak_peak_y', 'peak_peak_z'),
    ('crest_x/shape_x', 'crest_fct_x', 'shape_fct_x'),
    ('crest_y/shape_y', 'crest_fct_y', 'shape_fct_y'),
    ('crest_z/shape_z', 'crest_fct_z', 'shape_fct_z'),
]

RATIO_NAMES = [r[0] for r in RATIO_DEFS]
ALL_FEATURE_NAMES = RAW_FEATURES + RATIO_NAMES
N_RAW   = len(RAW_FEATURES)
N_RATIO = len(RATIO_DEFS)
N_TOTAL = N_RAW + N_RATIO

# -- Ucitavanje podataka ---------------------------------------------------
def load_ndjson(putanja, znacajke, omjeri):
    zapisi = []
    with open(putanja) as datoteka:
        for redak in datoteka:
            r = json.loads(redak)
            sirove_vrijednosti = [float(r.get(kljuc, 0.0)) for kljuc in znacajke]
            for _, num_key, den_key in omjeri:
                num = float(r.get(num_key, 0.0))
                den = float(r.get(den_key, 0.0))
                sirove_vrijednosti.append(num / (den + 1e-12))  # 1e-12 da se izbjegne dijeljenje s nulom
            zapisi.append(sirove_vrijednosti)
    return np.array(zapisi, dtype=np.float32)


def load_all_data():
    podaci = {}
    for klasa, lista_datoteka in FILES.items():
        arrays = []
        for naziv_datoteke in lista_datoteka:
            putanja = os.path.join(DATA_DIR, naziv_datoteke)
            if os.path.exists(putanja):
                niz = load_ndjson(putanja, RAW_FEATURES, RATIO_DEFS)
                arrays.append(niz)
            else:
                print(f"ERROR:{naziv_datoteke} nije pronadena")
        podaci[klasa] = np.concatenate(arrays, axis=0)
    return podaci


# -- Windowing & transforms ------------------------------------------------
def make_windows(data, window_size, stride=1):
    uzorci, broj_znacajki = data.shape
    broj_prozora = (uzorci - window_size) // stride + 1
    prozori = np.lib.stride_tricks.sliding_window_view(data, (window_size, broj_znacajki))
    prozori = prozori[::stride, 0, :, :]
    return prozori[:broj_prozora]


def minmax_normalize_windows(prozor):
    """Min-max normalizacija po prozoru, svaka znacajka ide na [0, 1]."""
    minimum = prozor.min(axis=1, keepdims=True)
    maksimum = prozor.max(axis=1, keepdims=True)
    return (prozor - minimum) / (maksimum - minimum + 1e-8)


def log_transform_windows(prozor):
    """Log1p transformacija s ouvanjem predznaka."""
    return np.sign(prozor) * np.log1p(np.abs(prozor))


def diff_windows(prozor):
    """Prva razlika duz vremenske osi: (N, W, F) -> (N, W-1, F)."""
    return np.diff(prozor, axis=1)


def augment_windows(prozori, oznake, n_augment, rng, scale_range=(0.85, 1.15)):
    lista_prozora = [prozori.copy()]
    lista_oznaka = [oznake.copy()]

    broj_uzoraka, velicina_prozora, broj_znacajki = prozori.shape

    for _ in range(n_augment - 1):
        w = prozori.copy()

        # korak 1: gaussov sum skaliran prema std znacajki
        std_znacajki = w.reshape(-1, broj_znacajki).std(axis=0)
        noise_scale = rng.uniform(0.02, 0.05)
        slucajni_sum = rng.randn(broj_uzoraka, velicina_prozora, broj_znacajki).astype(np.float32)
        slucajni_sum = slucajni_sum * std_znacajki * noise_scale
        w = w + slucajni_sum

        # korak 2: globalno skaliranje cijelog uzorka
        global_faktor_skaliranja = rng.uniform(scale_range[0], scale_range[1],
                                   size=(broj_uzoraka, 1, 1)).astype(np.float32)
        w = w * global_faktor_skaliranja

        # korak 3: mala promjena po znacajki
        faktor_promjene_znacajki = rng.uniform(0.92, 1.08, size=(broj_uzoraka, 1, broj_znacajki)).astype(np.float32)
        w = w * faktor_promjene_znacajki

        # korak 4: random dropout nekih znacajki (postavi na 0)
        for i in range(broj_uzoraka):
            n_droped = rng.randint(0, 3)
            if n_droped > 0:
                izbacene_znacajke = rng.choice(broj_znacajki, n_droped, replace=False)
                w[i, :, izbacene_znacajke] = 0.0

        lista_prozora.append(w)
        lista_oznaka.append(oznake.copy())

    svi_prozori = np.concatenate(lista_prozora, axis=0)
    sve_oznake = np.concatenate(lista_oznaka, axis=0)
    return svi_prozori, sve_oznake


def window_statistics(prozori):
    """(N, W, F) -> (N, 5*F): [prosjek, std, min, max, nagib]"""
    prosjek  = prozori.mean(axis=1)
    std      = prozori.std(axis=1)
    minimum  = prozori.min(axis=1)
    maksimum = prozori.max(axis=1)
    velicina_prozora = prozori.shape[1]
    vrijeme = np.arange(velicina_prozora, dtype=np.float32)
    srednje_vrijeme = vrijeme.mean()
    # nagib po metodi najmanjih kvadrata: sum((t - t_mean) * (y - y_mean)) / sum((t - t_mean)^2)
    devijacija_vremena = vrijeme - srednje_vrijeme
    varijanca_vremena = (devijacija_vremena ** 2).sum()

    devijacija_prozora = prozori - prosjek[:, None, :]
    povezanost_znacajki = (devijacija_prozora * devijacija_vremena[None, :, None]).sum(axis=1)
    nagibi = povezanost_znacajki / varijanca_vremena

    return np.hstack([prosjek, std, minimum, maksimum, nagibi])

# -- Skaliranje ------------------------------------------------------------
def scale_windows(prozori, skalar):
    broj_uzoraka, velicina_prozora, broj_znacajki = prozori.shape
    ravno = prozori.reshape(-1, broj_znacajki)
    ravno_skalirano = skalar.transform(ravno).astype(np.float32)
    return ravno_skalirano.reshape(broj_uzoraka, velicina_prozora, broj_znacajki)


# -- Ekstrakcija multi-view znacajki ---------------------------------------
def build_multiview_features(sirovi_prozori, skalar, log_skalar):
    """
    Gradi multi-view znacajke iz sirovih (neskaliranih) prozora.

    Pogledi:
      1. Raw: StandardScaler -> statistike -> 240 znacajki
      2. MinMax: po prozoru [0,1] -> statistike -> 240 znacajki
      3. Log: log1p -> poseban skalar -> statistike -> 240 znacajki
      4. Diff: prva razlika -> statistike -> 240 znacajki

    Vraca: (N, 960) matricu znacajki
    """
    # Pogled 1: sirovi (samo skaliranje)
    skalirani = scale_windows(sirovi_prozori, skalar)
    statistike_sirovih = window_statistics(skalirani)

    # Pogled 2: MinMax normalizacija
    normirani = minmax_normalize_windows(skalirani)
    statistike_minmax = window_statistics(normirani)

    # Pogled 3: Log transformacija (na sirovim podacima, poseban skalar)
    logaritmirani = log_transform_windows(sirovi_prozori)
    log_skalirani = scale_windows(logaritmirani, log_skalar)
    statistike_loga = window_statistics(log_skalirani)

    # Pogled 4: Prva razlika (na skaliranim podacima)
    razlika = diff_windows(skalirani)
    statistike_razlike = window_statistics(razlika)

    return np.hstack([statistike_sirovih, statistike_minmax, statistike_loga, statistike_razlike])


# ==========================================================================
#  MAIN
# ==========================================================================
def main():
    print("=" * 70)
    print("Multi-View LightGBM klasifikator napona")
    print("Pogledi: Raw + MinMax + Log + Diff -> 960 znacajki")
    print("=" * 70)

    # -- 1. Ucitavanje podataka -------------------------------------------
    print("\n[1/5] Ucitavanje podataka ...")
    svi_podaci = load_all_data()

    # -- 2. Podjela na train/test i kreiranje prozora ---------------------
    print("\n[2/5] Kreiranje train/test podjelu i prozore ...")
    lista_trening_prozora, lista_trening_oznaka = [], []
    lista_test_prozora, lista_test_oznaka = [], []

    for klasa, podaci in svi_podaci.items():
        indeks_klase = CLASS_TO_IDX[klasa]
        broj_trening_uzoraka = int(len(podaci) * TRAIN_RATIO)

        trening_podaci = podaci[:broj_trening_uzoraka]
        test_podaci    = podaci[broj_trening_uzoraka:]

        trening_prozori = make_windows(trening_podaci, WINDOW_SIZE, stride=1)
        test_prozori    = make_windows(test_podaci, WINDOW_SIZE, stride=1)

        lista_trening_prozora.append(trening_prozori)
        lista_trening_oznaka.append(np.full(len(trening_prozori), indeks_klase, dtype=np.int64))
        lista_test_prozora.append(test_prozori)
        lista_test_oznaka.append(np.full(len(test_prozori), indeks_klase, dtype=np.int64))

        print(f"{klasa}: trening={len(trening_prozori)}, test={len(test_prozori)}")

    X_train_seq = np.concatenate(lista_trening_prozora)
    y_train     = np.concatenate(lista_trening_oznaka)
    X_test_seq  = np.concatenate(lista_test_prozora)
    y_test      = np.concatenate(lista_test_oznaka)

    print(f"\nUkupno: trening={len(X_train_seq)}, test={len(X_test_seq)}")

    # treniranje skalara
    print("\nTreniranje skalara ...")
    ravni_trening = X_train_seq.reshape(-1, N_TOTAL)
    skalar = StandardScaler()
    skalar.fit(ravni_trening)
    print(f"[debug] skalar.mean_ shape: {skalar.mean_.shape}, N_TOTAL={N_TOTAL}")

    # log skalar: treniram na log-transformiranim sirovim podacima
    log_trening = log_transform_windows(X_train_seq)
    log_skalar = StandardScaler()
    log_skalar.fit(log_trening.reshape(-1, N_TOTAL))

    # -- 3. Augmentacija i izgradnja multi-view znacajki ------------------
    print("\n[3/5] Augmentiranje i gradnja multi-view znacajki ...")
    rng_augmentacija = np.random.RandomState(SEED)
    X_train_aug, y_train_aug = augment_windows(
        X_train_seq, y_train, N_AUGMENT, rng_augmentacija, scale_range=(0.85, 1.15))

    print(f"Augmentirano: {len(X_train_seq)} -> {len(X_train_aug)} prozora")

    X_train_mv = build_multiview_features(X_train_aug, skalar, log_skalar)

    X_test_mv = build_multiview_features(X_test_seq, skalar, log_skalar)

    # -- 4. Treniranje jednog LightGBM modela -----------------------------
    print(f"\n[4/5]Treniranje LightGBM ({X_train_mv.shape[1]} znacajki) ...")

    tezine_uzoraka = compute_sample_weight('balanced', y_train_aug)

    lgb_trening = lgb.Dataset(X_train_mv, label=y_train_aug, weight=tezine_uzoraka)
    lgb_validacija = lgb.Dataset(X_test_mv, label=y_test, reference=lgb_trening)

    lgb_params = {
        'objective': 'multiclass',
        'num_class': len(CLASSES),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'max_depth': 8,
        'learning_rate': 0.03,
        'feature_fraction': 0.3,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'seed': SEED,
    }

    model = lgb.train(
        lgb_params,
        lgb_trening,
        num_boost_round=LGB_ROUNDS,
        valid_sets=[lgb_validacija],
        callbacks=[lgb.early_stopping(LGB_EARLY_STOP), lgb.log_evaluation(50)],
    )

    predikcije = model.predict(X_test_mv)
    tocnost = (predikcije.argmax(1) == y_test).mean()
    print(f"\nTocnost na test skupu: {tocnost:.4f}")

    print("Tocnost po klasi:")
    for indeks_klase, klasa in enumerate(CLASSES):
        maska = y_test == indeks_klase
        tocnost_klase = (predikcije[maska].argmax(1) == indeks_klase).mean()
        print(f"{klasa}: {tocnost_klase:.4f}")

    # -- 4b. Mahalanobis detekcija anomalija (na ne-augmentiranom treningu) --
    print("\n[4b/5] Racunanje statistike za Mahalanobis detekciju anomalija ...")

    # multi-view znacajke iz ne-augmentiranih trening prozora
    X_train_mv_cisti = build_multiview_features(X_train_seq, skalar, log_skalar)

    # PCA - analiza glavnih komponenti za smanjenje dimenzionalnosti (960 -> N_PCA)
    pca = PCA(n_components=N_PCA, random_state=SEED)
    X_pca = pca.fit_transform(X_train_mv_cisti)
    print(f"PCA smanjio {X_train_mv_cisti.shape[1]} znacajki na {N_PCA}," 
          f" zadrzano {pca.explained_variance_ratio_.sum():.2%} informacije")

    # srednja vrijednost po klasi u PCA prostoru
    srednje_vrijednosti_klasa = {}
    uzorci_klasa = {}
    for indeks_klase, klasa in enumerate(CLASSES):
        maska = y_train == indeks_klase
        pca_klase = X_pca[maska]
        srednje_vrijednosti_klasa[klasa] = pca_klase.mean(axis=0)
        uzorci_klasa[klasa] = pca_klase

    # dijeljena povezanosti znacajki iz svih trening podataka
    matrica_povezanost_znacajki = np.cov(X_pca, rowvar=False)
    # regularizacija za osiguranje inverzije
    matrica_povezanost_znacajki += np.eye(N_PCA) * 1e-6
    matrica_preciznosti = np.linalg.inv(matrica_povezanost_znacajki)

    # Mahalanobis udaljenost svakog trening uzorka do vlastite klase
    print("Pragove udaljenosti iz trening podataka ...")
    pragovi_klasa = {}
    for indeks_klase, klasa in enumerate(CLASSES):
        uzorci_pca = uzorci_klasa[klasa]
        razlika = uzorci_pca - srednje_vrijednosti_klasa[klasa]
        # Mahalanobis: sqrt((x-mu)^T @ preciznost @ (x-mu)) @ - mnozenje matrica 
        udaljenosti = np.sqrt(np.sum(razlika @ matrica_preciznosti * razlika, axis=1))
        prag = np.percentile(udaljenosti, ANOMALY_PERCENTILE)
        srednja_udaljenost = udaljenosti.mean()
        std_udaljenosti = udaljenosti.std()
        pragovi_klasa[klasa] = float(prag)
        print(f"{klasa}: srednja_udaljenost={srednja_udaljenost:.2f}, std={std_udaljenosti:.2f}, "
              f"p{ANOMALY_PERCENTILE} prag={prag:.2f}")

    # provjera tocnosti s detekcijom anomalija na test skupu
    X_test_pca = pca.transform(X_test_mv)
    broj_anomalija_test = 0
    for i in range(len(X_test_pca)):
        indeks_predikcije = int(predikcije[i].argmax())
        klasa_predikcije = CLASSES[indeks_predikcije]
        razlika = X_test_pca[i] - srednje_vrijednosti_klasa[klasa_predikcije]
        udaljenost = np.sqrt(razlika @ matrica_preciznosti @ razlika)
        pouzdanost = float(predikcije[i].max())
        if udaljenost > pragovi_klasa[klasa_predikcije] or pouzdanost < MIN_CONFIDENCE:
            broj_anomalija_test += 1
    print(f"Test uzorci oznaceni kao anomalija: {broj_anomalija_test}/{len(X_test_pca)} "
          f"({broj_anomalija_test/len(X_test_pca):.1%})")

    # -- 5. Izvoz modela i konfiguracije -----------------------------------
    print("\n[5/5] Spremanje modela ...")

    model_path = os.path.join(MODEL_DIR, 'fan_lightgbm.txt')
    model.save_model(model_path)
    print(f"Spremljeno {model_path}")

    nazivi_pogleda = ['raw', 'minmax', 'log', 'diff']

    ensemble_cfg = {
        'version': 'voltage_classifier',
        'ensemble_type': 'multiview_single_lightgbm',
        'classes': CLASSES,
        'raw_features': RAW_FEATURES,
        'ratio_features': RATIO_NAMES,
        'ratio_defs': [[naziv, brojnik, nazivnik] for naziv, brojnik, nazivnik in RATIO_DEFS],
        'window_size': WINDOW_SIZE,
        'n_raw_features': N_RAW,
        'n_ratio_features': N_RATIO,
        'n_total_features': N_TOTAL,
        'scaler_mean': skalar.mean_.tolist(),
        'scaler_scale': skalar.scale_.tolist(),
        'log_scaler_mean': log_skalar.mean_.tolist(),
        'log_scaler_scale': log_skalar.scale_.tolist(),
        'model_file': 'fan_lightgbm.txt',
        'views': nazivi_pogleda,
        'stat_names': ['mean', 'std', 'min', 'max', 'slope'],
        'n_features_per_view': N_TOTAL * 5,
        'n_lgb_features': N_TOTAL * 5 * 4,
        'test_accuracy': float(tocnost),
        # parametri detekcije anomalija
        'anomaly_detection': {
            'n_pca_components': N_PCA,
            'pca_components': pca.components_.tolist(),
            'pca_mean': pca.mean_.tolist(),
            'precision_matrix': matrica_preciznosti.tolist(),
            'class_means_pca': {klasa: srednje_vrijednosti_klasa[klasa].tolist() for klasa in CLASSES},
            'class_thresholds': pragovi_klasa,
            'min_confidence': MIN_CONFIDENCE,
            'anomaly_percentile': ANOMALY_PERCENTILE,
        },
    }
    putanja_konfiguracije = os.path.join(MODEL_DIR, 'model_config.json')
    with open(putanja_konfiguracije, 'w') as datoteka:
        json.dump(ensemble_cfg, datoteka, indent=2)
    print(f"Spremljeno {putanja_konfiguracije}")

    # -- Sazetak ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAZETAK REZULTATA")
    print("=" * 70)
    print(f"Tocnost LightGBM-a na test skupu: {tocnost:.4f}")
    print(f"Pogledi: {nazivi_pogleda}")
    print(f"Ukupno znacajki: {X_train_mv.shape[1]}")
    print(f"Detekcija anomalija: PCA({N_PCA}), Mahalanobis, "
          f"min_pouzdanost={MIN_CONFIDENCE}")
    print(f"Pragovi anomalija (p{ANOMALY_PERCENTILE}):")
    for klasa in CLASSES:
        print(f"{klasa}: {pragovi_klasa[klasa]:.2f}")
    print(f"\nDatoteke spremljene u: {MODEL_DIR}")
    print("=" * 70)

    # brisanje starih datoteka trojnog modela ako postoje
    for stara_datoteka in ['fan_lightgbm_raw.txt', 'fan_lightgbm_norm.txt',
                           'fan_lightgbm_log.txt']:
        stara_putanja = os.path.join(MODEL_DIR, stara_datoteka)
        if os.path.exists(stara_putanja):
            os.remove(stara_putanja)


if __name__ == '__main__':
    main()
