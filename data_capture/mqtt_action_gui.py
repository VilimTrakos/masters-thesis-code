import tkinter as tk
from tkinter import ttk, messagebox
from mqtt_data_handler import (
    MQTTDataHandler,
    ActionDataManager,
    DEFAULT_BROKER,
    DEFAULT_PORT,
    DEFAULT_TOPIC,
)


class ActionRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MQTT Action Recorder")
        self.actions = []
        self.mqtt_handler = None
        self.data_manager = ActionDataManager()
        self.current_action_name = None
        self.recording = False
        self._build_gui()

    def add_action(self):
        frame = ttk.Frame(self.actions_frame)
        frame.pack(fill="x", pady=2)

        name_var = tk.StringVar()
        duration_var = tk.StringVar(value="0")

        ttk.Label(frame, text="Action Name:").pack(side="left", padx=2)
        name_entry = ttk.Entry(frame, textvariable=name_var, width=16)
        name_entry.pack(side="left", padx=2)

        ttk.Label(frame, text="Duration (s):").pack(side="left", padx=2)
        duration_entry = ttk.Entry(frame, textvariable=duration_var, width=6)
        duration_entry.pack(side="left", padx=2)

        record_btn = ttk.Button(
            frame,
            text="Start Recording",
            style="SaveRecord.TButton",
            command=lambda: self.toggle_recording(name_var, duration_var, record_btn),
        )
        record_btn.pack(side="left", padx=2)

        countdown_var = tk.StringVar(value="")
        countdown_label = ttk.Label(
            frame, textvariable=countdown_var, foreground="red", width=6
        )
        countdown_label.pack(side="left", padx=2)

        self.actions.append(
            {
                "name_var": name_var,
                "duration_var": duration_var,
                "countdown_var": countdown_var,
            }
        )

    def connect_mqtt(self):
        broker = self.broker_var.get()
        port = int(self.port_var.get())
        topic = self.topic_var.get()

        self.mqtt_handler = MQTTDataHandler(
            broker,
            port,
            topic,
            on_message_callback=self.on_mqtt_message,
            on_connect_callback=self.on_mqtt_connect,
        )
        self.mqtt_handler.connect()
        self.status_label.config(text="Connecting...", foreground="orange")

    def on_mqtt_connect(self, connected):
        def _update():
            if connected:
                self.status_label.config(text="Connected", foreground="green")
            else:
                self.status_label.config(text="Not connected", foreground="red")
        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def on_mqtt_message(self, data):
        if self.current_action_name:
            self.data_manager.add_data_point(data)

    def _build_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        recorder_tab = ttk.Frame(self.notebook)
        self.notebook.add(recorder_tab, text="Recorder")

        mqtt_frame = ttk.LabelFrame(recorder_tab, text="MQTT Connection")
        mqtt_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(mqtt_frame, text="Broker:").grid(row=0, column=0)
        self.broker_var = tk.StringVar(value=DEFAULT_BROKER)
        ttk.Entry(mqtt_frame, textvariable=self.broker_var, width=18).grid(
            row=0, column=1
        )
        ttk.Label(mqtt_frame, text="Port:").grid(row=0, column=2)
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        ttk.Entry(mqtt_frame, textvariable=self.port_var, width=6).grid(row=0, column=3)
        ttk.Label(mqtt_frame, text="Topic:").grid(row=0, column=4)
        self.topic_var = tk.StringVar(value=DEFAULT_TOPIC)
        ttk.Entry(mqtt_frame, textvariable=self.topic_var, width=18).grid(
            row=0, column=5
        )
        ttk.Button(mqtt_frame, text="Connect", command=self.connect_mqtt).grid(
            row=0, column=6, padx=5
        )
        self.status_label = ttk.Label(
            mqtt_frame, text="Not connected", foreground="red"
        )
        self.status_label.grid(row=0, column=7, padx=5)

        self.actions_frame = ttk.LabelFrame(recorder_tab, text="Actions")
        self.actions_frame.pack(fill="both", expand=True, padx=10, pady=5)
        ttk.Button(
            self.actions_frame, text="+ Add Action", command=self.add_action
        ).pack(anchor="w", pady=2)

    def save_action(self, name_var):
        action_name = name_var.get()
        if not action_name:
            messagebox.showerror("Error", "Action name cannot be empty")
            return

        filename = self.data_manager.save_action_data(action_name)
        if filename:
            messagebox.showinfo("Saved", f"Data saved to {filename}")
        else:
            messagebox.showwarning("Warning", "No data to save for this action")

    def toggle_recording(self, name_var, duration_var, btn):
        if not self.mqtt_handler or not self.mqtt_handler.connected:
            messagebox.showerror("MQTT not connected", "Please connect to MQTT first.")
            return

        action_name = name_var.get()
        if not action_name:
            messagebox.showerror("Error", "Please enter an action name")
            return

        action = next((a for a in self.actions if a["name_var"] == name_var), None)

        if not self.recording:
            self.current_action_name = action_name
            self.data_manager.start_action_recording(action_name)
            self.mqtt_handler.start_recording()
            btn.config(text="Stop & Save", style="Recording.TButton")
            self.recording = True

            try:
                duration = float(duration_var.get())
                if duration > 0:
                    self._start_countdown(action, duration)
                    self.root.after(
                        int(duration * 1000),
                        lambda: self.toggle_recording(name_var, duration_var, btn),
                    )
                else:
                    action["countdown_var"].set("")
            except ValueError:
                action["countdown_var"].set("")
        else:
            self.mqtt_handler.stop_recording()
            self.data_manager.stop_action_recording()
            btn.config(text="Start Recording", style="SaveRecord.TButton")
            self.recording = False
            self.current_action_name = None
            action["countdown_var"].set("")
            self.save_action(name_var)

    def _start_countdown(self, action, duration):
        def update_countdown(remaining):
            if not self.recording:
                action["countdown_var"].set("")
                return
            if remaining <= 0:
                action["countdown_var"].set("Done")
                return
            action["countdown_var"].set(f"{int(remaining)}s")
            self.root.after(1000, lambda: update_countdown(remaining - 1))

        update_countdown(duration)


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    style.configure(
        "SaveRecord.TButton", foreground="black", background="#90EE90"
    )  # light green
    style.map(
        "SaveRecord.TButton",
        background=[("active", "#90EE90"), ("!active", "#90EE90")],
        foreground=[("active", "black"), ("!active", "black")],
    )
    style.configure(
        "Recording.TButton", foreground="white", background="#8B0000"
    )  # dark red
    style.map(
        "Recording.TButton",
        background=[("active", "#8B0000"), ("!active", "#8B0000")],
        foreground=[("active", "white"), ("!active", "white")],
    )
    app = ActionRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
