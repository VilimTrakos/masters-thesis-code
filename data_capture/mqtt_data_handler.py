import threading
import json
import paho.mqtt.client as mqtt
from datetime import datetime

DEFAULT_BROKER = "192.168.1.100"
DEFAULT_PORT = 1883
DEFAULT_TOPIC = "sensor/mpb/1"


class MQTTDataHandler:
    def __init__(
        self, broker, port, topic, on_message_callback=None, on_connect_callback=None
    ):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.on_message_callback = on_message_callback
        self.on_connect_callback = on_connect_callback
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.connected = False
        self.recording = False

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            client.subscribe(self.topic)
        else:
            self.connected = False

        if self.on_connect_callback:
            self.on_connect_callback(self.connected)

    def on_message(self, client, userdata, msg):
        if self.recording and self.on_message_callback:
            try:
                payload = msg.payload.decode("utf-8")
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = payload
                self.on_message_callback(data)
            except Exception as e:
                self.on_message_callback(f"Error: {e}")

    def connect(self):
        threading.Thread(target=self._connect_loop, daemon=True).start()

    def _connect_loop(self):
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_forever()
        except Exception:
            self.connected = False
            if self.on_connect_callback:
                self.on_connect_callback(False)

    def start_recording(self):
        self.recording = True

    def stop_recording(self):
        self.recording = False


class ActionDataManager:
    def __init__(self):
        self.actions_data = {}
        self.current_action = None

    def start_action_recording(self, action_name):
        self.current_action = action_name
        if action_name not in self.actions_data:
            self.actions_data[action_name] = []

    def stop_action_recording(self):
        self.current_action = None

    def add_data_point(self, data):
        if self.current_action and self.current_action in self.actions_data:
            data_with_timestamp = {
                "timestamp": datetime.now().isoformat(),
                **data,
            }
            self.actions_data[self.current_action].append(data_with_timestamp)

    def save_action_data(self, action_name):
        if action_name not in self.actions_data or not self.actions_data[action_name]:
            return None

        filename = f"{action_name}_data.json"
        with open(filename, "w") as f:
            for entry in self.actions_data[action_name]:
                f.write(json.dumps(entry) + "\n")

        self.actions_data[action_name] = []
        return filename
