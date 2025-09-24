import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2 import OperationalError
from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDateTimeEdit,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import matplotlib.pyplot as plt

from vchstat import Model2d3d, q0_by_wpnadev

# Database connection parameters mirrored from plotter.py
DB_HOST = "192.168.120.100"
DB_NAME = "spacedb"
DB_USER = "postgres"
DB_PASSWORD = ""

PAIR_ID_MEAS1_REF = 29
PAIR_ID_MEAS2_REF = 30
MEAS2_TIME_SHIFT_SECONDS = 10
CONFIG_PATH = Path(__file__).with_name("ks_tester_config.json")


class KSTesterWindow(QMainWindow):
    """Main window that manages measurement parameters and DB connection."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KS Tester")
        self.setMinimumSize(640, 520)

        self._connection = None
        self.generated_time = None
        self.generated_phase = {}

        self._build_ui()
        self._load_config()

    def _build_ui(self):
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        self.status_label = QLabel("Not connected", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addWidget(self._build_time_range_box())
        layout.addWidget(self._build_signal_box("Reference", "ref"))
        layout.addWidget(self._build_signal_box("Measurement 1", "meas1"))
        layout.addWidget(self._build_signal_box("Measurement 2", "meas2"))

        self.generate_button = QPushButton("Generate measurements", self)
        self.generate_button.clicked.connect(self._on_generate_measurements)
        layout.addWidget(self.generate_button)

        self.delete_button = QPushButton("Delete measurements from DB", self)
        self.delete_button.clicked.connect(self._on_delete_measurements)
        layout.addWidget(self.delete_button)

        layout.addStretch()
        self.setCentralWidget(central)

    def _build_time_range_box(self):
        box = QGroupBox("Time Range", self)
        form = QFormLayout(box)

        self.start_datetime = QDateTimeEdit(QDateTime.currentDateTime(), self)
        self.start_datetime.setCalendarPopup(True)
        form.addRow("Start:", self.start_datetime)

        self.stop_datetime = QDateTimeEdit(QDateTime.currentDateTime(), self)
        self.stop_datetime.setCalendarPopup(True)
        form.addRow("Stop:", self.stop_datetime)

        self.dt_edit = self._create_line_edit("100")
        form.addRow("Interval dt (s):", self.dt_edit)

        return box

    def _build_signal_box(self, title, key):
        box = QGroupBox(f"{title} Signal", self)
        form = QFormLayout(box)

        init_phase = self._create_line_edit()
        form.addRow("Initial phase:", init_phase)

        init_freq = self._create_line_edit()
        form.addRow("Initial frequency:", init_freq)

        freq_drift = self._create_line_edit()
        form.addRow("Frequency drift:", freq_drift)

        setattr(
            self,
            f"{key}_controls",
            {
                "init_phase": init_phase,
                "init_frequency": init_freq,
                "frequency_drift": freq_drift,
            },
        )

        return box

    @staticmethod
    def _create_line_edit(default_text="0"):
        line_edit = QLineEdit()
        line_edit.setText(default_text)
        line_edit.setAlignment(Qt.AlignRight)
        return line_edit

    def _parse_float(self, widget, field_label):
        text = widget.text().strip().replace(",", ".")
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", f"{field_label} must be numeric.")
            widget.setFocus()
            return None

    def _connect_to_database(self):
        """Ensure a database connection exists before generating measurements."""
        if self._connection is not None and self._connection.closed == 0:
            return True

        try:
            self._connection = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            self.status_label.setText("Connected to spacedb")
            return True
        except OperationalError as exc:
            self.status_label.setText("Database connection failed")
            QMessageBox.critical(
                self,
                "Database Error",
                f"Could not connect to the database.\n\n{exc}",
            )
            return False

    def _get_signal_parameters(self, key):
        controls = getattr(self, f"{key}_controls")
        params = {}
        for label, widget in controls.items():
            value = self._parse_float(widget, f"{key} {label.replace('_', ' ')}")
            if value is None:
                return None
            params[label] = value
        return params

    def _config_snapshot(self):
        signals = {}
        for key in ("ref", "meas1", "meas2"):
            controls = getattr(self, f"{key}_controls")
            signals[key] = {
                label: ctrl.text().strip()
                for label, ctrl in controls.items()
            }

        return {
            "start_datetime": self.start_datetime.dateTime().toString(Qt.ISODate),
            "stop_datetime": self.stop_datetime.dateTime().toString(Qt.ISODate),
            "dt": self.dt_edit.text().strip(),
            "signals": signals,
        }

    def _load_config(self):
        if not CONFIG_PATH.exists():
            return

        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to load config: {exc}")
            return

        start_str = data.get("start_datetime")
        if start_str:
            dt = QDateTime.fromString(start_str, Qt.ISODate)
            if dt.isValid():
                self.start_datetime.setDateTime(dt)

        stop_str = data.get("stop_datetime")
        if stop_str:
            dt = QDateTime.fromString(stop_str, Qt.ISODate)
            if dt.isValid():
                self.stop_datetime.setDateTime(dt)

        dt_text = data.get("dt")
        if dt_text is not None:
            self.dt_edit.setText(str(dt_text))

        signals = data.get("signals", {})
        for key in ("ref", "meas1", "meas2"):
            controls = getattr(self, f"{key}_controls")
            for label, ctrl in controls.items():
                value = signals.get(key, {}).get(label)
                if value is not None:
                    ctrl.setText(str(value))

    def _save_config(self):
        try:
            CONFIG_PATH.write_text(
                json.dumps(self._config_snapshot(), indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"Failed to save config: {exc}")

    def _on_generate_measurements(self):
        start_dt = self.start_datetime.dateTime().toPyDateTime()
        stop_dt = self.stop_datetime.dateTime().toPyDateTime()

        if stop_dt <= start_dt:
            QMessageBox.warning(
                self,
                "Invalid Time Range",
                "Stop date/time must be after the start date/time.",
            )
            return

        dt_value = self._parse_float(self.dt_edit, "Interval dt")
        if dt_value is None or dt_value <= 0:
            QMessageBox.warning(
                self,
                "Invalid Interval",
                "Sample interval (dt) must be a positive number.",
            )
            return

        total_seconds = (stop_dt - start_dt).total_seconds()
        N = int(total_seconds // dt_value)
        if N <= 0:
            QMessageBox.warning(
                self,
                "Invalid Range",
                "Selected time range does not contain even one dt interval.",
            )
            return

        signal_configs = [
            ("ref", "Reference"),
            ("meas1", "Measurement 1"),
            ("meas2", "Measurement 2"),
        ]

        signal_params = {}
        for key, _ in signal_configs:
            params = self._get_signal_parameters(key)
            if params is None:
                return
            signal_params[key] = params

        self._save_config()

        q0 = q0_by_wpnadev(1, 8e-14)
        q = np.array([q0])

        self.generated_phase = {}
        self.generated_time = np.arange(N, dtype=float) * dt_value

        for key, _ in signal_configs:
            params = signal_params[key]
            freq = params["init_frequency"] #or 2e-15
            drift = params["frequency_drift"] / 86400. #or (5.0e-16 / 86400.0)
            phase0 = params["init_phase"]

            model = Model2d3d(q, dt_value, phase0, freq, drift)
            self.generated_phase[key] = model.generate(N)

        timestamps_meas1 = [start_dt + timedelta(seconds=i * dt_value) for i in range(N)]
        timestamps_meas2 = [
            start_dt + timedelta(seconds=i * dt_value + MEAS2_TIME_SHIFT_SECONDS)
            for i in range(N)
        ]

        phase_ref = self.generated_phase["ref"]
        phase_diff_meas1 = self.generated_phase["meas1"] - phase_ref
        phase_diff_meas2 = self.generated_phase["meas2"] - phase_ref

        print(f"Generated N = {N} samples with dt = {dt_value} s for 3 signals")
        self.status_label.setText(
            f"Generated {N} samples for ref/meas1/meas2 (dt = {dt_value:g} s)"
        )

        plt.figure("Generated Phase")
        plt.clf()
        for key, label in signal_configs:
            plt.plot(
                self.generated_time,
                self.generated_phase[key],
                linestyle="-",
                label=label,
            )
        plt.title("Generated Phase vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

        if not self._connect_to_database():
            return

        insert_sql = "INSERT INTO raw_phase (timestamp, phase, pair_id) VALUES (%s, %s, %s)"
        delete_sql = "DELETE FROM raw_phase WHERE pair_id = %s AND timestamp BETWEEN %s AND %s"

        records = []
        for ts, phase in zip(timestamps_meas1, phase_diff_meas1):
            records.append((ts, float(phase), PAIR_ID_MEAS1_REF))
        for ts, phase in zip(timestamps_meas2, phase_diff_meas2):
            records.append((ts, float(phase), PAIR_ID_MEAS2_REF))

        start_range_meas1 = timestamps_meas1[0]
        end_range_meas1 = timestamps_meas1[-1]
        start_range_meas2 = timestamps_meas2[0]
        end_range_meas2 = timestamps_meas2[-1]

        try:
            with self._connection.cursor() as cur:
                cur.execute(
                    delete_sql,
                    (PAIR_ID_MEAS1_REF, start_range_meas1, end_range_meas1),
                )
                cur.execute(
                    delete_sql,
                    (PAIR_ID_MEAS2_REF, start_range_meas2, end_range_meas2),
                )
                cur.executemany(insert_sql, records)
            self._connection.commit()
        except Exception as exc:  # pylint: disable=broad-except
            if self._connection is not None:
                self._connection.rollback()
            QMessageBox.critical(
                self,
                "Database Error",
                f"Failed to save generated data to the database.\n\n{exc}",
            )
            self.close_db_connection()
            return

        self.close_db_connection()
        self.status_label.setText(
            f"Generated and saved {N} samples for pair IDs 27 & 28 (connection closed)"
        )

    def _on_delete_measurements(self):
        self._save_config()

        if not self._connect_to_database():
            return

        delete_sql = "DELETE FROM raw_phase WHERE pair_id IN (%s, %s)"

        try:
            with self._connection.cursor() as cur:
                cur.execute(delete_sql, (PAIR_ID_MEAS1_REF, PAIR_ID_MEAS2_REF))
            self._connection.commit()
            self.status_label.setText("Deleted measurements for pair IDs 27 & 28")
        except Exception as exc:  # pylint: disable=broad-except
            if self._connection is not None:
                self._connection.rollback()
            QMessageBox.critical(
                self,
                "Database Error",
                f"Failed to delete measurements from the database.\n\n{exc}",
            )
        finally:
            self.close_db_connection()

    def close_db_connection(self):
        """Close the database connection if it is open."""
        if self._connection is not None and self._connection.closed == 0:
            self._connection.close()
        self._connection = None

    def closeEvent(self, event):
        self._save_config()
        self.close_db_connection()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = KSTesterWindow()
    app.aboutToQuit.connect(window.close_db_connection)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
