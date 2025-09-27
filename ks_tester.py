import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2 import OperationalError
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDateTimeEdit,
    QFormLayout,
    QGridLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
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

        self._real_time_models = {}
        self._real_time_next_timestamp = None
        self._real_time_dt = None

        self.real_time_timer = QTimer(self)
        self.real_time_timer.setSingleShot(False)
        self.real_time_timer.timeout.connect(self._on_real_time_timer_tick)

        self._build_ui()
        self._apply_theme()
        self._load_config()


    def _build_ui(self):
        central = QWidget(self)
        central.setObjectName("CentralWidget")

        layout = QVBoxLayout(central)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(20)

        layout.addWidget(self._build_header())
        layout.addWidget(self._build_status_card())
        layout.addWidget(self._build_time_range_box())
        layout.addWidget(self._build_signal_box("Reference", "ref"))
        layout.addWidget(self._build_signal_box("Measurement 1", "meas1"))
        layout.addWidget(self._build_signal_box("Measurement 2", "meas2"))

        self.generate_button = QPushButton("Generate measurements", self)
        self.generate_button.setObjectName("PrimaryButton")
        self.generate_button.setCursor(Qt.PointingHandCursor)
        self.generate_button.clicked.connect(self._on_generate_measurements)

        self.delete_button = QPushButton("Delete measurements from DB", self)
        self.delete_button.setObjectName("DangerButton")
        self.delete_button.setCursor(Qt.PointingHandCursor)
        self.delete_button.clicked.connect(self._on_delete_measurements)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        button_row.addStretch()
        button_row.addWidget(self.generate_button)
        button_row.addWidget(self.delete_button)

        layout.addStretch()
        layout.addLayout(button_row)

        self.setCentralWidget(central)

    def _build_header(self):
        header = QFrame(self)
        header.setObjectName("HeaderCard")

        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 18, 20, 20)
        header_layout.setSpacing(6)

        title = QLabel("KS Tester", header)
        title.setObjectName("HeaderTitle")
        title_font = QFont(self.font())
        title_font.setPointSize(title_font.pointSize() + 8)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QLabel(
            "Generate synthetic phase signals and manage database uploads.",
            header,
        )
        subtitle.setObjectName("HeaderSubtitle")
        subtitle.setWordWrap(True)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)

        return header

    def _build_status_card(self):
        card = QFrame(self)
        card.setObjectName("StatusCard")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 12, 18, 18)
        card_layout.setSpacing(6)

        title = QLabel("Current status", card)
        title.setObjectName("StatusTitle")

        self.status_label = QLabel("Not connected", card)
        self.status_label.setObjectName("StatusValue")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.status_label.setWordWrap(True)

        card_layout.addWidget(title)
        card_layout.addWidget(self.status_label)

        return card

    def _apply_theme(self):
        base_font = QFont(self.font())
        if base_font.pointSize() < 10:
            base_font.setPointSize(10)
        self.setFont(base_font)

        self.setStyleSheet(
            '''
            QWidget#CentralWidget {
                background-color: #f5f7fb;
            }
            QFrame#HeaderCard,
            QFrame#StatusCard,
            QGroupBox {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #d7dce5;
            }
            QFrame#HeaderCard {
                border: none;
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2d6cdf, stop:1 #3556b1
                );
            }
            QFrame#HeaderCard QLabel {
                color: #ffffff;
            }
            QLabel#HeaderTitle {
                font-size: 22px;
                font-weight: 600;
                margin-bottom: 4px;
            }
            QLabel#HeaderSubtitle {
                font-size: 12px;
                color: rgba(255, 255, 255, 210);
            }
            QLabel#StatusTitle {
                font-size: 11px;
                font-weight: 600;
                color: #52697a;
            }
            QLabel#StatusValue {
                font-size: 14px;
                font-weight: 500;
            }
            QGroupBox {
                padding-top: 20px;
                margin-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #2b3750;
                font-weight: 600;
            }
            QLineEdit,
            QDateTimeEdit {
                border: 1px solid #c6ccd8;
                border-radius: 6px;
                padding: 6px 8px;
                background-color: #fdfdff;
            }
            QLineEdit:focus,
            QDateTimeEdit:focus {
                border: 1px solid #2d6cdf;
            }
            QDateTimeEdit::up-button,
            QDateTimeEdit::down-button {
                width: 16px;
            }
            QPushButton {
                padding: 10px 18px;
                border-radius: 8px;
                border: none;
                font-weight: 600;
                background-color: #3e4c59;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #52697a;
            }
            QPushButton#PrimaryButton {
                background-color: #2d6cdf;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #264da8;
            }
            QPushButton#DangerButton {
                background-color: #d64545;
            }
            QPushButton#DangerButton:hover {
                background-color: #b23a3a;
            }
            QPushButton:pressed {
                background-color: #1f2933;
            }
            '''
        )
    def _build_time_range_box(self):
        box = QGroupBox("Time Range", self)
        grid = QGridLayout(box)
        grid.setSpacing(10)        
        grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.start_datetime = QDateTimeEdit(QDateTime.currentDateTime(), self)
        self.start_datetime.setCalendarPopup(True)
        grid.addWidget(QLabel("Start:"), 0, 0)
        grid.addWidget(self.start_datetime, 0, 1)

        self.real_time_checkbox = QCheckBox("Real-time", self)
        self.real_time_checkbox.toggled.connect(self._on_real_time_toggled)
        grid.addWidget(self.real_time_checkbox, 0, 2)

        self.stop_datetime = QDateTimeEdit(QDateTime.currentDateTime(), self)
        self.stop_datetime.setCalendarPopup(True)
        grid.addWidget(QLabel("Stop:"), 0, 3)
        grid.addWidget(self.stop_datetime, 0, 4)

        self.dt_edit = self._create_line_edit("100")
        grid.addWidget(QLabel("Interval dt (s):"), 0, 5)
        grid.addWidget(self.dt_edit, 0, 6)

        self._on_real_time_toggled(self.real_time_checkbox.isChecked())

        return box

    def _build_signal_box(self, title, key):
        box = QGroupBox(f"{title} Signal", self)
        grid = QGridLayout(box)
        grid.setSpacing(10)        
        grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        init_phase = self._create_line_edit()
        grid.addWidget(QLabel("Initial phase:"), 0, 0)
        grid.addWidget(init_phase, 0, 1)

        init_freq = self._create_line_edit()        
        grid.addWidget(QLabel("Initial frequency:"), 0, 2)
        grid.addWidget(init_freq, 0, 3)

        freq_drift = self._create_line_edit()
        grid.addWidget(QLabel("Initial drift:"), 0, 4)
        grid.addWidget(freq_drift, 0, 5)

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
            "real_time": self.real_time_checkbox.isChecked(),
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

        real_time_enabled = data.get("real_time")
        if real_time_enabled is not None:
            self.real_time_checkbox.setChecked(bool(real_time_enabled))

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

    def _on_real_time_toggled(self, checked):
        self.stop_datetime.setDisabled(checked)
        if checked:
            self.stop_datetime.setDateTime(QDateTime.currentDateTime())
        else:
            self._stop_real_time_generation()

    def _stop_real_time_generation(self):
        if getattr(self, "real_time_timer", None) is not None and self.real_time_timer.isActive():
            self.real_time_timer.stop()
        self._real_time_models = {}
        self._real_time_next_timestamp = None
        self._real_time_dt = None
        if self._connection is not None and self._connection.closed == 0:
            self.close_db_connection()

    def _start_real_time_generation(self, models, dt_value, last_timestamp):
        if not models:
            return
        self._real_time_models = models
        self._real_time_dt = dt_value
        self._real_time_next_timestamp = last_timestamp + timedelta(seconds=dt_value)
        interval_ms = max(1, int(round(dt_value * 1000)))
        self.real_time_timer.start(interval_ms)

    def _on_real_time_timer_tick(self):
        if not self._real_time_models or self._real_time_dt is None or self._real_time_next_timestamp is None:
            self._stop_real_time_generation()
            return

        phases = {}
        for key, model in self._real_time_models.items():
            sample = model.generate(1)
            phases[key] = float(sample[0]) if len(sample) else 0.0

        phase_ref = phases.get("ref")
        if phase_ref is None:
            self._stop_real_time_generation()
            return

        diff_meas1 = phases.get("meas1", 0.0) - phase_ref
        diff_meas2 = phases.get("meas2", 0.0) - phase_ref

        meas1_timestamp = self._real_time_next_timestamp
        meas2_timestamp = meas1_timestamp + timedelta(seconds=MEAS2_TIME_SHIFT_SECONDS)

        if not self._connect_to_database():
            self._stop_real_time_generation()
            self.status_label.setText("Real-time generation stopped: database unavailable")
            return

        insert_sql = "INSERT INTO raw_phase (timestamp, phase, pair_id) VALUES (%s, %s, %s)"
        records = [
            (meas1_timestamp, float(diff_meas1), PAIR_ID_MEAS1_REF),
            (meas2_timestamp, float(diff_meas2), PAIR_ID_MEAS2_REF),
        ]

        try:
            with self._connection.cursor() as cur:
                cur.executemany(insert_sql, records)
            self._connection.commit()
        except Exception as exc:  # pylint: disable=broad-except
            if self._connection is not None:
                self._connection.rollback()
            QMessageBox.critical(
                self,
                "Database Error",
                f"Failed to append real-time data to the database.\n\n{exc}",
            )
            self._stop_real_time_generation()
            return

        self._real_time_next_timestamp = meas1_timestamp + timedelta(seconds=self._real_time_dt)
        self.status_label.setText(
            f"Real-time mode active. Last saved sample at {meas1_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def _on_generate_measurements(self):
        self._stop_real_time_generation()

        real_time_mode = self.real_time_checkbox.isChecked()

        start_dt = self.start_datetime.dateTime().toPyDateTime()
        if real_time_mode:
            current_qdt = QDateTime.currentDateTime()
            self.stop_datetime.setDateTime(current_qdt)
            stop_dt = current_qdt.toPyDateTime()
        else:
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

        realtime_models = {}
        for key, _ in signal_configs:
            params = signal_params[key]
            freq = params["init_frequency"] #or 2e-15
            drift = params["frequency_drift"] / 86400. #or (5.0e-16 / 86400.0)
            phase0 = params["init_phase"]

            model = Model2d3d(q, dt_value, phase0, freq, drift)
            self.generated_phase[key] = model.generate(N)
            if real_time_mode:
                realtime_models[key] = model

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

        if real_time_mode:
            self._start_real_time_generation(
                realtime_models,
                dt_value,
                timestamps_meas1[-1],
            )
            self.status_label.setText(
                f"Real-time mode active after seeding {N} samples (dt = {dt_value:g} s)"
            )
        else:
            self.close_db_connection()
            self.status_label.setText(
                f"Generated and saved {N} samples for pair IDs {PAIR_ID_MEAS1_REF} & {PAIR_ID_MEAS2_REF} (connection closed)"
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
            self.status_label.setText(
                f"Deleted measurements for pair IDs {PAIR_ID_MEAS1_REF} & {PAIR_ID_MEAS2_REF}"
            )
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
        self._stop_real_time_generation()
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
