import csv
import json
import sys
import traceback
import zipfile
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2
from PyQt5.QtCore import (
    QDateTime,
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDateTimeEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class ConfigError(Exception):
    """Raised when db.txt cannot be parsed into a valid export configuration."""


@dataclass
class DatabaseSettings:
    host: str
    port: int
    dbname: str
    user: str
    password: str = ""

    def to_dsn_kwargs(self) -> Dict[str, str]:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password,
        }


@dataclass
class ExportQuery:
    key: str
    sql: str
    filename: str


@dataclass
class ExportConfig:
    database: DatabaseSettings
    queries: List[ExportQuery]
    output_root: Path


def load_export_config(config_path: Path) -> ExportConfig:
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ConfigError("Configuration file is empty.")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Configuration must be valid JSON. Parsing failed: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ConfigError("Configuration root element must be a JSON object.")

    database_section = parsed.get("database")
    if not isinstance(database_section, dict):
        raise ConfigError("Configuration must contain a 'database' object.")

    required_db_keys = {"host", "port", "dbname", "user"}
    missing_db_keys = required_db_keys - database_section.keys()
    if missing_db_keys:
        raise ConfigError(
            f"Database configuration is missing keys: {', '.join(sorted(missing_db_keys))}"
        )

    try:
        port_value = int(database_section["port"])
    except (TypeError, ValueError) as exc:
        raise ConfigError("Database 'port' must be an integer.") from exc

    db_settings = DatabaseSettings(
        host=str(database_section["host"]),
        port=port_value,
        dbname=str(database_section["dbname"]),
        user=str(database_section["user"]),
        password=str(database_section.get("password", "")),
    )

    queries_section = parsed.get("queries")
    if not isinstance(queries_section, dict) or not queries_section:
        raise ConfigError(
            "Configuration must contain a non-empty 'queries' object mapping export keys to SQL."
        )

    export_queries: List[ExportQuery] = []
    for key, value in queries_section.items():
        if isinstance(value, dict):
            sql_text = value.get("sql")
            filename = value.get("filename", key)
        else:
            sql_text = value
            filename = key

        if not isinstance(sql_text, str) or not sql_text.strip():
            raise ConfigError(f"Query '{key}' must provide a non-empty SQL string.")

        cleaned_filename = str(filename).strip() or key
        if not cleaned_filename.lower().endswith(".csv"):
            cleaned_filename = f"{cleaned_filename}.csv"

        export_queries.append(
            ExportQuery(key=key, sql=sql_text.strip(), filename=cleaned_filename)
        )

    output_root_value = parsed.get("output_root")
    if output_root_value:
        output_root = Path(str(output_root_value)).expanduser().resolve()
    else:
        output_root = Path.cwd() / "exports"

    return ExportConfig(
        database=db_settings,
        queries=export_queries,
        output_root=output_root,
    )


def ensure_unique_path(base_path: Path) -> Path:
    candidate = base_path
    suffix = 1
    suffix_str = "".join(base_path.suffixes)
    stem_name = base_path.name[: len(base_path.name) - len(suffix_str)] if suffix_str else base_path.name
    while candidate.exists():
        if suffix_str:
            candidate = base_path.parent / f"{stem_name}_{suffix}{suffix_str}"
        else:
            candidate = base_path.parent / f"{stem_name}_{suffix}"
        suffix += 1
    return candidate


def export_queries_to_csv(
    start_dt: datetime,
    stop_dt: datetime,
    config: ExportConfig,
    signals: "ExportWorkerSignals",
) -> Tuple[Path, Path]:
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    folder_name = stop_dt.strftime("%Y%m%d_%H%M%S")
    export_dir = ensure_unique_path(output_root / folder_name)
    export_dir.mkdir(parents=True, exist_ok=False)

    signals.log.emit(f"Connecting to database {config.database.dbname}@{config.database.host}")
    with closing(psycopg2.connect(**config.database.to_dsn_kwargs())) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            for query in config.queries:
                signals.log.emit(f"Executing query '{query.key}'")
                sql_lower = query.sql.lower()
                params: Dict[str, datetime] = {}
                if "%(start" in sql_lower:
                    params["start"] = start_dt
                if "%(stop" in sql_lower:
                    params["stop"] = stop_dt
                print(start_dt)
                print(stop_dt)

                if params:
                    cursor.execute(query.sql, params)
                else:
                    cursor.execute(query.sql)

                column_names = [
                    getattr(col, "name", col[0]) for col in (cursor.description or [])
                ]
                csv_path = export_dir / query.filename

                row_count = 0
                with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.writer(csv_file)
                    if column_names:
                        writer.writerow(column_names)

                    while True:
                        rows = cursor.fetchmany(size=1000)
                        if not rows:
                            break
                        writer.writerows(rows)
                        row_count += len(rows)

                signals.log.emit(
                    f"Exported {row_count} rows to {csv_path.name}"
                )

    zip_path = export_dir.with_suffix(".zip")
    zip_path = ensure_unique_path(zip_path)

    signals.log.emit(f"Compressing export directory into {zip_path.name}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in export_dir.rglob("*"):
            if file_path.is_file():
                zip_file.write(file_path, arcname=file_path.relative_to(export_dir))

    return export_dir, zip_path


class ExportWorkerSignals(QObject):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    log = pyqtSignal(str)


class ExportWorker(QRunnable):
    def __init__(self, start_dt: datetime, stop_dt: datetime, config: ExportConfig):
        super().__init__()
        self.start_dt = start_dt
        self.stop_dt = stop_dt
        self.config = config
        self.signals = ExportWorkerSignals()

    def run(self) -> None:  # type: ignore[override]
        try:
            export_dir, zip_path = export_queries_to_csv(
                self.start_dt, self.stop_dt, self.config, self.signals
            )
        except Exception as exc:  # pragma: no cover - GUI surface
            error_text = "".join(
                traceback.format_exception(exc.__class__, exc, exc.__traceback__)
            )
            self.signals.error.emit(error_text)
        else:
            self.signals.finished.emit(str(export_dir), str(zip_path))


class ExportWindow(QMainWindow):
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Database Exporter")
        self.setMinimumSize(640, 420)

        self.thread_pool = QThreadPool.globalInstance()
        if config_path is None:
            self.config_path = self._discover_config_path()
        else:
            self.config_path = config_path.expanduser().resolve()

        self._now_timer = QTimer(self)
        self._now_timer.setInterval(1000)
        self._now_timer.timeout.connect(self._sync_stop_with_now)

        self._build_ui()
        self._set_defaults()

    def _discover_config_path(self) -> Path:
        candidates = [
            Path(__file__).with_name("db_export_config.json"),
            Path(__file__).with_name("db.txt"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        layout.addWidget(self._build_time_group())

        self.status_label = QLabel("Ready", self)
        layout.addWidget(self.status_label)

        self.log_output = QPlainTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Export logs will appear here...")
        layout.addWidget(self.log_output, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.export_button = QPushButton("Export to CSV", self)
        self.export_button.clicked.connect(self._on_export_clicked)
        button_row.addWidget(self.export_button)
        layout.addLayout(button_row)

        self.setCentralWidget(central_widget)

    def _build_time_group(self) -> QGroupBox:
        group = QGroupBox("Time Range", self)
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.start_edit = QDateTimeEdit(self)
        self.start_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.start_edit.setCalendarPopup(True)

        self.stop_edit = QDateTimeEdit(self)
        self.stop_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.stop_edit.setCalendarPopup(True)

        self.use_current_checkbox = QCheckBox("Use current time as stop", self)
        self.use_current_checkbox.stateChanged.connect(self._on_use_current_changed)

        form.addRow("Start:", self.start_edit)
        form.addRow("Stop:", self.stop_edit)
        form.addRow("", self.use_current_checkbox)
        return group

    def _set_defaults(self) -> None:
        now_dt = QDateTime.currentDateTime()
        self.start_edit.setDateTime(now_dt.addDays(-1))
        self.stop_edit.setDateTime(now_dt)

    def _sync_stop_with_now(self) -> None:
        if self.use_current_checkbox.isChecked():
            self.stop_edit.setDateTime(QDateTime.currentDateTime())

    def _on_use_current_changed(self, state: int) -> None:
        use_now = state == Qt.Checked
        self.stop_edit.setDisabled(use_now)
        if use_now:
            self._sync_stop_with_now()
            self._now_timer.start()
        else:
            self._now_timer.stop()

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")

    def _on_export_clicked(self) -> None:
        try:
            config = load_export_config(self.config_path)
        except ConfigError as exc:
            self._show_error(str(exc))
            return

        start_dt = self.start_edit.dateTime().toPyDateTime()
        if self.use_current_checkbox.isChecked():
            stop_dt = datetime.now()
        else:
            stop_dt = self.stop_edit.dateTime().toPyDateTime()

        if start_dt >= stop_dt:
            self._show_error("Start datetime must be earlier than stop datetime.")
            return

        self.export_button.setDisabled(True)
        self.status_label.setText("Running export...")
        self._append_log(
            f"Starting export window from {start_dt} to {stop_dt}"
        )

        worker = ExportWorker(start_dt=start_dt, stop_dt=stop_dt, config=config)
        worker.signals.log.connect(self._append_log)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_worker_finished)
        self.thread_pool.start(worker)

    def _on_worker_error(self, details: str) -> None:
        self.export_button.setEnabled(True)
        self.status_label.setText("Export failed")
        self._append_log(details)
        self._show_error("Export failed. Check the log for details.")

    def _on_worker_finished(self, export_dir: str, zip_path: str) -> None:
        self.export_button.setEnabled(True)
        self.status_label.setText("Export completed")
        self._append_log(f"Export finished. Folder: {export_dir}, Zip: {zip_path}")
        QMessageBox.information(
            self,
            "Export completed",
            f"Data exported to:\n{export_dir}\n\nZip archive created:\n{zip_path}",
        )

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


def run_app() -> None:
    app = QApplication(sys.argv)
    config_arg: Optional[Path] = None
    if len(sys.argv) > 1:
        config_arg = Path(sys.argv[1]).expanduser().resolve()

    window = ExportWindow(config_path=config_arg)
    window.show()
    app.exec_()


if __name__ == "__main__":
    run_app()
