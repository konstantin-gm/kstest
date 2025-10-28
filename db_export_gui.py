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

import pandas as pd
import psycopg2
from PyQt5.QtCore import (
    QAbstractTableModel,
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
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableView,
    QTabWidget,
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
        raise ConfigError(f"Файл конфигурации не найден: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ConfigError("Файл конфигурации пуст.")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Конфигурация должна быть корректным JSON. Ошибка разбора: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ConfigError("Корневой элемент конфигурации должен быть объектом JSON.")

    database_section = parsed.get("database")
    if not isinstance(database_section, dict):
        raise ConfigError("Конфигурация должна содержать объект 'database'.")

    required_db_keys = {"host", "port", "dbname", "user"}
    missing_db_keys = required_db_keys - database_section.keys()
    if missing_db_keys:
        raise ConfigError(
            f"В конфигурации базы данных отсутствуют ключи: {', '.join(sorted(missing_db_keys))}"
        )

    try:
        port_value = int(database_section["port"])
    except (TypeError, ValueError) as exc:
        raise ConfigError("Параметр 'port' базы данных должен быть целым числом.") from exc

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
            "Конфигурация должна содержать непустой объект 'queries', связывающий ключи выгрузок с SQL."
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
            raise ConfigError(f"Запрос '{key}' должен содержать непустую строку SQL.")

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

    signals.log.emit(f"Подключение к базе данных {config.database.dbname}@{config.database.host}")
    with closing(psycopg2.connect(**config.database.to_dsn_kwargs())) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            for query in config.queries:
                signals.log.emit(f"Выполнение запроса '{query.key}'")
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
                    f"Экспортировано {row_count} строк в {csv_path.name}"
                )

    zip_path = export_dir.with_suffix(".zip")
    zip_path = ensure_unique_path(zip_path)

    signals.log.emit(f"Сжатие каталога выгрузки в архив {zip_path.name}")
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


class DataFrameModel(QAbstractTableModel):
    def __init__(self, frame: pd.DataFrame):
        super().__init__()
        self._frame = frame

    def rowCount(self, parent=None):  # type: ignore[override]
        return len(self._frame.index)

    def columnCount(self, parent=None):  # type: ignore[override]
        return len(self._frame.columns)

    def data(self, index, role=Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._frame.iat[index.row(), index.column()]
            if pd.isna(value):
                return ""
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # type: ignore[override]
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._frame.columns[section])
        return str(self._frame.index[section])

    def flags(self, index):  # type: ignore[override]
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable


class DataFrameTableWindow(QDialog):
    def __init__(self, frame: pd.DataFrame, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Просмотр CSV")
        self.resize(900, 600)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        self.table_view = QTableView(self)
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self._model = DataFrameModel(frame)
        self.table_view.setModel(self._model)

        layout.addWidget(self.table_view)


class ExportWindow(QMainWindow):
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Экспорт данных")
        self.setMinimumSize(640, 420)

        self.thread_pool = QThreadPool.globalInstance()
        if config_path is None:
            self.config_path = self._discover_config_path()
        else:
            self.config_path = config_path.expanduser().resolve()

        self._now_timer = QTimer(self)
        self._now_timer.setInterval(1000)
        self._now_timer.timeout.connect(self._sync_stop_with_now)

        self._current_output_root: Path = self._default_output_root()
        self._table_windows: List[QWidget] = []
        self._analysis_last_path: Path = self._current_output_root

        self._build_ui()
        self._set_defaults()

    def _discover_config_path(self) -> Path:
        candidate_names = ["db_export_config.json", "db.txt"]
        search_roots = []

        if hasattr(sys, "_MEIPASS"):
            search_roots.append(Path(getattr(sys, "_MEIPASS")))

        try:
            search_roots.append(Path(__file__).resolve().parent)
        except NameError:
            pass

        if hasattr(sys, "executable"):
            search_roots.append(Path(sys.executable).resolve().parent)

        search_roots.append(Path.cwd())
        search_roots.append(Path("/usr/share/db-exporter"))

        seen = set()
        ordered_roots = []
        for root in search_roots:
            if root in seen:
                continue
            seen.add(root)
            ordered_roots.append(root)

        for root in ordered_roots:
            for name in candidate_names:
                candidate = root / name
                if candidate.exists():
                    return candidate

        return ordered_roots[0] / candidate_names[0]

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.addTab(self._build_export_tab(), "Выгрузка")
        self.tab_widget.addTab(self._build_analysis_tab(), "Анализ")

        outer_layout.addWidget(self.tab_widget)
        self.setCentralWidget(central_widget)

    def _build_time_group(self) -> QGroupBox:
        group = QGroupBox("Интервал времени", self)
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.start_edit = QDateTimeEdit(self)
        self.start_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.start_edit.setCalendarPopup(True)

        self.stop_edit = QDateTimeEdit(self)
        self.stop_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.stop_edit.setCalendarPopup(True)

        self.use_current_checkbox = QCheckBox("Использовать текущее время как конец", self)
        self.use_current_checkbox.stateChanged.connect(self._on_use_current_changed)

        form.addRow("Начало:", self.start_edit)
        form.addRow("Конец:", self.stop_edit)
        form.addRow("", self.use_current_checkbox)
        return group

    def _build_export_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        layout.addWidget(self._build_time_group())
        layout.addWidget(self._build_output_group())

        self.status_label = QLabel("Готово", self)
        layout.addWidget(self.status_label)

        self.log_output = QPlainTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Журнал выгрузок будет отображаться здесь...")
        layout.addWidget(self.log_output, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.export_button = QPushButton("Экспорт в CSV", self)
        self.export_button.clicked.connect(self._on_export_clicked)
        button_row.addWidget(self.export_button)
        layout.addLayout(button_row)

        return tab

    def _build_analysis_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        file_row = QHBoxLayout()
        self.analysis_file_edit = QLineEdit(self)
        self.analysis_file_edit.setReadOnly(True)
        self.analysis_file_edit.setPlaceholderText("Выберите файл CSV для просмотра")
        self.analysis_file_edit.textChanged.connect(self._update_show_table_button_state)
        file_row.addWidget(self.analysis_file_edit, stretch=1)

        self.analysis_select_button = QPushButton("Обзор...", self)
        self.analysis_select_button.clicked.connect(self._on_select_analysis_file)
        file_row.addWidget(self.analysis_select_button)

        layout.addLayout(file_row)

        self.show_table_button = QPushButton("Показать таблицу", self)
        self.show_table_button.setEnabled(False)
        self.show_table_button.clicked.connect(self._on_show_table_clicked)
        layout.addWidget(self.show_table_button)

        layout.addStretch(1)
        return tab

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Каталог вывода", self)
        layout = QHBoxLayout(group)

        self.output_path_edit = QLineEdit(self)
        self.output_path_edit.setReadOnly(True)
        layout.addWidget(self.output_path_edit, stretch=1)

        self.output_select_button = QPushButton("Выбрать папку", self)
        self.output_select_button.clicked.connect(self._on_select_output_folder)
        layout.addWidget(self.output_select_button)

        return group

    def _set_defaults(self) -> None:
        now_dt = QDateTime.currentDateTime()
        self.start_edit.setDateTime(now_dt.addDays(-1))
        self.stop_edit.setDateTime(now_dt)
        self._load_output_root_setting()
        self._update_show_table_button_state()

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

    def _default_output_root(self) -> Path:
        return Path.cwd() / "exports"

    def _update_show_table_button_state(self) -> None:
        if not hasattr(self, "show_table_button"):
            return
        has_path = bool(self.analysis_file_edit.text().strip())
        self.show_table_button.setEnabled(has_path)

    def _on_select_analysis_file(self) -> None:
        start_dir = self._analysis_last_path
        if not start_dir.exists():
            start_dir = self._current_output_root
        if not start_dir.exists():
            start_dir = Path.home()

        selected_file, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл CSV",
            str(start_dir),
            "Файлы CSV (*.csv);;Все файлы (*)",
        )
        if not selected_file:
            return

        csv_path = Path(selected_file).expanduser().resolve()
        self.analysis_file_edit.setText(str(csv_path))
        self.analysis_file_edit.setToolTip(str(csv_path))
        self._analysis_last_path = csv_path.parent
        self._update_show_table_button_state()
        self._append_log(f"Выбран файл для анализа: {csv_path}")

    def _on_show_table_clicked(self) -> None:
        file_text = self.analysis_file_edit.text().strip()
        if not file_text:
            self._show_error("Выберите файл CSV для отображения.")
            return

        csv_path = Path(file_text)
        if not csv_path.exists():
            self._show_error("Выбранный файл CSV не существует.")
            return

        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            self._show_error(f"Не удалось прочитать файл CSV:\n{exc}")
            return

        table_window = DataFrameTableWindow(frame, self)
        table_window.setWindowTitle(f"Просмотр CSV - {csv_path.name}")
        self._register_table_window(table_window)
        table_window.show()
        self._analysis_last_path = csv_path.parent
        self._append_log(f"Открыто окно просмотра для {csv_path}")

    def _register_table_window(self, window: QWidget) -> None:
        self._table_windows.append(window)

        def _cleanup(_obj: QObject) -> None:
            if window in self._table_windows:
                self._table_windows.remove(window)

        window.destroyed.connect(_cleanup)

    def _load_output_root_setting(self) -> None:
        output_root = self._read_output_root_from_config()
        self._current_output_root = output_root
        self._analysis_last_path = output_root
        self.output_path_edit.setText(str(output_root))
        self.output_path_edit.setToolTip(str(output_root))

    def _read_output_root_from_config(self) -> Path:
        default_root = self._default_output_root()
        try:
            if not self.config_path.exists():
                return default_root
            raw_text = self.config_path.read_text(encoding="utf-8")
        except OSError:
            return default_root

        if not raw_text.strip():
            return default_root

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return default_root

        if not isinstance(parsed, dict):
            return default_root

        value = parsed.get("output_root")
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser().resolve()

        return default_root

    def _save_output_root_setting(self, selected_path: Path) -> bool:
        if not self.config_path.exists():
            self._show_error(f"Файл конфигурации не найден: {self.config_path}")
            return False

        try:
            raw_text = self.config_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._show_error(f"Не удалось прочитать файл конфигурации: {exc}")
            return False

        if not raw_text.strip():
            self._show_error("Файл конфигурации пуст, невозможно сохранить каталог вывода.")
            return False

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self._show_error(f"Файл конфигурации имеет неверный формат JSON: {exc}")
            return False

        if not isinstance(parsed, dict):
            self._show_error("Файл конфигурации должен содержать объект JSON.")
            return False

        parsed["output_root"] = str(selected_path)

        try:
            self.config_path.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            self._show_error(f"Не удалось записать файл конфигурации: {exc}")
            return False

        return True

    def _on_select_output_folder(self) -> None:
        start_dir = str(self._current_output_root)
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Выберите папку для вывода", start_dir
        )
        if not selected_dir:
            return

        new_path = Path(selected_dir).expanduser().resolve()
        if new_path == self._current_output_root:
            return

        if self._save_output_root_setting(new_path):
            self._current_output_root = new_path
            self._analysis_last_path = new_path
            new_path_str = str(new_path)
            self.output_path_edit.setText(new_path_str)
            self.output_path_edit.setToolTip(new_path_str)
            self._append_log(f"Каталог вывода установлен: {new_path_str}")

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
            self._show_error("Время начала должно быть раньше времени окончания.")
            return

        self.export_button.setDisabled(True)
        self.status_label.setText("Выполняется выгрузка...")
        self._append_log(
            f"Запуск выгрузки за период с {start_dt} по {stop_dt}"
        )

        worker = ExportWorker(start_dt=start_dt, stop_dt=stop_dt, config=config)
        worker.signals.log.connect(self._append_log)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_worker_finished)
        self.thread_pool.start(worker)

    def _on_worker_error(self, details: str) -> None:
        self.export_button.setEnabled(True)
        self.status_label.setText("Выгрузка завершилась с ошибкой")
        self._append_log(details)
        self._show_error("Выгрузка не выполнена. Подробности смотрите в журнале.")

    def _on_worker_finished(self, export_dir: str, zip_path: str) -> None:
        self.export_button.setEnabled(True)
        self.status_label.setText("Выгрузка завершена")
        self._append_log(f"Выгрузка завершена. Папка: {export_dir}, Архив: {zip_path}")
        QMessageBox.information(
            self,
            "Выгрузка завершена",
            f"Данные экспортированы в:\n{export_dir}\n\nСоздан архив:\n{zip_path}",
        )

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)


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
