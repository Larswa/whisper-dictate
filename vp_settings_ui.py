"""Optional PySide/Qt settings UI for whisper-dictate."""
from __future__ import annotations

import sys
from pathlib import Path

from vp_config import SETTING_BY_KEY, config_path, effective_config, load_config, save_config, touch_reload_signal
from vp_parakeet import DEFAULT_MODEL as DEFAULT_PARAKEET_MODEL


def _missing_pyside_error() -> RuntimeError:
    return RuntimeError(
        "The settings UI requires PySide6. Install the optional UI bundle: "
        "python -m pip install -r requirements-ui.txt"
    )


def run_settings_ui() -> int:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QAction, QIcon
        from PySide6.QtWidgets import (
            QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
            QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
            QPushButton, QSpinBox, QDoubleSpinBox, QSystemTrayIcon,
            QTabWidget, QTextEdit, QVBoxLayout, QWidget,
        )
    except ImportError as exc:
        raise _missing_pyside_error() from exc

    from vp_dictionary import dictionary_target_path, ensure_dictionary_file, open_dictionary

    class SettingsWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("whisper-dictate settings")
            self.resize(760, 560)
            self._controls: dict[str, object] = {}
            self._status = QLabel("")
            self._build()
            self._load()

        def _build(self) -> None:
            tabs = QTabWidget()
            tabs.addTab(self._build_core_tab(), "Core")
            tabs.addTab(self._build_quality_tab(), "Quality")
            tabs.addTab(self._build_dictionary_tab(), "Dictionary")
            tabs.addTab(self._build_output_tab(), "Output")

            save_btn = QPushButton("Save")
            save_btn.clicked.connect(self.save)
            reload_btn = QPushButton("Signal reload")
            reload_btn.clicked.connect(self.signal_reload)

            buttons = QHBoxLayout()
            buttons.addWidget(save_btn)
            buttons.addWidget(reload_btn)
            buttons.addStretch(1)
            buttons.addWidget(self._status)

            root = QVBoxLayout()
            root.addWidget(tabs)
            root.addLayout(buttons)
            w = QWidget()
            w.setLayout(root)
            self.setCentralWidget(w)

        def _combo(self, key: str, values: list[str], editable: bool = False) -> QComboBox:
            c = QComboBox()
            c.addItems(values)
            c.setEditable(editable)
            self._controls[key] = c
            return c

        def _line(self, key: str) -> QLineEdit:
            c = QLineEdit()
            self._controls[key] = c
            return c

        def _check(self, key: str) -> QCheckBox:
            c = QCheckBox()
            self._controls[key] = c
            return c

        def _spin(self, key: str, minimum: int, maximum: int) -> QSpinBox:
            c = QSpinBox()
            c.setRange(minimum, maximum)
            self._controls[key] = c
            return c

        def _dspin(self, key: str, minimum: float, maximum: float, step: float = 0.1) -> QDoubleSpinBox:
            c = QDoubleSpinBox()
            c.setRange(minimum, maximum)
            c.setSingleStep(step)
            c.setDecimals(2)
            self._controls[key] = c
            return c

        def _build_core_tab(self) -> QWidget:
            form = QFormLayout()
            form.addRow("STT backend", self._combo("stt_backend", ["whisper", "parakeet"]))
            form.addRow("Whisper model", self._combo("model", [
                "large-v3-turbo", "large-v3", "distil-large-v3", "medium", "small", "base", "tiny"
            ], editable=True))
            form.addRow("Parakeet model", self._line("parakeet_model"))
            form.addRow("Device", self._combo("device", ["auto", "cuda", "cpu"]))
            form.addRow("Compute type", self._combo("compute_type", [
                "", "int8_float16", "float16", "bfloat16", "float32", "int8"
            ], editable=True))
            form.addRow("Language", self._combo("lang", ["", "da", "en", "de", "fr", "sv", "nb", "nl", "es", "it"], editable=True))
            form.addRow("Hotkey", self._line("key"))
            return self._wrap(form)

        def _build_quality_tab(self) -> QWidget:
            form = QFormLayout()
            form.addRow("Beam size", self._spin("beam_size", 1, 16))
            form.addRow("Temperature ladder", self._line("temperature"))
            form.addRow("Context min seconds", self._dspin("context_min_seconds", 0, 60, 0.5))
            form.addRow("VAD threshold", self._dspin("vad_threshold", 0, 1, 0.05))
            form.addRow("VAD min silence ms", self._spin("vad_min_silence_ms", 0, 5000))
            form.addRow("Target dBFS", self._dspin("target_dbfs", -60, 0, 1))
            form.addRow("Min input dBFS", self._dspin("min_input_dbfs", -90, 0, 1))
            form.addRow("Min SNR dB", self._dspin("min_snr_db", 0, 80, 1))
            prompt = QTextEdit()
            prompt.setMaximumHeight(90)
            self._controls["initial_prompt"] = prompt
            form.addRow("Initial prompt", prompt)
            return self._wrap(form)

        def _build_dictionary_tab(self) -> QWidget:
            form = QFormLayout()
            row = QHBoxLayout()
            path = self._line("dictionary")
            browse = QPushButton("Browse")
            browse.clicked.connect(self._browse_dictionary)
            open_btn = QPushButton("Open")
            open_btn.clicked.connect(lambda: open_dictionary(Path(path.text()) if path.text() else None))
            row.addWidget(path)
            row.addWidget(browse)
            row.addWidget(open_btn)
            form.addRow("Dictionary path", row)
            form.addRow("Dictionary enabled", self._check("dictionary_enabled"))
            form.addRow("Max prompt terms", self._spin("dictionary_max_terms", 0, 500))
            form.addRow("Prompt char cap", self._spin("dictionary_prompt_chars", 0, 10000))
            return self._wrap(form)

        def _build_output_tab(self) -> QWidget:
            form = QFormLayout()
            form.addRow("Inject mode", self._combo("inject_mode", ["auto", "type", "paste", "print"]))
            form.addRow("JSON stdout", self._check("json_output"))
            form.addRow("Metrics JSONL", self._line("metrics_jsonl"))
            form.addRow("VOICEPI_DEBUG", self._check("debug"))
            form.addRow("VOICEPI_STT_DEBUG", self._check("stt_debug"))
            return self._wrap(form)

        def _wrap(self, layout) -> QWidget:
            w = QWidget()
            outer = QVBoxLayout()
            outer.addLayout(layout)
            outer.addStretch(1)
            w.setLayout(outer)
            return w

        def _browse_dictionary(self) -> None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Dictionary JSON", str(dictionary_target_path()), "JSON (*.json);;All files (*)")
            if path:
                self._controls["dictionary"].setText(path)  # type: ignore[attr-defined]

        def _load(self) -> None:
            values = effective_config()
            if not values.get("dictionary"):
                values["dictionary"] = str(dictionary_target_path())
            if not values.get("parakeet_model"):
                values["parakeet_model"] = DEFAULT_PARAKEET_MODEL
            for key, control in self._controls.items():
                value = values.get(key, "")
                if isinstance(control, QComboBox):
                    idx = control.findText(str(value))
                    if idx >= 0:
                        control.setCurrentIndex(idx)
                    else:
                        control.setEditText(str(value))
                elif isinstance(control, QLineEdit):
                    control.setText(str(value))
                elif isinstance(control, QCheckBox):
                    control.setChecked(str(value).lower() not in ("", "0", "false", "no", "off"))
                elif isinstance(control, QSpinBox):
                    control.setValue(int(float(value or 0)))
                elif isinstance(control, QDoubleSpinBox):
                    control.setValue(float(value or 0))
                elif isinstance(control, QTextEdit):
                    control.setPlainText(str(value))
            self._status.setText(f"Config: {config_path()}")

        def _collect(self) -> dict[str, str]:
            out: dict[str, str] = {}
            for key, control in self._controls.items():
                if isinstance(control, QComboBox):
                    value = control.currentText().strip()
                elif isinstance(control, QLineEdit):
                    value = control.text().strip()
                elif isinstance(control, QCheckBox):
                    value = "1" if control.isChecked() else "0"
                elif isinstance(control, QSpinBox):
                    value = str(control.value())
                elif isinstance(control, QDoubleSpinBox):
                    value = str(control.value())
                elif isinstance(control, QTextEdit):
                    value = control.toPlainText().strip()
                else:
                    continue
                if key == "parakeet_model" and value == DEFAULT_PARAKEET_MODEL:
                    continue
                if key in SETTING_BY_KEY and value != "":
                    out[key] = value
            return out

        def save(self) -> None:
            data = load_config()
            collected = self._collect()
            if "parakeet_model" not in collected and data.get("parakeet_model") == DEFAULT_PARAKEET_MODEL:
                data.pop("parakeet_model", None)
            data.update(collected)
            path = save_config(data)
            ensure_dictionary_file(Path(data["dictionary"])) if data.get("dictionary") else ensure_dictionary_file()
            self.signal_reload(show=False)
            self._status.setText(f"Saved: {path}")

        def signal_reload(self, show: bool = True) -> None:
            path = touch_reload_signal()
            if show:
                QMessageBox.information(self, "Reload signalled", f"Runtime reload signal written:\n{path}")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    win = SettingsWindow()
    tray = QSystemTrayIcon(QIcon(), app)
    tray.setToolTip("whisper-dictate")
    menu = tray.contextMenu()
    if menu is None:
        from PySide6.QtWidgets import QMenu
        menu = QMenu()
        tray.setContextMenu(menu)
    show_action = QAction("Settings", tray)
    show_action.triggered.connect(lambda: (win.show(), win.raise_(), win.activateWindow()))
    dict_action = QAction("Open dictionary", tray)
    dict_action.triggered.connect(lambda: open_dictionary())
    quit_action = QAction("Quit UI", tray)
    quit_action.triggered.connect(app.quit)
    menu.addAction(show_action)
    menu.addAction(dict_action)
    menu.addSeparator()
    menu.addAction(quit_action)
    tray.show()
    win.show()
    return int(app.exec())
