# gui.py

import csv
import math
import sys
import time as _pytime
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from threading import Event
from typing import List, Tuple, Optional, Dict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QListWidget, QTextEdit, QFileDialog, QListWidgetItem,
    QHBoxLayout, QVBoxLayout, QGroupBox, QSizePolicy, QTimeEdit,
    QMenu, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGridLayout
)
from PySide6.QtGui import QPixmap, QImage, QColor, QBrush
from PySide6.QtCore import Qt, Signal, QTime, QPoint, QThread, QSettings, QTimer, Slot

from backend import (
    process_video_parallel,
    AnalysisCancelled,
    save_emotion_map,
    load_emotion_map_ext,
)
from comparison import (
    METHODS, METHOD_TITLES, METHOD_COMBINED, RECOMMENDED_WEIGHTS,
    RECOMMENDED_SHIFT_SECONDS, RECOMMENDED_SKIP_NEUTRAL,
    CompareParams, prepare_map, compare_prepared, format_value, verdict
)

ALL_METHODS_KEY = "all"


@dataclass
class LoadedMap:
    name: str
    labels: List[Tuple[str, str]]
    fps: int
    item: QListWidgetItem
    probs: Optional[List[List[float]]] = None


class VideoAnalysisWorker(QThread):
    """Поток координирует пул процессов; результат читается после finished."""

    def __init__(self, video_path, workers=2, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.workers = workers
        self.cancel_event = Event()
        self.result = None
        self.error_message = None

    def cancel(self):
        # Потокобезопасный сигнал, не зависящий от цикла событий QThread.
        self.cancel_event.set()

    def run(self):
        try:
            self.result = process_video_parallel(
                self.video_path,
                workers=self.workers,
                cancel_event=self.cancel_event
            )
        except AnalysisCancelled:
            self.cancel_event.set()
        except Exception as e:
            self.error_message = str(e)


class VideoPreview(QLabel):
    video_changed = Signal(object)  # emits Path or None

    def __init__(self):
        super().__init__("Перетащите или нажмите для выбора видео")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self._path = None

        self.clear_btn = QPushButton("✕", self)
        self.clear_btn.setFixedSize(20, 20)
        self.clear_btn.clicked.connect(self._on_clear)
        self.clear_btn.hide()

    def resizeEvent(self, event):
        self.clear_btn.move(self.width() - 25, 5)
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        settings = QSettings("Emotionalyze", "Emotionalyze")

        last_dir = settings.value("last_video_dir", str(Path.home()))

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видео",
            last_dir,
            "Видео (*.mp4 *.avi *.mov)"
        )

        if file_path:
            settings.setValue("last_video_dir", str(Path(file_path).parent))
            self.load_video(Path(file_path))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        url = event.mimeData().urls()[0].toLocalFile()
        self.load_video(Path(url))

    def load_video(self, path: Path):
        from cv2 import VideoCapture, cvtColor, COLOR_BGR2RGB
        cap = VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        rgb = cvtColor(frame, COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.setPixmap(pix.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self._path = path
        self.clear_btn.show()
        self.video_changed.emit(path)

    def _on_clear(self):
        self._path = None
        self.setPixmap(QPixmap())
        self.setText("Перетащите или нажмите для выбора видео")
        self.clear_btn.hide()
        self.video_changed.emit(None)

    def current_path(self):
        return self._path


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None
        self._analysis_start = None
        self._closing = False
        self.setWindowTitle("Emotionalyze")
        self.resize(1100, 700)

        # Состояние
        self._video_path: Optional[Path] = None
        self._emotion_map: Optional[List[Tuple[str, str]]] = None
        self._emotion_probs: Optional[List[List[float]]] = None
        self._last_percentages: Dict[str, float] = {}
        self._fps: int = 0
        self._loaded_maps: List[LoadedMap] = []
        self._test_map_index: Optional[int] = None
        # строки результатов сравнения для выгрузки в CSV
        self._results: List[Dict[str, str]] = []

        # --- Виджет предпросмотра видео ---
        video_box = QGroupBox("Предпросмотр видео")
        v_layout = QVBoxLayout(video_box)
        self.preview = VideoPreview()
        self.preview.setFixedHeight(200)
        v_layout.addWidget(self.preview)
        self.analyze_btn = QPushButton("Проанализировать видео")
        self.analyze_btn.setEnabled(False)
        v_layout.addWidget(self.analyze_btn)
        self.cancel_btn = QPushButton("Отменить анализ")
        self.cancel_btn.setEnabled(False)
        v_layout.addWidget(self.cancel_btn)

        # --- Загрузка JSON-карт ---
        maps_box = QGroupBox("Загруженные эталонные карты")
        m_layout = QVBoxLayout(maps_box)
        self.maps_list = QListWidget()
        self.maps_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.maps_list.customContextMenuRequested.connect(self.on_map_context_menu)
        self.maps_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        m_layout.addWidget(self.maps_list)
        maps_btns = QHBoxLayout()
        self.load_maps_btn = QPushButton("Загрузить JSON-карты")
        self.clear_maps_btn = QPushButton("Очистить")
        maps_btns.addWidget(self.load_maps_btn)
        maps_btns.addWidget(self.clear_maps_btn)
        m_layout.addLayout(maps_btns)

        # --- Лог + скачивание результирующей карты ---
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.download_btn = QPushButton("Скачать результирующую карту")
        self.download_btn.setEnabled(False)
        self.download_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # --- Параметры сравнения ---
        params_box = QGroupBox("Параметры сравнения")
        grid = QGridLayout(params_box)
        grid.setContentsMargins(6, 6, 6, 6)

        grid.addWidget(QLabel("Метод сравнения:"), 0, 0)
        self.method_combo = QComboBox()
        for key in METHODS:
            self.method_combo.addItem(METHOD_TITLES[key], key)
        self.method_combo.addItem("Все методы", ALL_METHODS_KEY)
        self.method_combo.setCurrentIndex(METHODS.index(METHOD_COMBINED))
        grid.addWidget(self.method_combo, 0, 1, 1, 3)

        grid.addWidget(QLabel("Макс. сдвиг, с:"), 1, 0)
        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(0.0, 30.0)
        self.shift_spin.setSingleStep(0.1)
        self.shift_spin.setDecimals(2)
        self.shift_spin.setValue(RECOMMENDED_SHIFT_SECONDS)
        self.shift_spin.setToolTip("Пересчитывается в кадры по частоте кадров тестируемой карты")
        grid.addWidget(self.shift_spin, 1, 1)

        grid.addWidget(QLabel("Окно сглаживания, кадров:"), 1, 2)
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(0, 301)
        self.smooth_spin.setValue(0)
        self.smooth_spin.setToolTip("0 - без сглаживания")
        grid.addWidget(self.smooth_spin, 1, 3)

        grid.addWidget(QLabel("Вес уступающей эмоции α:"), 2, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.5)
        grid.addWidget(self.alpha_spin, 2, 1)

        self.probs_check = QCheckBox("Полные векторы вероятностей")
        self.probs_check.setChecked(True)
        self.probs_check.setToolTip("Используются, если они сохранены в карте; "
                                    "иначе вектор строится по доминантной и уступающей эмоции")
        grid.addWidget(self.probs_check, 2, 2, 1, 2)

        self.neutral_check = QCheckBox("Исключать безэмоциональные фрагменты (нейтральная – нейтральная)")
        self.neutral_check.setChecked(RECOMMENDED_SKIP_NEUTRAL)
        grid.addWidget(self.neutral_check, 3, 0, 1, 4)

        grid.addWidget(QLabel("Веса комб. оценки:"), 4, 0)
        w_row = QHBoxLayout()
        self.weight_spins: List[QDoubleSpinBox] = []
        for caption, val in zip(("баз.", "вект.", "DTW", "част."), RECOMMENDED_WEIGHTS):
            w_row.addWidget(QLabel(caption))
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.05)
            sp.setValue(val)
            sp.setFixedWidth(60)
            self.weight_spins.append(sp)
            w_row.addWidget(sp)
        w_row.addStretch()
        grid.addLayout(w_row, 4, 1, 1, 3)

        # --- Выбор времени и кнопка сравнения ---
        t_row = QHBoxLayout()
        t_row.setContentsMargins(0, 0, 0, 0)
        t_row.setSpacing(2)
        t_row.addWidget(QLabel("Время сравнения с"))
        self.start_time = QTimeEdit(QTime(0, 0, 0))
        self.start_time.setDisplayFormat("HH:mm:ss")
        self.start_time.setFixedWidth(80)
        t_row.addWidget(self.start_time)
        t_row.addWidget(QLabel("по"))
        self.end_time = QTimeEdit(QTime(0, 0, 0))
        self.end_time.setDisplayFormat("HH:mm:ss")
        self.end_time.setFixedWidth(80)
        t_row.addWidget(self.end_time)
        t_row.addSpacing(20)
        self.compare_btn = QPushButton("Сравнить карты")
        self.compare_btn.setEnabled(False)
        self.compare_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        t_row.addWidget(self.compare_btn)

        # --- Межэталонное сравнение и выгрузка результатов ---
        r_row = QHBoxLayout()
        self.inter_btn = QPushButton("Межэталонное сравнение")
        self.inter_btn.setEnabled(False)
        self.inter_btn.setToolTip("Среднее сходство эталонных карт между собой "
                                  "(критерий оценки тестируемой карты)")
        self.export_btn = QPushButton("Сохранить результаты (CSV)")
        self.export_btn.setEnabled(False)
        self.clear_results_btn = QPushButton("Очистить результаты")
        r_row.addWidget(self.inter_btn)
        r_row.addWidget(self.export_btn)
        r_row.addWidget(self.clear_results_btn)

        # --- Компоновка интерфейса ---
        right = QVBoxLayout()
        right.addWidget(self.log, 1)
        right.addWidget(self.download_btn)
        right.addWidget(params_box)
        right.addLayout(t_row)
        right.addLayout(r_row)

        main = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(video_box, 1)
        left.addWidget(maps_box, 1)
        main.addLayout(left, 1)
        main.addLayout(right, 2)

        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

        # --- Связи сигналов ---
        self.preview.video_changed.connect(self.on_video_changed)
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.cancel_btn.clicked.connect(self.on_cancel_analysis)
        self.download_btn.clicked.connect(self.on_download)
        self.load_maps_btn.clicked.connect(self.on_load_maps)
        self.clear_maps_btn.clicked.connect(self.on_clear_maps)
        self.compare_btn.clicked.connect(self.on_compare)
        self.inter_btn.clicked.connect(self.on_inter_reference)
        self.export_btn.clicked.connect(self.on_export_results)
        self.clear_results_btn.clicked.connect(self.on_clear_results)
        self.method_combo.currentIndexChanged.connect(self.update_weight_controls)
        self.update_weight_controls()

    def log_msg(self, text: str):
        self.log.append(text)

    # ------------------------------------------------------------------
    # Видео
    # ------------------------------------------------------------------

    def _reset_video_result(self):
        self._emotion_map = None
        self._emotion_probs = None
        self._last_percentages = {}
        self._fps = 0
        self.download_btn.setEnabled(False)
        self.update_compare_btn()

    def on_video_changed(self, path: Optional[Path]):
        if self.thread is not None:
            self.on_cancel_analysis()

        self._video_path = path
        self._test_map_index = None
        self.clear_test_highlight()
        self._reset_video_result()
        self.analyze_btn.setEnabled(
            bool(path) and self.thread is None and not self._closing
        )
        if path:
            self.log_msg(f"Видео загружено: {path.name}")
        else:
            self.log_msg("Видео удалено")

    def on_analyze(self):
        if self.thread is not None or self._video_path is None or self._closing:
            return

        self._reset_video_result()
        self._analysis_start = _pytime.time()
        self.thread = VideoAnalysisWorker(self._video_path, workers=2, parent=self)
        self.thread.finished.connect(self._on_analysis_stopped)

        self.analyze_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log_msg("Старт анализа видео…")
        self.thread.start()

    def on_cancel_analysis(self):
        if self.thread is None:
            return
        if not self.thread.cancel_event.is_set():
            self.thread.cancel()
            self.log_msg("Отмена анализа: ожидаем завершения процессов…")
        self.cancel_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self._reset_video_result()

    @Slot()
    def _on_analysis_stopped(self):
        thread = self.thread
        if thread is None:
            return
        # finished приходит после выхода из process_video_parallel, включая
        # shutdown пула процессов. Дожидаемся и завершения самого QThread.
        thread.wait()
        cancelled = thread.cancel_event.is_set() or self._closing
        result = thread.result
        error = thread.error_message
        self.thread = None
        thread.result = None
        thread.deleteLater()
        self.cancel_btn.setEnabled(False)

        try:
            if cancelled:
                self._reset_video_result()
                self.log_msg("Анализ отменён. Процессы завершены.")
            elif error is not None:
                self._reset_video_result()
                self.log_msg("Ошибка: " + error)
            elif result is not None:
                self._on_analysis_done(*result)
        finally:
            self.analyze_btn.setEnabled(
                self._video_path is not None and not self._closing
            )
            if self._closing:
                QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self.thread is not None:
            self._closing = True
            self.on_cancel_analysis()
            event.ignore()
            return
        super().closeEvent(event)

    def _print_emotion_stats(self, percentages: Dict[str, float]):
        self.log_msg("Распределение эмоций (%):")
        for emo, pct in percentages.items():
            self.log_msg(f"  {emo:8s}: {pct:6.2f}%")

    def _set_time_limits(self, frames: int, fps: int):
        duration = frames / fps if fps else 0
        h, rem = divmod(duration, 3600)
        m, s = divmod(rem, 60)
        max_t = QTime(int(h), int(m), int(s))
        self.start_time.setMaximumTime(max_t)
        self.end_time.setMaximumTime(max_t)
        self.end_time.setTime(max_t)

    def _on_analysis_done(self, fps, percentages, emotion_map, probs):
        # сохраним результат
        self._fps = fps
        self._emotion_map = emotion_map
        self._emotion_probs = probs
        self._last_percentages = percentages

        self._set_time_limits(len(emotion_map), fps)

        # лог
        self._print_emotion_stats(percentages)
        elapsed = _pytime.time() - self._analysis_start
        self.log_msg(f"Анализ завершён за {elapsed:.2f} сек.")

        self.download_btn.setEnabled(True)
        self.update_compare_btn()

    def on_download(self):
        if self._emotion_map is None:
            return

        settings = QSettings("Emotionalyze", "Emotionalyze")

        # Получаем последнюю директорию сохранения
        last_dir = settings.value("last_save_dir", str(Path.home()))

        # Формируем имя файла
        default_name = f"EM_{self._video_path.stem}.json" if self._video_path else "emotion_map.json"

        # Формируем полный путь
        default_path = str(Path(last_dir) / default_name)

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить карту",
            default_path,
            "JSON (*.json)"
        )

        if save_path:
            save_emotion_map(
                self._emotion_map,
                self._last_percentages,
                Path(save_path),
                self._fps,
                self._emotion_probs
            )

            # Запоминаем директорию сохранения
            settings.setValue("last_save_dir", str(Path(save_path).parent))

            self.log_msg(f"Карта сохранена: {Path(save_path).name}")
        else:
            self.log_msg("Сохранение отменено.")

    # ------------------------------------------------------------------
    # Загруженные карты
    # ------------------------------------------------------------------

    def _add_loaded_map(self, name: str, m: List[Tuple[str, str]], fps: int,
                        probs: Optional[List[List[float]]] = None):
        item = QListWidgetItem()
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(5, 0, 5, 0)
        lbl = QLabel(name)
        btn = QPushButton("✕")
        btn.setFixedSize(16, 16)
        lay.addWidget(lbl)
        lay.addStretch()
        lay.addWidget(btn)
        item.setSizeHint(w.sizeHint())
        self.maps_list.addItem(item)
        self.maps_list.setItemWidget(item, w)
        btn.clicked.connect(lambda _, it=item: self._remove_map(it))
        self._loaded_maps.append(LoadedMap(name, m, fps, item, probs))
        kind = "с полными векторами вероятностей" if probs is not None else "доминантная и уступающая эмоции"
        self.log_msg(f"Загружена карта: {name} ({kind})")
        self.update_compare_btn()

    def _remove_map(self, item: QListWidgetItem):
        for idx, lm in enumerate(self._loaded_maps):
            if lm.item is item:
                self._loaded_maps.pop(idx)
                if self._test_map_index == idx:
                    self._test_map_index = None
                elif self._test_map_index is not None and self._test_map_index > idx:
                    self._test_map_index -= 1
                self.maps_list.takeItem(self.maps_list.row(item))
                self.log_msg(f"Карта удалена: {lm.name}")
                break
        self.clear_test_highlight()
        self.highlight_test_map()
        self.update_compare_btn()

    def on_load_maps(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите JSON-файлы", "", "JSON (*.json)"
        )
        for f in files:
            try:
                fps, percentages, m, probs = load_emotion_map_ext(Path(f))
                self._add_loaded_map(Path(f).name, m, fps, probs)
            except Exception as e:
                self.log_msg(f"Ошибка загрузки {Path(f).name}: {e}")
        self.update_compare_btn()

    def on_clear_maps(self):
        self._loaded_maps.clear()
        self.maps_list.clear()
        self._test_map_index = None
        self.log_msg("Список карт очищен.")
        self.update_compare_btn()

    def on_map_context_menu(self, pos: QPoint):
        if len(self._loaded_maps) < 2:
            return
        item = self.maps_list.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        act = menu.addAction("Тестируемая")
        act.triggered.connect(lambda: self.set_test_map(item))
        menu.exec(self.maps_list.viewport().mapToGlobal(pos))

    def set_test_map(self, item: QListWidgetItem):
        for idx, lm in enumerate(self._loaded_maps):
            if lm.item is item:
                self._test_map_index = idx
                break
        self.clear_test_highlight()
        self.highlight_test_map()
        lm = self._loaded_maps[self._test_map_index]
        self.log_msg(f"Тестируемая карта: {lm.name}")
        self._set_time_limits(len(lm.labels), lm.fps)
        self.update_compare_btn()

    def highlight_test_map(self):
        if self._test_map_index is None:
            return
        w = self.maps_list.itemWidget(self._loaded_maps[self._test_map_index].item)
        if w is not None:
            w.setStyleSheet("background-color: rgba(0,120,215,100);")

    def clear_test_highlight(self):
        for lm in self._loaded_maps:
            w = self.maps_list.itemWidget(lm.item)
            if w is not None:
                w.setStyleSheet("")

    # ------------------------------------------------------------------
    # Сравнение
    # ------------------------------------------------------------------

    def update_weight_controls(self, *_):
        key = self.method_combo.currentData()
        enabled = key in (METHOD_COMBINED, ALL_METHODS_KEY)
        for sp in self.weight_spins:
            sp.setEnabled(enabled)

    def current_params(self, fps: int = 30) -> CompareParams:
        return CompareParams(
            max_shift=int(round(self.shift_spin.value() * fps)),
            alpha=self.alpha_spin.value(),
            smooth_window=self.smooth_spin.value(),
            skip_neutral=self.neutral_check.isChecked(),
            use_probs=self.probs_check.isChecked(),
            weights=tuple(sp.value() for sp in self.weight_spins),
        )

    def _effective_params(self, probs_list, fps: int) -> CompareParams:
        """Полные векторы используются, только если они есть во всех сравниваемых картах."""
        params = self.current_params(fps)
        if params.use_probs and any(p is None for p in probs_list):
            params.use_probs = False
            self.log_msg("Не во всех картах есть полные векторы вероятностей – "
                         "используется представление по доминантной и уступающей эмоции.")
        return params

    def current_methods(self) -> List[str]:
        key = self.method_combo.currentData()
        return list(METHODS) if key == ALL_METHODS_KEY else [key]

    def _frame_range(self, fps: int, max_frames: int) -> Optional[Tuple[int, int, int, int]]:
        t1, t2 = self.start_time.time(), self.end_time.time()
        sec1 = t1.hour() * 3600 + t1.minute() * 60 + t1.second()
        sec2 = t2.hour() * 3600 + t2.minute() * 60 + t2.second()
        if sec2 <= sec1:
            self.log_msg("Ошибка: 'по' должно быть позже 'с'")
            return None
        sf, ef = int(sec1 * fps), int(sec2 * fps)
        if sf < 0 or ef > max_frames:
            self.log_msg(f"Диапазон кадров 0–{max_frames}, задан {sf}–{ef}")
            return None
        return sf, ef, sec1, sec2

    def _add_result(self, test_name, ref_name, sec1, sec2, sf, ef, method, value, params):
        self._results.append({
            "Дата и время": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Тестируемая карта": test_name,
            "Эталонная карта": ref_name,
            "Начало, с": str(sec1),
            "Конец, с": str(sec2),
            "Начальный кадр": str(sf),
            "Конечный кадр": str(ef),
            "Метод": METHOD_TITLES.get(method, method),
            "Сходство, %": "" if value is None or math.isnan(value) else f"{value:.2f}".replace(".", ","),
            "Макс. сдвиг": str(params.max_shift),
            "alpha": f"{params.alpha:.2f}".replace(".", ","),
            "Окно сглаживания": str(params.smooth_window),
            "Исключение нейтральных": "да" if params.skip_neutral else "нет",
            "Полные векторы": "да" if params.use_probs else "нет",
            "Веса": " / ".join(f"{w:.2f}" for w in params.weights),
        })
        self.export_btn.setEnabled(True)

    def on_compare(self):
        use_video = (self._test_map_index is None)
        if use_video:
            if not self._emotion_map or not self._loaded_maps:
                return
            test_name = self._video_path.name if self._video_path else "результат анализа видео"
            test_labels, test_probs, fps = self._emotion_map, self._emotion_probs, self._fps
            refs = list(self._loaded_maps)
        else:
            if len(self._loaded_maps) < 2:
                return
            tm = self._loaded_maps[self._test_map_index]
            test_name, test_labels, test_probs, fps = tm.name, tm.labels, tm.probs, tm.fps
            refs = [lm for k, lm in enumerate(self._loaded_maps) if k != self._test_map_index]

        rng = self._frame_range(fps, len(test_labels))
        if rng is None:
            return
        sf, ef, sec1, sec2 = rng
        params = self._effective_params([test_probs] + [r.probs for r in refs], fps)
        methods = self.current_methods()

        self.log_msg(f"Сравнение с кадра {sf} по {ef}…")
        self.log_msg(f"Параметры: {params.describe()}")
        t0 = _pytime.time()
        test_prep = prepare_map(test_labels, test_probs, params).slice(sf, ef)

        best: Dict[str, Tuple[Optional[str], float]] = {m: (None, -1.0) for m in methods}
        scores: Dict[str, List[float]] = {m: [] for m in methods}
        ref_preps = []
        for ref in refs:
            ref_prep = prepare_map(ref.labels, ref.probs, params).slice(sf, ef)
            ref_preps.append(ref_prep)
            res = compare_prepared(ref_prep, test_prep, methods, params)
            for m in methods:
                scores[m].append(res.get(m, math.nan))
            if len(methods) == 1:
                self.log_msg(f"{ref.name} → {format_value(res[methods[0]])}")
            else:
                parts = " | ".join(f"{METHOD_TITLES[m]}: {format_value(res[m])}" for m in methods)
                self.log_msg(f"{ref.name} → {parts}")
            for m in methods:
                val = res.get(m, math.nan)
                self._add_result(test_name, ref.name, sec1, sec2, sf, ef, m, val, params)
                if not math.isnan(val) and val > best[m][1]:
                    best[m] = (ref.name, val)

        for m in methods:
            name, val = best[m]
            prefix = "Лучшее совпадение" if len(methods) == 1 else f"Лучшее совпадение ({METHOD_TITLES[m]})"
            if name is None:
                self.log_msg(f"{prefix}: н/д")
            else:
                self.log_msg(f"{prefix}: {name} – {val:.2f}%")

        # вердикт по критерию межэталонного сравнения
        if len(ref_preps) >= 2:
            main_method = METHOD_COMBINED if len(methods) > 1 else methods[0]
            pair_scores = [compare_prepared(a, b, [main_method], params)[main_method]
                           for a, b in combinations(ref_preps, 2)]
            mean_test, crit, ok = verdict(scores[main_method], pair_scores)
            if ok is None:
                self.log_msg("Вердикт: недостаточно данных")
            else:
                self.log_msg(f"Среднее сходство с эталонами ({METHOD_TITLES[main_method]}): "
                             f"{mean_test:.2f}%, критерий (среднее межэталонное): {crit:.2f}%")
                if ok:
                    self.log_msg("Вердикт: реакция соответствует эталонным")
                else:
                    self.log_msg("Вердикт: реакция отличается от эталонных, "
                                 "рекомендуется дополнительный анализ специалистом")
        self.log_msg(f"Время сравнения: {_pytime.time() - t0:.2f} сек.")

    def on_inter_reference(self):
        """Среднее сходство эталонных карт между собой (все пары без повторений)."""
        refs = [lm for k, lm in enumerate(self._loaded_maps) if k != self._test_map_index]
        if len(refs) < 2:
            return
        fps = refs[0].fps
        rng = self._frame_range(fps, min(len(r.labels) for r in refs))
        if rng is None:
            return
        sf, ef, sec1, sec2 = rng
        params = self._effective_params([r.probs for r in refs], fps)
        methods = self.current_methods()
        prepared = [prepare_map(r.labels, r.probs, params).slice(sf, ef) for r in refs]

        self.log_msg(f"Межэталонное сравнение с кадра {sf} по {ef} ({len(refs)} карт)…")
        self.log_msg(f"Параметры: {params.describe()}")
        sums = {m: 0.0 for m in methods}
        counts = {m: 0 for m in methods}
        for (i, a), (j, b) in combinations(enumerate(prepared), 2):
            res = compare_prepared(a, b, methods, params)
            for m in methods:
                val = res.get(m, math.nan)
                self._add_result(refs[j].name, refs[i].name, sec1, sec2, sf, ef, m, val, params)
                if not math.isnan(val):
                    sums[m] += val
                    counts[m] += 1
        for m in methods:
            mean = sums[m] / counts[m] if counts[m] else math.nan
            self.log_msg(f"Среднее совпадение эталонов ({METHOD_TITLES[m]}): {format_value(mean)}")

    def on_export_results(self):
        if not self._results:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты сравнения", "comparison_results.csv", "CSV (*.csv)"
        )
        if not save_path:
            self.log_msg("Сохранение результатов отменено.")
            return
        fields = list(self._results[0].keys())
        with open(save_path, "w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields, delimiter=";")
            writer.writeheader()
            writer.writerows(self._results)
        self.log_msg(f"Результаты сохранены: {Path(save_path).name} ({len(self._results)} строк)")

    def on_clear_results(self):
        self._results.clear()
        self.export_btn.setEnabled(False)
        self.log_msg("Результаты сравнения очищены.")

    def update_compare_btn(self):
        if self._test_map_index is not None:
            ready = len(self._loaded_maps) >= 2
        else:
            ready = bool(self._emotion_map) and bool(self._loaded_maps)
        self.compare_btn.setEnabled(ready)
        n_refs = len(self._loaded_maps) - (1 if self._test_map_index is not None else 0)
        self.inter_btn.setEnabled(n_refs >= 2)


def run_app():
    app = QApplication(sys.argv)
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet())
    except ImportError:
        from PySide6.QtGui import QColor
        palette = app.palette()
        palette.setColor(palette.Window, QColor(53, 53, 53))
        palette.setColor(palette.WindowText, Qt.white)
        palette.setColor(palette.Base, QColor(35, 35, 35))
        palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(palette.ToolTipBase, Qt.white)
        palette.setColor(palette.ToolTipText, Qt.white)
        palette.setColor(palette.Text, Qt.white)
        palette.setColor(palette.Button, QColor(53, 53, 53))
        palette.setColor(palette.ButtonText, Qt.white)
        palette.setColor(palette.Highlight, QColor(142, 45, 197).darker())
        palette.setColor(palette.HighlightedText, Qt.black)
        app.setPalette(palette)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
