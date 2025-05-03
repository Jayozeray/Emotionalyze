# gui.py

import sys
import time as _pytime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QListWidget, QTextEdit, QFileDialog, QListWidgetItem,
    QHBoxLayout, QVBoxLayout, QGroupBox, QSizePolicy, QTimeEdit,
    QMenu
)
from PySide6.QtGui import QPixmap, QImage, QColor, QBrush
from PySide6.QtCore import Qt, Signal, QTime, QPoint, QThread, QObject

from backend import (
    process_video_parallel,
    save_emotion_map,
    load_emotion_map,
    compare_emotion_maps
)

class VideoAnalysisWorker(QObject):
    started = Signal()
    finished = Signal(int, dict, list)   # fps, percentages, emotion_map
    error = Signal(str)

    def __init__(self, video_path, workers=None):
        super().__init__()
        self.video_path = video_path
        self.workers = workers

    def run(self):
        self.started.emit()
        try:
            fps, percentages, emap = process_video_parallel(
                self.video_path,
                workers=self.workers
            )
            self.finished.emit(fps, percentages, emap)
        except Exception as e:
            self.error.emit(str(e))

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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "", "Видео (*.mp4 *.avi *.mov)"
        )
        if file_path:
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
        img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
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
        self.worker = None
        self.thread = None
        self._analysis_start = None
        self.setWindowTitle("Emotionalyze")
        self.resize(1000, 600)

        # Состояние
        self._video_path: Optional[Path] = None
        self._emotion_map: Optional[List[Tuple[str, str]]] = None
        self._fps: int = 0
        # теперь храним и fps загруженных карт: (name, map, fps, item)
        self._loaded_maps: List[Tuple[str, List[Tuple[str, str]], int, QListWidgetItem]] = []
        self._test_map_index: Optional[int] = None

        # --- Виджет предпросмотра видео ---
        video_box = QGroupBox("Предпросмотр видео")
        v_layout = QVBoxLayout(video_box)
        self.preview = VideoPreview()
        self.preview.setFixedHeight(200)
        v_layout.addWidget(self.preview)
        self.analyze_btn = QPushButton("Проанализировать видео")
        self.analyze_btn.setEnabled(False)
        v_layout.addWidget(self.analyze_btn)

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

        # --- Выбор времени и кнопка сравнения ---
        t_row = QHBoxLayout()
        t_row.setContentsMargins(0,0,0,0)
        t_row.setSpacing(2)
        t_row.addWidget(QLabel("Время сравнения с"))
        self.start_time = QTimeEdit(QTime(0,0,0))
        self.start_time.setDisplayFormat("HH:mm:ss")
        self.start_time.setFixedWidth(80)
        t_row.addWidget(self.start_time)
        t_row.addWidget(QLabel("по"))
        self.end_time = QTimeEdit(QTime(0,0,0))
        self.end_time.setDisplayFormat("HH:mm:ss")
        self.end_time.setFixedWidth(80)
        t_row.addWidget(self.end_time)
        t_row.addSpacing(20)
        self.compare_btn = QPushButton("Сравнить карты")
        self.compare_btn.setEnabled(False)
        self.compare_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        t_row.addWidget(self.compare_btn)

        # --- Компоновка интерфейса ---
        right = QVBoxLayout()
        right.addWidget(self.log, 1)
        right.addWidget(self.download_btn)
        right.addLayout(t_row)

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
        self.download_btn.clicked.connect(self.on_download)
        self.load_maps_btn.clicked.connect(self.on_load_maps)
        self.clear_maps_btn.clicked.connect(self.on_clear_maps)
        self.compare_btn.clicked.connect(self.on_compare)

    def log_msg(self, text: str):
        self.log.append(text)

    def on_video_changed(self, path: Optional[Path]):
        self._video_path = path
        self._test_map_index = None
        self.clear_test_highlight()
        ok = bool(path)
        self.analyze_btn.setEnabled(ok)
        if ok:
            self.download_btn.setEnabled(False)
            self._emotion_map = None
            self.log_msg(f"Видео загружено: {path.name}")
        else:
            self.log_msg("Видео удалено")
            self.compare_btn.setEnabled(False)

    def on_analyze(self):

        # отмечаем момент старта
        self._analysis_start = _pytime.time()
        # создаём QThread + worker
        self.thread = QThread(self)
        self.worker = VideoAnalysisWorker(self._video_path, workers=None)
        self.worker.moveToThread(self.thread)

        # сигналы
        self.worker.started.connect(lambda: self.log_msg("Старт анализа видео…"))
        self.worker.finished.connect(self._on_analysis_done)
        self.worker.error.connect(lambda msg: self.log_msg("Ошибка: " + msg))
        self.thread.started.connect(self.worker.run)

        # блокируем кнопки и запускаем
        self.analyze_btn.setEnabled(False)
        self.thread.start()

    def _print_emotion_stats(self, percentages: Dict[str, float]):
        self.log_msg("Распределение эмоций (%):")
        for emo, pct in percentages.items():
            self.log_msg(f"  {emo:8s}: {pct:6.2f}%")

    def _on_analysis_done(self, fps,percentages, emotion_map):
        # получили результат — запомним
        self._fps = fps
        self._emotion_map = emotion_map

        # обновим таймеры, кнопки, лог
        duration = len(emotion_map) / fps
        h, rem = divmod(duration, 3600)
        m, s = divmod(rem, 60)
        max_t = QTime(int(h), int(m), int(s))
        self.start_time.setMaximumTime(max_t)
        self.end_time.setMaximumTime(max_t)
        self.end_time.setTime(max_t)

        # Выводим отформатированную таблицу в лог
        self._print_emotion_stats(percentages)

        # вычисляем реальное время анализа
        elapsed = _pytime.time() - self._analysis_start
        self.log_msg(f"Анализ завершён за {elapsed:.2f} сек.")

        self.download_btn.setEnabled(True)
        self.update_compare_btn()

        # аккуратно останавливаем поток
        self.thread.quit()
        self.thread.wait()
        self.worker = None
        self.thread = None

    def on_download(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить карту", "emotion_map.json", "JSON (*.json)"
        )
        if path:
            save_emotion_map(self._emotion_map, Path(path), self._fps)
            self.log_msg(f"Карта сохранена: {Path(path).name}")
        else:
            self.log_msg("Сохранение отменено.")

    def _add_loaded_map(self, name: str, m: List[Tuple[str, str]], fps: int):
        item = QListWidgetItem()
        w = QWidget()
        lay = QHBoxLayout(w); lay.setContentsMargins(5,0,5,0)
        lbl = QLabel(name)
        btn = QPushButton("✕"); btn.setFixedSize(16,16)
        lay.addWidget(lbl); lay.addStretch(); lay.addWidget(btn)
        item.setSizeHint(w.sizeHint())
        self.maps_list.addItem(item)
        self.maps_list.setItemWidget(item, w)
        btn.clicked.connect(lambda _, it=item: self._remove_map(it))
        self._loaded_maps.append((name, m, fps, item))
        self.log_msg(f"Загружена карта: {name}")
        self.update_compare_btn()

    def _remove_map(self, item: QListWidgetItem):
        for idx, (nm, m, fps, it) in enumerate(self._loaded_maps):
            if it is item:
                self._loaded_maps.pop(idx)
                if self._test_map_index == idx:
                    self._test_map_index = None
                self.maps_list.takeItem(self.maps_list.row(item))
                self.log_msg(f"Карта удалена: {nm}")
                break
        self.clear_test_highlight()
        self.update_compare_btn()

    def on_load_maps(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите JSON-файлы", "", "JSON (*.json)"
        )
        for f in files:
            try:
                fps, m = load_emotion_map(Path(f))
                self._add_loaded_map(Path(f).name, m, fps)
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
        # нашли индекс тестовой карты
        for idx, (_n, _m, fps, it) in enumerate(self._loaded_maps):
            if it is item:
                self._test_map_index = idx
                break

        self.clear_test_highlight()
        # подсвечиваем ей фон
        w = self.maps_list.itemWidget(item)
        w.setStyleSheet("background-color: rgba(0,120,215,100);")
        name, m, fps, _ = self._loaded_maps[self._test_map_index]
        self.log_msg(f"Тестируемая карта: {name}")

        # выставляем время по длительности этой карты
        duration = len(m) / fps
        h, rem = divmod(duration, 3600)
        m2, s = divmod(rem, 60)
        max_t = QTime(int(h), int(m2), int(s))
        self.start_time.setMaximumTime(max_t)
        self.end_time.setMaximumTime(max_t)
        self.end_time.setTime(max_t)

        self.update_compare_btn()

    def clear_test_highlight(self):
        for _n, _m, _fps, it in self._loaded_maps:
            w = self.maps_list.itemWidget(it)
            w.setStyleSheet("")

    def on_compare(self):
        use_video = (self._test_map_index is None)
        if use_video:
            if not self._emotion_map or not self._loaded_maps:
                return
        else:
            if len(self._loaded_maps) < 2:
                return

        t1, t2 = self.start_time.time(), self.end_time.time()
        sec1 = t1.hour()*3600 + t1.minute()*60 + t1.second()
        sec2 = t2.hour()*3600 + t2.minute()*60 + t2.second()
        if sec2 <= sec1:
            return self.log_msg("Ошибка: 'по' должно быть позже 'с'")

        if use_video:
            sf, ef = int(sec1*self._fps), int(sec2*self._fps)
            maxf = len(self._emotion_map)
        else:
            _, test_map, fps, _ = self._loaded_maps[self._test_map_index]
            sf, ef = int(sec1*fps), int(sec2*fps)
            maxf = len(test_map)

        if sf < 0 or ef > maxf:
            return self.log_msg(f"Диапазон кадров 0–{maxf}, задан {sf}–{ef}")

        self.log_msg(f"Сравнение с кадра {sf} по {ef}…")
        best_name, best_pct = None, -1.0

        if use_video:
            for name, ref_map, _fps, _ in self._loaded_maps:
                pct = compare_emotion_maps(
                    self._emotion_map, ref_map,
                    start_frame=sf, end_frame=ef
                )
                self.log_msg(f"{name} → {pct:.2f}%")
                if pct > best_pct:
                    best_pct, best_name = pct, name
        else:
            _, test_map, _, _ = self._loaded_maps[self._test_map_index]
            for idx, (name, ref_map, _fps, _) in enumerate(self._loaded_maps):
                if idx == self._test_map_index:
                    continue
                pct = compare_emotion_maps(
                    ref_map, test_map,
                    start_frame=sf, end_frame=ef
                )
                self.log_msg(f"{name} → {pct:.2f}%")
                if pct > best_pct:
                    best_pct, best_name = pct, name

        self.log_msg(f"Лучшее совпадение: {best_name} — {best_pct:.2f}%")

    def update_compare_btn(self):
        if self._test_map_index is not None:
            ready = len(self._loaded_maps) >= 2
        else:
            ready = bool(self._emotion_map) and bool(self._loaded_maps)
        self.compare_btn.setEnabled(ready)


def run_app():
    app = QApplication(sys.argv)
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet())
    except ImportError:
        palette = app.palette()
        palette.setColor(palette.Window,        QColor(53,53,53))
        palette.setColor(palette.WindowText,    Qt.white)
        palette.setColor(palette.Base,          QColor(35,35,35))
        palette.setColor(palette.AlternateBase, QColor(53,53,53))
        palette.setColor(palette.ToolTipBase,   Qt.white)
        palette.setColor(palette.ToolTipText,   Qt.white)
        palette.setColor(palette.Text,          Qt.white)
        palette.setColor(palette.Button,        QColor(53,53,53))
        palette.setColor(palette.ButtonText,    Qt.white)
        palette.setColor(palette.Highlight,     QColor(142,45,197).darker())
        palette.setColor(palette.HighlightedText, Qt.black)
        app.setPalette(palette)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
