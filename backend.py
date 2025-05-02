# backend.py

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from deepface import DeepFace
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_EMOTIONS = (
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
)


def _analyze_chunk(
    video_path: str,
    start: int,
    end: int,
    actions: Tuple[str, ...],
    enforce_detection: bool,
    worker_id: int
) -> Tuple[List[Tuple[int, Tuple[str, str]]], Dict[str, float], int]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    local_sums = {emo: 0.0 for emo in DEFAULT_EMOTIONS}
    local_top2: List[Tuple[int, Tuple[str, str]]] = []
    processed = 0

    for frame_idx in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break

        processed += 1
        try:
            res = DeepFace.analyze(
                img_path=frame,
                actions=list(actions),
                enforce_detection=enforce_detection
            )
            if isinstance(res, list):
                res = res[0]
            emotions = res.get("emotion", {})
        except Exception:
            # при ошибке — нулевые вероятности
            emotions = {}

        # гарантируем наличие всех ключей
        for emo in DEFAULT_EMOTIONS:
            emotions.setdefault(emo, 0.0)

        # накапливаем вероятности
        for emo, p in emotions.items():
            local_sums[emo] += float(p)

        # выбираем топ-2
        sorted_by_prob = sorted(
            DEFAULT_EMOTIONS,
            key=lambda e: emotions[e],
            reverse=True
        )
        local_top2.append((frame_idx, (sorted_by_prob[0], sorted_by_prob[1])))

    cap.release()
    return local_top2, local_sums, processed


def process_video_parallel(
    video_path: Path,
    actions: Tuple[str, ...] = ("emotion",),
    enforce_detection: bool = False,
    workers: int = None
) -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
    """
    Параллельно анализирует видео:
      - возвращает распределение эмоций (по средним «сырым» вероятностям)
      - и карту top-2 эмоций по каждому кадру
    """
    video_str = str(video_path)
    cap = cv2.VideoCapture(video_str)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = math.floor(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if workers is None:
        from multiprocessing import cpu_count
        workers = cpu_count()

    chunk_size = (total_frames + workers - 1) // workers
    ranges = [
        (i, min(i + chunk_size, total_frames))
        for i in range(0, total_frames, chunk_size)
    ]

    emotion_map: List[Tuple[str, str]] = [("", "")] * total_frames
    global_sums = {emo: 0.0 for emo in DEFAULT_EMOTIONS}
    frames_counted = 0

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [
            exe.submit(
                _analyze_chunk,
                video_str, start, end,
                actions, enforce_detection, wid
            )
            for wid, (start, end) in enumerate(ranges)
        ]
        for fut in as_completed(futures):
            local_top2, local_sums, processed = fut.result()
            # заполняем карту и складываем суммы
            for idx, (d, i) in local_top2:
                emotion_map[idx] = (d, i)
            for emo, s in local_sums.items():
                global_sums[emo] += s
            frames_counted += processed

    denom = frames_counted or total_frames
    emotion_percentages = {
        emo: prob_sum / denom
        for emo, prob_sum in global_sums.items()
    }

    return fps, emotion_percentages, emotion_map


def save_emotion_map(
    emotion_map: List[Tuple[str, str]],
    filename: Path,
    fps: int
) -> None:
    """
    Сохраняет emotion_map в файл JSON.
    Формат: [ {"dominant": "...", "inferior": "..."}, ... ]
    """
    data = [{"fps": fps}] + [
        {"dominant": d, "inferior": i}
        for d, i in emotion_map
    ]
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_emotion_map(
    filename: Path
) -> Tuple[int, List[Tuple[str, str]]]:
    """
    Загружает JSON-файл и возвращает (fps, emotion_map).
    Формат JSON:
      [
        {"fps": 23},
        {"dominant": "...", "inferior": "..."},
        {"dominant": "...", "inferior": "..."},
        ...
      ]
    """
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    # проверяем, что первый элемент — это fps
    if not data or not isinstance(data[0], dict) or "fps" not in data[0]:
        raise ValueError("JSON должен начинаться с объекта {\"fps\": <число>}")

    fps_val = data[0]["fps"]
    if not isinstance(fps_val, (int, float)):
        raise ValueError("Значение fps должно быть числом")
    fps = int(fps_val)

    emotion_map: List[Tuple[str, str]] = []
    # теперь перебираем все последующие элементы
    for idx, item in enumerate(data[1:], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Элемент {idx}: ожидается словарь")
        if "dominant" not in item or "inferior" not in item:
            raise KeyError(f"Элемент {idx}: нет ключей 'dominant' и 'inferior'")
        d = item["dominant"]
        i = item["inferior"]
        if not isinstance(d, str) or not isinstance(i, str):
            raise ValueError(f"Элемент {idx}: неверные типы, ожидаются строки")
        emotion_map.append((d, i))

    return fps, emotion_map


def compare_emotion_maps(
    ref_map: List[Tuple[str, str]],
    test_map: List[Tuple[str, str]],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_shift: int = 50
) -> float:
    """
    Сравнивает две карты эмоций и возвращает % лучшего совпадения.
    Сдвигами (до max_shift) учитывает отсутствие кадров как несовпадение.
    """
    slice_ref = ref_map[start_frame:end_frame]
    slice_test = test_map[start_frame:end_frame]
    def match_pct(
        anchor: List[Tuple[str, str]],
        moving: List[Tuple[str, str]],
        shift: int
    ) -> float:
        hits = 0
        total = len(anchor)
        for i in range(total):
            j = i + shift
            if 0 <= j < len(moving):
                d1, i1 = anchor[i]
                d2, i2 = moving[j]
                if d1 == d2 or (d1 == i2 and d2 == i1):
                    hits += 1
        return hits / total * 100.0

    best = 0.0
    # тестовая карта движется относительно эталона
    for s in range(max_shift + 1):
        best = max(best, match_pct(slice_ref, slice_test, s))
    for s in range(1, max_shift + 1):
        best = max(best, match_pct(slice_test, slice_ref, s))
    return best


__all__ = [
    "process_video_parallel",
    "save_emotion_map",
    "load_emotion_map",
    "compare_emotion_maps",
]
