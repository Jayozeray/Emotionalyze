# backend.py

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
from deepface import DeepFace
from concurrent.futures import ProcessPoolExecutor, as_completed

from comparison import base_similarity
from emotion_map_io import save_emotion_map, load_emotion_map_ext, load_emotion_map

DEFAULT_EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def _analyze_chunk(
    video_path: str,
    start: int,
    end: int,
    actions: Tuple[str, ...],
    enforce_detection: bool,
    worker_id: int
) -> Tuple[List[Tuple[int, Tuple[str, str]]], Dict[str, float], int, List[Tuple[int, List[float]]]]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    local_sums = {emo: 0.0 for emo in DEFAULT_EMOTIONS}
    local_top2: List[Tuple[int, Tuple[str, str]]] = []
    local_probs: List[Tuple[int, List[float]]] = []
    processed = 0

    while processed < (end - start):
        ret, frame = cap.read()
        if not ret:
            break
        idx = start + processed
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
            emotions = {}

        for emo in DEFAULT_EMOTIONS:
            emotions.setdefault(emo, 0.0)

        for emo, p in emotions.items():
            local_sums[emo] += float(p)

        sorted_probs = sorted(
            DEFAULT_EMOTIONS,
            key=lambda e: emotions[e],
            reverse=True
        )
        local_top2.append((idx, (sorted_probs[0], sorted_probs[1])))
        # полный вектор вероятностей (DeepFace возвращает проценты)
        local_probs.append((idx, [float(emotions[e]) / 100.0 for e in DEFAULT_EMOTIONS]))

    cap.release()
    return local_top2, local_sums, processed, local_probs


def process_video_parallel(
    video_path: Path,
    actions: Tuple[str, ...] = ("emotion",),
    enforce_detection: bool = False,
    workers: int = None
) -> Tuple[int, Dict[str, float], List[Tuple[str, str]], List[List[float]]]:
    """
    Параллельно анализирует видео:
      - возвращает fps,
      - распределение эмоций (по средним «сырым» вероятностям),
      - карту top-2 эмоций по каждому кадру,
      - полные векторы вероятностей эмоций по каждому кадру.
    """
    vid = str(video_path)
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise FileNotFoundError(vid)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if workers is None:
        from multiprocessing import cpu_count
        workers = cpu_count()

    chunk = (total + workers - 1) // workers
    ranges = [(i, min(i + chunk, total)) for i in range(0, total, chunk)]

    emotion_map: List[Tuple[str, str]] = [("", "")] * total
    probs_map: List[List[float]] = [[0.0] * len(DEFAULT_EMOTIONS) for _ in range(total)]
    global_sums = {emo: 0.0 for emo in DEFAULT_EMOTIONS}
    processed = 0

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [
            exe.submit(_analyze_chunk, vid, start, end, actions, enforce_detection, wid)
            for wid, (start, end) in enumerate(ranges)
        ]
        for fut in as_completed(futures):
            local_top2, local_sums, got, local_probs = fut.result()
            for idx, (d, i) in local_top2:
                emotion_map[idx] = (d, i)
            for idx, p in local_probs:
                probs_map[idx] = p
            for emo, s in local_sums.items():
                global_sums[emo] += s
            processed += got

    # если OpenCV завысил число кадров, отбрасываем непрочитанный хвост
    filled = [k for k, (d, _) in enumerate(emotion_map) if d]
    if filled and filled[-1] + 1 < total:
        last = filled[-1] + 1
        emotion_map = emotion_map[:last]
        probs_map = probs_map[:last]

    denom = processed or total
    percentages = {emo: s / denom for emo, s in global_sums.items()}
    return fps, percentages, emotion_map, probs_map


def compare_emotion_maps(
    ref_map: List[Tuple[str, str]],
    test_map: List[Tuple[str, str]],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_shift: int = 50
) -> float:
    """
    Базовый алгоритм: сравнивает две карты эмоций сдвигами (макс. max_shift).
    Возвращает % лучшего совпадения в диапазоне.
    Остальные методы сравнения находятся в модуле comparison.
    """
    return base_similarity(
        ref_map[start_frame:end_frame],
        test_map[start_frame:end_frame],
        max_shift=max_shift
    )


__all__ = [
    "process_video_parallel",
    "save_emotion_map",
    "load_emotion_map",
    "load_emotion_map_ext",
    "compare_emotion_maps",
]
