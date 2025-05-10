# backend.py

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
from deepface import DeepFace
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


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

    cap.release()
    return local_top2, local_sums, processed


def process_video_parallel(
    video_path: Path,
    actions: Tuple[str, ...] = ("emotion",),
    enforce_detection: bool = False,
    workers: int = None
) -> Tuple[int, Dict[str, float], List[Tuple[str, str]]]:
    """
    Параллельно анализирует видео:
      - возвращает fps,
      - распределение эмоций (по средним «сырым» вероятностям),
      - карту top-2 эмоций по каждому кадру.
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
    global_sums = {emo: 0.0 for emo in DEFAULT_EMOTIONS}
    processed = 0

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [
            exe.submit(_analyze_chunk, vid, start, end, actions, enforce_detection, wid)
            for wid, (start, end) in enumerate(ranges)
        ]
        for fut in as_completed(futures):
            local_top2, local_sums, got = fut.result()
            for idx, (d, i) in local_top2:
                emotion_map[idx] = (d, i)
            for emo, s in local_sums.items():
                global_sums[emo] += s
            processed += got

    denom = processed or total
    percentages = {emo: s / denom for emo, s in global_sums.items()}
    return fps, percentages, emotion_map


def save_emotion_map(
    emotion_map: List[Tuple[str, str]],
    percentages: Dict[str, float],
    filename: Path,
    fps: int
) -> None:
    """
    Сохраняет в JSON:
      1) {"fps": fps}
      2) {"percentages": {...}}
      3+) топ-2 по каждому кадру.
    """
    data = [
        {"fps": fps},
        {"percentages": percentages},
    ] + [
        {"dominant": d, "inferior": i}
        for d, i in emotion_map
    ]
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_emotion_map(
    filename: Path
) -> Tuple[int, Dict[str, float], List[Tuple[str, str]]]:
    """
    Читает JSON:
      [ {"fps":…}, {"percentages":…}, {"dominant":…, "inferior":…}, ... ]
    Возвращает (fps, percentages, [(dom,inferior),...]).
    """
    with open(filename, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    if len(data) < 3 or "fps" not in data[0] or "percentages" not in data[1]:
        raise ValueError("Неверный формат JSON: ожидается fps, percentages, затем кадры")

    fps_val = data[0]["fps"]
    if not isinstance(fps_val, (int, float)):
        raise ValueError("fps должен быть числом")
    fps = int(fps_val)

    percentages = data[1]["percentages"]
    if not isinstance(percentages, dict):
        raise ValueError("percentages должен быть объектом")

    emot_list: List[Tuple[str, str]] = []
    for idx, rec in enumerate(data[2:], start=2):
        if not isinstance(rec, dict):
            raise ValueError(f"Элемент {idx}: ожидается объект")
        if "dominant" not in rec or "inferior" not in rec:
            raise KeyError(f"Элемент {idx}: пропущены ключи dominant/inferior")
        d, i = rec["dominant"], rec["inferior"]
        if not isinstance(d, str) or not isinstance(i, str):
            raise ValueError(f"Элемент {idx}: доминирующая/второстепенная не строки")
        emot_list.append((d, i))

    return fps, percentages, emot_list


def compare_emotion_maps(
    ref_map: List[Tuple[str, str]],
    test_map: List[Tuple[str, str]],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_shift: int = 50
) -> float:
    """
    Сравнивает две карты эмоций сдвигами (макс. max_shift).
    Возвращает % лучшего совпадения в диапазоне.
    """
    a = ref_map[start_frame:end_frame]
    b = test_map[start_frame:end_frame]
    total = len(a)
    if total == 0:
        return 0.0

    def match_pct(anchor, moving, shift):
        hits = 0
        for i, (d1, i1) in enumerate(anchor):
            j = i + shift
            if 0 <= j < len(moving):
                d2, i2 = moving[j]
                if d1 == d2 or (d1 == i2 and d2 == i1):
                    hits += 1
        return hits / total * 100.0

    best = 0.0
    for s in range(max_shift + 1):
        best = max(best, match_pct(a, b, s))
    for s in range(1, max_shift + 1):
        best = max(best, match_pct(b, a, s))
    return best


__all__ = [
    "process_video_parallel",
    "save_emotion_map",
    "load_emotion_map",
    "compare_emotion_maps",
]
