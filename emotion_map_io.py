# emotion_map_io.py
"""Чтение и запись эмоциональных карт в формате JSON (без зависимостей от OpenCV и DeepFace)."""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

DEFAULT_EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def save_emotion_map(
    emotion_map: List[Tuple[str, str]],
    percentages: Dict[str, float],
    filename: Path,
    fps: int,
    probs: Optional[List[List[float]]] = None
) -> None:
    """
    Сохраняет в JSON:
      1) {"fps": fps}
      2) {"percentages": {...}}
      3+) топ-2 по каждому кадру и (если есть) полный вектор вероятностей
          {"dominant": ..., "inferior": ..., "probs": {"angry": ..., ...}}.
    Поле "probs" необязательное, поэтому старые карты читаются без изменений.
    """
    frames = []
    for k, (d, i) in enumerate(emotion_map):
        rec = {"dominant": d, "inferior": i}
        if probs is not None and k < len(probs) and probs[k] is not None:
            rec["probs"] = {e: round(float(p), 4) for e, p in zip(DEFAULT_EMOTIONS, probs[k])}
        frames.append(rec)
    data = [
        {"fps": fps},
        {"percentages": percentages},
    ] + frames
    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_emotion_map_ext(
    filename: Path
) -> Tuple[int, Dict[str, float], List[Tuple[str, str]], Optional[List[List[float]]]]:
    """
    Читает JSON:
      [ {"fps":…}, {"percentages":…}, {"dominant":…, "inferior":…[, "probs": {...}]}, ... ]
    Возвращает (fps, percentages, [(dom, inferior), ...], probs или None).
    probs возвращаются, только если вектор вероятностей есть у каждого кадра.
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
    probs: List[List[float]] = []
    has_probs = True
    for idx, rec in enumerate(data[2:], start=2):
        if not isinstance(rec, dict):
            raise ValueError(f"Элемент {idx}: ожидается объект")
        if "dominant" not in rec or "inferior" not in rec:
            raise KeyError(f"Элемент {idx}: пропущены ключи dominant/inferior")
        d, i = rec["dominant"], rec["inferior"]
        if not isinstance(d, str) or not isinstance(i, str):
            raise ValueError(f"Элемент {idx}: доминирующая/второстепенная не строки")
        emot_list.append((d, i))
        p = rec.get("probs")
        if has_probs and isinstance(p, dict):
            probs.append([float(p.get(e, 0.0)) for e in DEFAULT_EMOTIONS])
        else:
            has_probs = False

    return fps, percentages, emot_list, (probs if has_probs else None)


def load_emotion_map(
    filename: Path
) -> Tuple[int, Dict[str, float], List[Tuple[str, str]]]:
    """Совместимая со старой версией функция: (fps, percentages, [(dom, inferior), ...])."""
    fps, percentages, emot_list, _ = load_emotion_map_ext(filename)
    return fps, percentages, emot_list


__all__ = ["save_emotion_map", "load_emotion_map_ext", "load_emotion_map"]
