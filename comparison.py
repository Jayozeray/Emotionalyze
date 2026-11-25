# comparison.py
"""
Методы сравнения эмоциональных карт (ЭК).

Модуль не зависит от OpenCV и DeepFace, поэтому его можно использовать
отдельно от анализа видео, например в скрипте экспериментов.

Реализованы:
  - базовый алгоритм (сравнение доминантной и уступающей эмоции со сдвигами),
  - векторный метод (косинусная близость векторов эмоций со сдвигами),
  - динамическое выравнивание временных рядов (DTW) с полосой Сакоэ-Чибы,
  - сравнение частотных распределений эмоций (пересечение гистограмм),
  - комбинированная оценка (взвешенная сумма перечисленных методов).

Дополнительно:
  - сглаживание ЭК скользящим средним по векторному представлению,
  - исключение безэмоциональных фрагментов (пар кадров, где в обеих картах
    доминирует нейтральная эмоция).
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
EMO_INDEX = {e: k for k, e in enumerate(EMOTIONS)}
NEUTRAL = "neutral"
NEUTRAL_IDX = EMO_INDEX[NEUTRAL]

METHOD_BASE = "base"
METHOD_VECTOR = "vector"
METHOD_DTW = "dtw"
METHOD_FREQ = "freq"
METHOD_COMBINED = "combined"

# порядок важен: первые четыре входят в комбинированную оценку
COMPONENT_METHODS = (METHOD_BASE, METHOD_VECTOR, METHOD_DTW, METHOD_FREQ)
METHODS = COMPONENT_METHODS + (METHOD_COMBINED,)

METHOD_TITLES = {
    METHOD_BASE: "Базовый алгоритм",
    METHOD_VECTOR: "Векторный метод",
    METHOD_DTW: "Динамическое выравнивание (DTW)",
    METHOD_FREQ: "Частотное распределение",
    METHOD_COMBINED: "Комбинированная оценка",
}

DEFAULT_WEIGHTS = (0.25, 0.35, 0.30, 0.10)       # экспертные веса из отчёта за 2 семестр

# Рекомендуемая по результатам экспериментов конфигурация:
# комбинированная оценка из динамического выравнивания и частотного распределения
# с исключением безэмоциональных фрагментов, максимальный сдвиг 2 с, без сглаживания.
RECOMMENDED_WEIGHTS = (0.0, 0.0, 0.5, 0.5)
RECOMMENDED_SHIFT_SECONDS = 2.0
RECOMMENDED_SKIP_NEUTRAL = True

Labels = List[Tuple[str, str]]


@dataclass
class CompareParams:
    max_shift: int = 50            # максимальный сдвиг кадров (S в формулах)
    alpha: float = 0.5             # вес уступающей эмоции в приближённом векторе
    smooth_window: int = 0         # окно сглаживания, кадров (0 или 1 - без сглаживания)
    skip_neutral: bool = False     # исключать пары "нейтральная - нейтральная"
    use_probs: bool = True         # использовать полные векторы вероятностей, если они есть
    weights: Tuple[float, float, float, float] = DEFAULT_WEIGHTS

    def describe(self) -> str:
        w = ", ".join(f"{x:.2f}" for x in self.weights)
        return (f"сдвиг={self.max_shift}, alpha={self.alpha:.2f}, "
                f"сглаживание={self.smooth_window}, "
                f"искл. нейтр.={'да' if self.skip_neutral else 'нет'}, "
                f"полные векторы={'да' if self.use_probs else 'нет'}, веса=({w})")


# ---------------------------------------------------------------------------
# Представление эмоциональной карты
# ---------------------------------------------------------------------------

def probs_to_array(probs, n: int) -> Optional[np.ndarray]:
    """Приводит список векторов вероятностей (списки или словари) к массиву (n, 7)."""
    if probs is None or len(probs) != n or n == 0:
        return None
    arr = np.zeros((n, len(EMOTIONS)), dtype=float)
    for k, p in enumerate(probs):
        if p is None:
            continue
        if isinstance(p, dict):
            for e, val in p.items():
                if e in EMO_INDEX:
                    arr[k, EMO_INDEX[e]] = float(val)
        else:
            arr[k, :] = np.asarray(p, dtype=float)[:len(EMOTIONS)]
    return arr


def to_vectors(labels: Labels, probs: Optional[np.ndarray] = None,
               alpha: float = 0.5, use_probs: bool = True) -> np.ndarray:
    """
    Векторное представление ЭК.
    При наличии полных вероятностей - формула (7), иначе приближение по
    доминантной и уступающей эмоции - формула (9): 1 для доминантной,
    alpha для уступающей, 0 для остальных.
    """
    n = len(labels)
    if use_probs and probs is not None and len(probs) == n:
        v = np.clip(np.asarray(probs, dtype=float), 0.0, None)
        s = v.sum(axis=1, keepdims=True)
        return np.divide(v, s, out=np.zeros_like(v), where=s > 0)
    v = np.zeros((n, len(EMOTIONS)), dtype=float)
    for k, (d, c) in enumerate(labels):
        if d in EMO_INDEX:
            v[k, EMO_INDEX[d]] = 1.0
        if c in EMO_INDEX and c != d:
            v[k, EMO_INDEX[c]] = max(v[k, EMO_INDEX[c]], alpha)
    return v


def labels_from_vectors(v: np.ndarray) -> Labels:
    """Доминантная и уступающая эмоции по вектору (для нулевого вектора - пустые)."""
    out: Labels = []
    if len(v) == 0:
        return out
    order = np.argsort(-v, axis=1, kind="stable")
    nonzero = v.max(axis=1) > 0
    for k in range(len(v)):
        if not nonzero[k]:
            out.append(("", ""))
        else:
            out.append((EMOTIONS[order[k, 0]], EMOTIONS[order[k, 1]]))
    return out


def smooth_vectors(v: np.ndarray, window: int) -> np.ndarray:
    """Сглаживание скользящим средним с окном window кадров (центрированное)."""
    n = len(v)
    if window is None or window <= 1 or n == 0:
        return v
    half = window // 2
    csum = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx + half + 1, 0, n)
    return (csum[hi] - csum[lo]) / (hi - lo)[:, None]


@dataclass
class PreparedMap:
    labels: Labels
    vectors: np.ndarray

    def __len__(self):
        return len(self.labels)

    def slice(self, start: int, end: Optional[int]) -> "PreparedMap":
        return PreparedMap(self.labels[start:end], self.vectors[start:end])

    def neutral_mask(self) -> np.ndarray:
        return np.array([d == NEUTRAL for d, _ in self.labels], dtype=bool)


def prepare_map(labels: Labels, probs=None, params: Optional[CompareParams] = None) -> PreparedMap:
    """Строит векторы, при необходимости сглаживает и пересчитывает доминантную и уступающую эмоции."""
    params = params or CompareParams()
    probs_arr = probs if isinstance(probs, np.ndarray) else probs_to_array(probs, len(labels))
    v = to_vectors(labels, probs_arr, params.alpha, params.use_probs)
    lab = list(labels)
    if params.smooth_window and params.smooth_window > 1:
        v = smooth_vectors(v, params.smooth_window)
        lab = labels_from_vectors(v)
    return PreparedMap(lab, v)


# ---------------------------------------------------------------------------
# Базовый алгоритм, формулы (1)-(4)
# ---------------------------------------------------------------------------

def _encode_pair(a: Labels, b: Labels):
    codes = {e: k for k, e in enumerate(EMOTIONS)}

    def enc(labels):
        d = np.empty(len(labels), dtype=np.int64)
        c = np.empty(len(labels), dtype=np.int64)
        for k, (x, y) in enumerate(labels):
            d[k] = codes.setdefault(x, len(codes))
            c[k] = codes.setdefault(y, len(codes))
        return d, c

    return enc(a), enc(b)


def base_similarity(a: Labels, b: Labels, max_shift: int = 50,
                    skip_neutral: bool = False) -> float:
    """
    Базовый алгоритм из ВКР. Без исключения нейтральных фрагментов результат
    совпадает с прежней функцией compare_emotion_maps.
    """
    total = len(a)
    if total == 0:
        return 0.0
    (da, ca), (db, cb) = _encode_pair(a, b)

    def match_pct(d1, c1, d2, c2, shift):
        length = min(len(d1), len(d2) - shift)
        if length <= 0:
            return 0.0
        x_d, x_c = d1[:length], c1[:length]
        y_d, y_c = d2[shift:shift + length], c2[shift:shift + length]
        hit = (x_d == y_d) | ((x_d == y_c) & (y_d == x_c))
        denom = total
        if skip_neutral:
            nn = (x_d == NEUTRAL_IDX) & (y_d == NEUTRAL_IDX)
            if int(nn.sum()) == length:
                return math.nan   # в перекрытии только безэмоциональные пары
            hit = hit & ~nn
            denom = total - int(nn.sum())
        return float(hit.sum()) / denom * 100.0

    vals = [match_pct(da, ca, db, cb, s) for s in range(max_shift + 1)]
    vals += [match_pct(db, cb, da, ca, s) for s in range(1, max_shift + 1)]
    vals = [x for x in vals if not math.isnan(x)]
    return max(vals) if vals else math.nan


# ---------------------------------------------------------------------------
# Векторный метод, формулы (10)-(13)
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def vector_similarity(a: PreparedMap, b: PreparedMap, max_shift: int = 50,
                      skip_neutral: bool = False) -> float:
    ua, ub = _unit(a.vectors), _unit(b.vectors)
    na, nb = a.neutral_mask(), b.neutral_mask()
    n, m = len(ua), len(ub)
    if n == 0 or m == 0:
        return 0.0
    best = -math.inf
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            length = min(n, m - s)
            if length <= 0:
                continue
            x, y = ua[:length], ub[s:s + length]
            mx, my = na[:length], nb[s:s + length]
        else:
            length = min(n + s, m)
            if length <= 0:
                continue
            x, y = ua[-s:-s + length], ub[:length]
            mx, my = na[-s:-s + length], nb[:length]
        cos = np.einsum("ij,ij->i", x, y)
        if skip_neutral:
            keep = ~(mx & my)
            if not keep.any():
                continue
            cos = cos[keep]
        best = max(best, float(cos.mean()))
    return min(best, 1.0) * 100.0 if best > -math.inf else math.nan


# ---------------------------------------------------------------------------
# Динамическое выравнивание временных рядов, формулы (14)-(16)
# ---------------------------------------------------------------------------

def dtw_path(a: PreparedMap, b: PreparedMap, window: int = 50):
    """
    DTW с полосой Сакоэ-Чибы ширины window вокруг диагонали.
    Стоимость сопоставления кадров - 1 - cos (формула 14).
    Возвращает накопленную стоимость D(N, M) и оптимальный путь.
    """
    ua, ub = _unit(a.vectors), _unit(b.vectors)
    n, m = len(ua), len(ub)
    w = max(int(window), abs(n - m), 1)
    if n > 1:
        centers = np.rint(np.arange(n) * (m - 1) / (n - 1)).astype(int)
    else:
        centers = np.array([m - 1])
    lo = np.maximum(centers - w, 0)
    hi = np.minimum(centers + w, m - 1)
    lo[0] = 0
    hi[-1] = m - 1

    rows: List[np.ndarray] = []
    cost0 = 1.0 - ub[lo[0]:hi[0] + 1] @ ua[0]
    prev = np.cumsum(cost0)
    prev_lo = lo[0]
    rows.append(prev)

    def take(row, row_lo, js):
        out = np.full(len(js), np.inf)
        k = js - row_lo
        ok = (k >= 0) & (k < len(row))
        out[ok] = row[k[ok]]
        return out

    for i in range(1, n):
        l, h = lo[i], hi[i]
        js = np.arange(l, h + 1)
        c = 1.0 - ub[l:h + 1] @ ua[i]
        # A[j] = min(D(i-1, j), D(i-1, j-1)); D(i, j) = c + min(A[j], D(i, j-1))
        a_min = np.minimum(take(prev, prev_lo, js), take(prev, prev_lo, js - 1))
        csum = np.cumsum(c)
        cshift = np.concatenate(([0.0], csum[:-1]))
        cur = csum + np.minimum.accumulate(a_min - cshift)
        rows.append(cur)
        prev, prev_lo = cur, l

    def val(i, j):
        if i < 0 or j < 0:
            return np.inf
        k = j - lo[i]
        if 0 <= k < len(rows[i]):
            return rows[i][k]
        return np.inf

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        cand = ((val(i - 1, j - 1), i - 1, j - 1),
                (val(i - 1, j), i - 1, j),
                (val(i, j - 1), i, j - 1))
        _, i, j = min(cand, key=lambda t: t[0])
        path.append((i, j))
    path.reverse()
    return float(rows[-1][-1]), path


def dtw_similarity(a: PreparedMap, b: PreparedMap, window: int = 50,
                   skip_neutral: bool = False) -> float:
    """
    P_dtw = 100 * (1 - D_norm), где D_norm - средняя стоимость сопоставления
    на оптимальном пути. При исключении нейтральных фрагментов среднее
    считается только по парам, в которых хотя бы один кадр не нейтральный.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    _, path = dtw_path(a, b, window)
    ua, ub = _unit(a.vectors), _unit(b.vectors)
    ii = np.array([p[0] for p in path])
    jj = np.array([p[1] for p in path])
    delta = 1.0 - np.einsum("ij,ij->i", ua[ii], ub[jj])
    if skip_neutral:
        keep = ~(a.neutral_mask()[ii] & b.neutral_mask()[jj])
        if not keep.any():
            return math.nan
        delta = delta[keep]
    d_norm = float(np.clip(delta.mean(), 0.0, 1.0))
    return 100.0 * (1.0 - d_norm)


# ---------------------------------------------------------------------------
# Частотное распределение эмоций
# ---------------------------------------------------------------------------

def emotion_frequencies(labels: Labels, skip_neutral: bool = False) -> np.ndarray:
    """Доли кадров с каждой доминантной эмоцией."""
    f = np.zeros(len(EMOTIONS), dtype=float)
    for d, _ in labels:
        if d in EMO_INDEX:
            f[EMO_INDEX[d]] += 1.0
    if skip_neutral:
        f[NEUTRAL_IDX] = 0.0
    s = f.sum()
    return f / s if s > 0 else f


def freq_similarity(a: Labels, b: Labels, skip_neutral: bool = False) -> float:
    """P_freq = 100 * sum_k min(f_k^R, f_k^T) - пересечение гистограмм."""
    fa = emotion_frequencies(a, skip_neutral)
    fb = emotion_frequencies(b, skip_neutral)
    if fa.sum() == 0 or fb.sum() == 0:
        return math.nan
    return 100.0 * float(np.minimum(fa, fb).sum())


# ---------------------------------------------------------------------------
# Комбинированная оценка, формула (17)
# ---------------------------------------------------------------------------

def combine(values: Dict[str, float], weights: Sequence[float] = DEFAULT_WEIGHTS) -> float:
    num, den = 0.0, 0.0
    for name, w in zip(COMPONENT_METHODS, weights):
        x = values.get(name, math.nan)
        if w > 0 and x is not None and not math.isnan(x):
            num += w * x
            den += w
    return num / den if den > 0 else math.nan


# ---------------------------------------------------------------------------
# Общая точка входа
# ---------------------------------------------------------------------------

def compare_prepared(ref: PreparedMap, test: PreparedMap,
                     methods: Iterable[str] = METHODS,
                     params: Optional[CompareParams] = None) -> Dict[str, float]:
    params = params or CompareParams()
    methods = list(methods)
    need = set(methods)
    if METHOD_COMBINED in need:
        need.update(name for name, w in zip(COMPONENT_METHODS, params.weights) if w > 0)
    res: Dict[str, float] = {}
    if METHOD_BASE in need:
        res[METHOD_BASE] = base_similarity(ref.labels, test.labels, params.max_shift, params.skip_neutral)
    if METHOD_VECTOR in need:
        res[METHOD_VECTOR] = vector_similarity(ref, test, params.max_shift, params.skip_neutral)
    if METHOD_DTW in need:
        res[METHOD_DTW] = dtw_similarity(ref, test, params.max_shift, params.skip_neutral)
    if METHOD_FREQ in need:
        res[METHOD_FREQ] = freq_similarity(ref.labels, test.labels, params.skip_neutral)
    if METHOD_COMBINED in need:
        res[METHOD_COMBINED] = combine(res, params.weights)
    return {k: res[k] for k in methods if k in res}


def compare_maps(ref_labels: Labels, test_labels: Labels,
                 ref_probs=None, test_probs=None,
                 start_frame: int = 0, end_frame: Optional[int] = None,
                 methods: Iterable[str] = METHODS,
                 params: Optional[CompareParams] = None) -> Dict[str, float]:
    """
    Сравнивает эталонную и тестируемую карты на интервале кадров
    [start_frame, end_frame). Сглаживание выполняется по всей карте,
    затем вырезается интервал.
    """
    params = params or CompareParams()
    ref = prepare_map(ref_labels, ref_probs, params).slice(start_frame, end_frame)
    test = prepare_map(test_labels, test_probs, params).slice(start_frame, end_frame)
    return compare_prepared(ref, test, methods, params)


def verdict(test_scores: Sequence[float], ref_pair_scores: Sequence[float]):
    """
    Правило принятия решения: среднее сходство тестируемой карты с эталонами
    сравнивается со средним сходством эталонов между собой (критерием).
    Возвращает (среднее сходство, критерий, соответствует ли реакция эталонным).
    """
    ts = [x for x in test_scores if x is not None and not math.isnan(x)]
    rs = [x for x in ref_pair_scores if x is not None and not math.isnan(x)]
    if not ts or not rs:
        return math.nan, math.nan, None
    mean_test = float(np.mean(ts))
    crit = float(np.mean(rs))
    return mean_test, crit, mean_test >= crit


def format_value(x: float) -> str:
    return "н/д" if x is None or math.isnan(x) else f"{x:.2f}%"


__all__ = [
    "EMOTIONS", "METHODS", "COMPONENT_METHODS", "METHOD_TITLES", "DEFAULT_WEIGHTS",
    "METHOD_BASE", "METHOD_VECTOR", "METHOD_DTW", "METHOD_FREQ", "METHOD_COMBINED",
    "RECOMMENDED_WEIGHTS", "RECOMMENDED_SHIFT_SECONDS", "RECOMMENDED_SKIP_NEUTRAL", "verdict",
    "CompareParams", "PreparedMap", "prepare_map", "compare_prepared", "compare_maps",
    "base_similarity", "vector_similarity", "dtw_similarity", "freq_similarity",
    "combine", "smooth_vectors", "to_vectors", "labels_from_vectors",
    "emotion_frequencies", "probs_to_array", "format_value",
]
