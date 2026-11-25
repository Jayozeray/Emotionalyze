# experiments.py
"""
Экспериментальное исследование методов сравнения эмоциональных карт.

Запуск:
    python experiments.py --config experiment_config.json --out results

Формат конфигурации (время в секундах, end = null - до конца карты):
{
  "maps_dir": "maps",
  "test_person": "6",
  "maps": [
    {"file": "EM_1_video1.json", "person": "1", "video": "1"},
    ...
  ],
  "segments": [
    {"name": "1",   "video": "1", "start": 0,  "end": null},
    {"name": "3.1", "video": "3", "start": 0,  "end": 25},
    ...
  ]
}

Эксперименты:
  e0  воспроизведение результатов ВКР базовым алгоритмом (межэталонное
      среднее и сравнение тестируемого с эталонами);
  e1  то же для всех методов и вариантов предобработки;
  e2  разделяющая способность: пары карт одного отрезка (соответствующие
      реакции) против пар карт видеозаписей с разной целевой эмоцией
      (несоответствующие реакции), площадь под ROC-кривой и проверка
      по критерию межэталонного сравнения с поочерёдным исключением;
  e3  устойчивость к шуму распознавания (случайная замена кадров);
  e4  устойчивость к безэмоциональным фрагментам (общий нейтральный участок);
  e5  устойчивость к запаздыванию и изменению темпа реакции;
  e6  подбор весов комбинированной оценки;
  e7  время вычисления.
"""

import argparse
import csv
import json
import math
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from comparison import (
    EMOTIONS, COMPONENT_METHODS, METHODS, METHOD_TITLES, METHOD_COMBINED,
    DEFAULT_WEIGHTS, CompareParams, prepare_map, base_similarity,
    vector_similarity, dtw_similarity, freq_similarity, combine,
)
from emotion_map_io import load_emotion_map_ext

VARIANT_TITLES = {
    "raw": "без предобработки",
    "nn": "исключение нейтральных фрагментов",
    "sm": "сглаживание",
    "sm_nn": "сглаживание и исключение нейтральных фрагментов",
}


def variant_params(variant: str, base: CompareParams, smooth: int) -> CompareParams:
    return CompareParams(
        max_shift=base.max_shift, alpha=base.alpha, use_probs=base.use_probs,
        weights=base.weights,
        smooth_window=smooth if variant in ("sm", "sm_nn") else 0,
        skip_neutral=variant in ("nn", "sm_nn"),
    )


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

class Dataset:
    def __init__(self, config_path: Path):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        root = Path(config_path).parent / cfg.get("maps_dir", ".")
        self.test_person = str(cfg.get("test_person", ""))
        self.maps = {}
        for rec in cfg["maps"]:
            fps, _perc, labels, probs = load_emotion_map_ext(root / rec["file"])
            key = (str(rec["person"]), str(rec["video"]))
            self.maps[key] = {
                "file": rec["file"], "fps": fps, "labels": labels,
                "probs": np.asarray(probs) if probs is not None else None,
            }
        self.segments = [dict(s, name=str(s["name"]), video=str(s["video"])) for s in cfg["segments"]]
        self.persons = sorted({p for p, _ in self.maps}, key=lambda x: (len(x), x))
        self.all_probs = all(m["probs"] is not None for m in self.maps.values())

    def segment(self, person: str, seg: dict, length: Optional[int] = None):
        m = self.maps.get((person, seg["video"]))
        if m is None:
            return None
        fps = m["fps"]
        sf = int(round(seg.get("start", 0) * fps))
        ef = None if seg.get("end") is None else int(round(seg["end"] * fps))
        labels = m["labels"][sf:ef]
        probs = m["probs"][sf:ef] if m["probs"] is not None else None
        if length is not None:
            labels = labels[:length]
            probs = probs[:length] if probs is not None else None
        return labels, probs

    def seg_len(self, seg: dict) -> int:
        lens = [len(self.segment(p, seg)[0]) for p in self.persons if self.segment(p, seg) is not None]
        return min(lens) if lens else 0

    def persons_for(self, seg: dict) -> List[str]:
        return [p for p in self.persons if (p, seg["video"]) in self.maps]


# ---------------------------------------------------------------------------
# Искажения карт
# ---------------------------------------------------------------------------

def add_noise(labels, probs, level: float, rng: random.Random):
    """Заменяет долю level кадров случайными эмоциями (имитация ошибок распознавания)."""
    labels = list(labels)
    probs = None if probs is None else np.array(probs, dtype=float)
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    for k in range(len(labels)):
        if rng.random() < level:
            d = rng.choice(EMOTIONS)
            c = rng.choice([e for e in EMOTIONS if e != d])
            if probs is not None:
                v = nprng.dirichlet(np.ones(len(EMOTIONS)))
                order = np.argsort(-v)
                probs[k] = v
                d, c = EMOTIONS[order[0]], EMOTIONS[order[1]]
            labels[k] = (d, c)
    return labels, probs


def neutral_frame(probs_dim: bool):
    lab = ("neutral", "sad")
    vec = None
    if probs_dim:
        vec = np.zeros(len(EMOTIONS))
        vec[EMOTIONS.index("neutral")] = 0.7
        vec[EMOTIONS.index("sad")] = 0.3
    return lab, vec


def prepend_neutral(labels, probs, count: int):
    lab, vec = neutral_frame(probs is not None)
    new_labels = [lab] * count + list(labels)
    new_probs = None if probs is None else np.vstack([np.tile(vec, (count, 1)), probs])
    return new_labels, new_probs


def delay(labels, probs, frames: int):
    """Запаздывание реакции: начало карты повторяется, хвост отбрасывается."""
    if frames <= 0:
        return list(labels), probs
    n = len(labels)
    idx = np.concatenate([np.zeros(frames, dtype=int), np.arange(n - frames)])
    idx = np.clip(idx, 0, n - 1)
    return [labels[i] for i in idx], (None if probs is None else probs[idx])


def stretch(labels, probs, factor: float):
    """Изменение темпа реакции с сохранением длины карты."""
    n = len(labels)
    idx = np.clip(np.floor(np.arange(n) / factor).astype(int), 0, n - 1)
    return [labels[i] for i in idx], (None if probs is None else probs[idx])


# ---------------------------------------------------------------------------
# Вычисление сходства
# ---------------------------------------------------------------------------

def components(a, b, params: CompareParams, with_time: bool = False):
    """Значения четырёх базовых методов (комбинированная оценка считается позже)."""
    A = prepare_map(a[0], a[1], params)
    B = prepare_map(b[0], b[1], params)
    out, times = {}, {}
    t = time.perf_counter()
    out["base"] = base_similarity(A.labels, B.labels, params.max_shift, params.skip_neutral)
    times["base"] = time.perf_counter() - t
    t = time.perf_counter()
    out["vector"] = vector_similarity(A, B, params.max_shift, params.skip_neutral)
    times["vector"] = time.perf_counter() - t
    t = time.perf_counter()
    out["dtw"] = dtw_similarity(A, B, params.max_shift, params.skip_neutral)
    times["dtw"] = time.perf_counter() - t
    t = time.perf_counter()
    out["freq"] = freq_similarity(A.labels, B.labels, params.skip_neutral)
    times["freq"] = time.perf_counter() - t
    return (out, times) if with_time else out


def _task(args):
    key, a, b, params = args
    vals, times = components(a, b, params, with_time=True)
    return key, vals, times


def run_tasks(tasks, workers: int):
    if workers <= 1:
        return [_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as exe:
        return list(exe.map(_task, tasks, chunksize=4))


COMBINED_WEIGHTS = list(DEFAULT_WEIGHTS)


def with_combined(vals: Dict[str, float], weights=None) -> Dict[str, float]:
    res = dict(vals)
    res[METHOD_COMBINED] = combine(vals, weights if weights is not None else COMBINED_WEIGHTS)
    return res


def auc(pos: List[float], neg: List[float]) -> float:
    pos = np.array([x for x in pos if not math.isnan(x)])
    neg = np.array([x for x in neg if not math.isnan(x)])
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def nanmean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.mean(xs)) if xs else math.nan


def nanstd(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.std(xs)) if xs else math.nan


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}".replace(".", ",") if isinstance(v, float) else v) for k, v in r.items()})


# ---------------------------------------------------------------------------
# Пары для экспериментов e1, e2, e6
# ---------------------------------------------------------------------------

def pair_specs(ds: Dataset):
    """
    Возвращает словарь key -> (a, b), где key = (вид, ...).
      ("pos", g, p, q)          - одна видеозапись, разные зрители, полный отрезок;
      ("ref", g, L, p, q)       - то же, отрезок обрезан до L кадров (для критерия);
      ("neg", g, h, L, p, q)    - зритель p на отрезке h против зрителя q на отрезке g,
                                  видеозаписи g и h с разной целевой эмоцией.
    """
    specs = {}
    segs = ds.segments
    for g in segs:
        persons = ds.persons_for(g)
        for p, q in combinations(persons, 2):
            specs[("pos", g["name"], p, q)] = (ds.segment(p, g), ds.segment(q, g))
    for g, h in product(segs, segs):
        if g["video"] == h["video"]:
            continue
        L = min(ds.seg_len(g), ds.seg_len(h))
        if L < ds.seg_len(g):
            for p, q in combinations(ds.persons_for(g), 2):
                specs.setdefault(("ref", g["name"], L, p, q), (ds.segment(p, g, L), ds.segment(q, g, L)))
        for p in ds.persons_for(h):
            for q in ds.persons_for(g):
                if p == q:
                    continue
                specs[("neg", g["name"], h["name"], L, p, q)] = (ds.segment(q, g, L), ds.segment(p, h, L))
    return specs


def sim_lookup(cache, variant, key):
    return cache[(variant,) + key]


def ref_pair_key(g, L_full, L, p, q):
    a, b = sorted([p, q], key=lambda x: (len(x), x))
    return ("pos", g, a, b) if L >= L_full else ("ref", g, L, a, b)


def loo_evaluation(ds: Dataset, cache, variant: str, weights=None):
    """
    Проверка по критерию межэталонного сравнения (как в ВКР) с поочерёдным
    исключением: каждый зритель по очереди считается тестируемым,
    остальные - эталонными.
    """
    results = {m: {"pos_ok": 0, "pos_n": 0, "neg_ok": 0, "neg_n": 0} for m in METHODS}
    segs = {s["name"]: s for s in ds.segments}

    def val(key):
        return with_combined(sim_lookup(cache, variant, key), weights)

    for g in ds.segments:
        persons = ds.persons_for(g)
        Lg = ds.seg_len(g)
        for p in persons:
            refs = [q for q in persons if q != p]
            if len(refs) < 2:
                continue
            # соответствующая реакция
            crit = {m: nanmean([val(ref_pair_key(g["name"], Lg, Lg, q, r))[m]
                                for q, r in combinations(refs, 2)]) for m in METHODS}
            score = {m: max([val(ref_pair_key(g["name"], Lg, Lg, p, q))[m] for q in refs],
                            default=math.nan) for m in METHODS}
            for m in METHODS:
                results[m]["pos_n"] += 1
                if not math.isnan(score[m]) and score[m] >= crit[m]:
                    results[m]["pos_ok"] += 1
            # несоответствующие реакции: тот же зритель на другой видеозаписи
            for h in ds.segments:
                if h["video"] == g["video"] or (p, h["video"]) not in ds.maps:
                    continue
                L = min(Lg, ds.seg_len(h))
                crit_l = {m: nanmean([val(ref_pair_key(g["name"], Lg, L, q, r))[m]
                                      for q, r in combinations(refs, 2)]) for m in METHODS}
                score_l = {m: max([val(("neg", g["name"], h["name"], L, p, q))[m] for q in refs],
                                  default=math.nan) for m in METHODS}
                for m in METHODS:
                    results[m]["neg_n"] += 1
                    if math.isnan(score_l[m]) or score_l[m] < crit_l[m]:
                        results[m]["neg_ok"] += 1
    return results


# ---------------------------------------------------------------------------
# Основная процедура
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Исследование методов сравнения эмоциональных карт")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default="e0,e1,e2,e3,e4,e5,e6,e7",
                    help="список экспериментов через запятую")
    ap.add_argument("--max-shift", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=15, help="окно сглаживания, кадров")
    ap.add_argument("--no-probs", action="store_true", help="не использовать полные векторы")
    ap.add_argument("--noise", default="0.1,0.2,0.3")
    ap.add_argument("--neutral", default="0.5,1.0", help="длина нейтрального участка в долях отрезка")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--weights", default=None, help="веса комбинированной оценки через запятую")
    ap.add_argument("--variants", default=",".join(VARIANT_TITLES), help="варианты предобработки")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(","))
    ds = Dataset(Path(args.config))
    weights = tuple(float(x) for x in args.weights.split(",")) if args.weights else DEFAULT_WEIGHTS
    base_params = CompareParams(max_shift=args.max_shift, alpha=args.alpha,
                                use_probs=ds.all_probs and not args.no_probs, weights=weights)
    variants = [v for v in args.variants.split(",") if v in VARIANT_TITLES]
    COMBINED_WEIGHTS[:] = list(weights)
    vparams = {v: variant_params(v, base_params, args.smooth) for v in variants}
    print(f"Карт: {len(ds.maps)}, зрителей: {len(ds.persons)}, отрезков: {len(ds.segments)}, "
          f"полные векторы: {'да' if base_params.use_probs else 'нет'}")

    # --- статистика нейтральных кадров ---
    stats = []
    for g in ds.segments:
        for p in ds.persons_for(g):
            labels, _ = ds.segment(p, g)
            n = len(labels)
            neu = sum(1 for d, _ in labels if d == "neutral")
            stats.append({"Отрезок": g["name"], "Зритель": p, "Кадров": n,
                          "Доля нейтральных кадров": neu / n if n else math.nan})
    write_csv(out / "stats_neutral.csv", stats)

    # --- кэш попарных сравнений ---
    cache_path = out / "cache_pairs.pkl"
    cache, timing = {}, {m: [] for m in COMPONENT_METHODS}
    if cache_path.exists():
        cache, timing = pickle.loads(cache_path.read_bytes())
    if only & {"e0", "e1", "e2", "e6", "e7"}:
        specs = pair_specs(ds)
        tasks = [((v,) + key, a, b, vparams[v]) for v in variants for key, (a, b) in specs.items()
                 if (v,) + key not in cache]
        if tasks:
            print(f"Попарных сравнений: {len(tasks)}")
            t0 = time.time()
            for key, vals, times in run_tasks(tasks, args.workers):
                cache[key] = vals
                if key[0] == "raw":
                    for m, t in times.items():
                        timing[m].append(t)
            print(f"  готово за {time.time() - t0:.1f} с")
            cache_path.write_bytes(pickle.dumps((cache, timing)))

    test = ds.test_person

    # --- e0, e1: таблицы как в ВКР ---
    if only & {"e0", "e1"}:
        rows_crit, rows_test = [], []
        for v in variants:
            for g in ds.segments:
                persons = ds.persons_for(g)
                refs = [q for q in persons if q != test]
                Lg = ds.seg_len(g)
                crit = {m: nanmean([with_combined(sim_lookup(cache, v, ref_pair_key(g["name"], Lg, Lg, q, r)))[m]
                                    for q, r in combinations(refs, 2)]) for m in METHODS}
                for m in METHODS:
                    row = {"Вариант": VARIANT_TITLES[v], "Метод": METHOD_TITLES[m], "Отрезок": g["name"],
                           "Среднее межэталонное, %": crit[m]}
                    best = -1.0
                    for q in refs:
                        val = with_combined(sim_lookup(cache, v, ref_pair_key(g["name"], Lg, Lg, test, q)))[m]
                        row[f"Эталон {q}, %"] = val
                        if not math.isnan(val):
                            best = max(best, val)
                    row["Лучшее совпадение, %"] = best
                    row["Критерий достигнут"] = "да" if best >= crit[m] else "нет"
                    rows_test.append(row)
        write_csv(out / "e1_test_vs_refs.csv", rows_test)
        write_csv(out / "e0_reproduction.csv",
                  [r for r in rows_test if r["Вариант"] == VARIANT_TITLES["raw"]
                   and r["Метод"] == METHOD_TITLES["base"]])

    # --- e2: разделяющая способность ---
    if "e2" in only:
        rows_pairs, rows_sum = [], []
        for v in variants:
            pos = {m: [] for m in METHODS}
            neg = {m: [] for m in METHODS}
            for key, vals in cache.items():
                if key[0] != v or key[1] not in ("pos", "neg"):
                    continue
                full = with_combined(vals)
                for m in METHODS:
                    (pos if key[1] == "pos" else neg)[m].append(full[m])
                rows_pairs.append(dict({"Вариант": v, "Вид": key[1], "Ключ": "|".join(map(str, key[2:]))},
                                       **{METHOD_TITLES[m]: full[m] for m in METHODS}))
            loo = loo_evaluation(ds, cache, v)
            for m in METHODS:
                r = loo[m]
                tpr = r["pos_ok"] / r["pos_n"] if r["pos_n"] else math.nan
                tnr = r["neg_ok"] / r["neg_n"] if r["neg_n"] else math.nan
                rows_sum.append({
                    "Вариант": VARIANT_TITLES[v], "Метод": METHOD_TITLES[m],
                    "Среднее (соответствующие), %": nanmean(pos[m]),
                    "СКО (соответствующие), %": nanstd(pos[m]),
                    "Среднее (несоответствующие), %": nanmean(neg[m]),
                    "СКО (несоответствующие), %": nanstd(neg[m]),
                    "Разность средних, %": nanmean(pos[m]) - nanmean(neg[m]),
                    "AUC": auc(pos[m], neg[m]),
                    "Чувствительность (принято соответствующих)": tpr,
                    "Специфичность (отклонено несоответствующих)": tnr,
                    "Сбалансированная доля верных решений": (tpr + tnr) / 2,
                    "Пар соответствующих": len(pos[m]), "Пар несоответствующих": len(neg[m]),
                })
        write_csv(out / "e2_pairs.csv", rows_pairs)
        write_csv(out / "e2_summary.csv", rows_sum)

    # --- e6: подбор весов ---
    if "e6" in only:
        rows = []
        grid = [w / 10 for w in range(11)]
        for v in variants:
            pos_vals = [vals for key, vals in cache.items() if key[0] == v and key[1] == "pos"]
            neg_vals = [vals for key, vals in cache.items() if key[0] == v and key[1] == "neg"]
            for w1, w2, w3 in product(grid, grid, grid):
                w4 = round(1.0 - w1 - w2 - w3, 10)
                if w4 < -1e-9:
                    continue
                w = (w1, w2, w3, max(w4, 0.0))
                a = auc([combine(x, w) for x in pos_vals], [combine(x, w) for x in neg_vals])
                rows.append({"Вариант": VARIANT_TITLES[v], "w1": w[0], "w2": w[1], "w3": w[2], "w4": w[3], "AUC": a})
        def dist_to_expert(r):
            return sum((r[k] - w) ** 2 for k, w in zip(("w1", "w2", "w3", "w4"), DEFAULT_WEIGHTS))
        # при равной AUC предпочтение отдаётся весам, ближайшим к экспертным
        rows.sort(key=lambda r: (r["Вариант"], -round(r["AUC"], 6), dist_to_expert(r)))
        write_csv(out / "e6_weights_all.csv", rows)
        best_rows = []
        for v in variants:
            vr = [r for r in rows if r["Вариант"] == VARIANT_TITLES[v]]
            best = vr[0]
            loo = loo_evaluation(ds, cache, v, (best["w1"], best["w2"], best["w3"], best["w4"]))[METHOD_COMBINED]
            pos_vals = [vals for key, vals in cache.items() if key[0] == v and key[1] == "pos"]
            neg_vals = [vals for key, vals in cache.items() if key[0] == v and key[1] == "neg"]
            exp_auc = auc([combine(x, DEFAULT_WEIGHTS) for x in pos_vals],
                          [combine(x, DEFAULT_WEIGHTS) for x in neg_vals])
            best_rows.append(dict(best, **{
                "AUC при экспертных весах": exp_auc,
                "Чувствительность": loo["pos_ok"] / max(loo["pos_n"], 1),
                "Специфичность": loo["neg_ok"] / max(loo["neg_n"], 1),
            }))
        write_csv(out / "e6_weights_best.csv", best_rows)

    # --- e7: время ---
    if "e7" in only:
        rows = [{"Метод": METHOD_TITLES[m], "Среднее время, мс": 1000 * nanmean(timing[m]),
                 "Сравнений": len(timing[m])} for m in COMPONENT_METHODS]
        write_csv(out / "e7_timing.csv", rows)

    rng = random.Random(args.seed)
    base_pairs = []
    for g in ds.segments:
        for p, q in combinations(ds.persons_for(g), 2):
            base_pairs.append(("pos", ds.segment(p, g), ds.segment(q, g)))
    for g, h in combinations(ds.segments, 2):
        if g["video"] == h["video"]:
            continue
        L = min(ds.seg_len(g), ds.seg_len(h))
        for p in ds.persons_for(h):
            for q in ds.persons_for(g):
                if p != q:
                    base_pairs.append(("neg", ds.segment(q, g, L), ds.segment(p, h, L)))

    # --- e3: шум ---
    if "e3" in only:
        levels = [float(x) for x in args.noise.split(",") if x]
        tasks = []
        for lvl in levels:
            for rep in range(args.reps):
                noisy = []
                for kind, a, b in base_pairs:
                    na = add_noise(a[0], a[1], lvl, rng)
                    nb = add_noise(b[0], b[1], lvl, rng)
                    noisy.append((kind, na, nb))
                for v in variants:
                    for k, (kind, na, nb) in enumerate(noisy):
                        tasks.append(((v, lvl, rep, kind, k), na, nb, vparams[v]))
        print(f"e3: сравнений {len(tasks)}")
        t0 = time.time()
        res = run_tasks(tasks, args.workers)
        print(f"  готово за {time.time() - t0:.1f} с")
        rows = []
        for v in variants:
            for lvl in [0.0] + levels:
                aucs = {m: [] for m in METHODS}
                pos_mean = {m: [] for m in METHODS}
                neg_mean = {m: [] for m in METHODS}
                if lvl == 0.0:
                    groups = [[(key[1], vals) for key, vals in cache.items()
                               if key[0] == v and key[1] in ("pos", "neg")]] if cache else []
                else:
                    groups = [[(key[3], vals) for key, vals, _t in res
                               if key[0] == v and key[1] == lvl and key[2] == rep] for rep in range(args.reps)]
                for grp in groups:
                    pos = {m: [] for m in METHODS}
                    neg = {m: [] for m in METHODS}
                    for kind, vals in grp:
                        full = with_combined(vals)
                        for m in METHODS:
                            (pos if kind == "pos" else neg)[m].append(full[m])
                    for m in METHODS:
                        aucs[m].append(auc(pos[m], neg[m]))
                        pos_mean[m].append(nanmean(pos[m]))
                        neg_mean[m].append(nanmean(neg[m]))
                for m in METHODS:
                    rows.append({"Вариант": VARIANT_TITLES[v], "Метод": METHOD_TITLES[m],
                                 "Уровень шума": lvl, "AUC": nanmean(aucs[m]),
                                 "Среднее (соответствующие), %": nanmean(pos_mean[m]),
                                 "Среднее (несоответствующие), %": nanmean(neg_mean[m])})
        write_csv(out / "e3_noise.csv", rows)

    # --- e4: безэмоциональные фрагменты ---
    if "e4" in only:
        fracs = [float(x) for x in args.neutral.split(",") if x]
        tasks = []
        for fr in fracs:
            for v in variants:
                for k, (kind, a, b) in enumerate(base_pairs):
                    cnt = int(round(fr * len(a[0])))
                    tasks.append(((v, fr, kind, k), prepend_neutral(a[0], a[1], cnt),
                                  prepend_neutral(b[0], b[1], cnt), vparams[v]))
        print(f"e4: сравнений {len(tasks)}")
        t0 = time.time()
        res = run_tasks(tasks, args.workers)
        print(f"  готово за {time.time() - t0:.1f} с")
        rows = []
        for v in variants:
            clean = {}
            for key, vals in cache.items():
                if key[0] == v and key[1] in ("pos", "neg"):
                    clean.setdefault(key[1], []).append(with_combined(vals))
            for fr in [0.0] + fracs:
                if fr == 0.0:
                    pos = clean.get("pos", [])
                    neg = clean.get("neg", [])
                else:
                    pos = [with_combined(vals) for key, vals, _t in res if key[0] == v and key[1] == fr and key[2] == "pos"]
                    neg = [with_combined(vals) for key, vals, _t in res if key[0] == v and key[1] == fr and key[2] == "neg"]
                for m in METHODS:
                    rows.append({"Вариант": VARIANT_TITLES[v], "Метод": METHOD_TITLES[m],
                                 "Доля нейтрального участка": fr,
                                 "AUC": auc([x[m] for x in pos], [x[m] for x in neg]),
                                 "Среднее (соответствующие), %": nanmean([x[m] for x in pos]),
                                 "Среднее (несоответствующие), %": nanmean([x[m] for x in neg])})
        write_csv(out / "e4_neutral.csv", rows)

    # --- e5: запаздывание и темп ---
    if "e5" in only:
        tasks = []
        items = [(p, g) for g in ds.segments for p in ds.persons_for(g)]
        for v in variants:
            for p, g in items:
                labels, probs = ds.segment(p, g)
                fps = ds.maps[(p, g["video"])]["fps"]
                for sec in (0.5, 1.0, 2.0, 3.0):
                    d = delay(labels, probs, int(round(sec * fps)))
                    tasks.append(((v, "Запаздывание, с", sec, p, g["name"]), (labels, probs), d, vparams[v]))
                for f in (0.8, 0.9, 1.1, 1.2):
                    s = stretch(labels, probs, f)
                    tasks.append(((v, "Коэффициент темпа", f, p, g["name"]), (labels, probs), s, vparams[v]))
        print(f"e5: сравнений {len(tasks)}")
        t0 = time.time()
        res = run_tasks(tasks, args.workers)
        print(f"  готово за {time.time() - t0:.1f} с")
        agg = {}
        for key, vals, _t in res:
            full = with_combined(vals)
            for m in METHODS:
                agg.setdefault((key[0], key[1], key[2], m), []).append(full[m])
        rows = [{"Вариант": VARIANT_TITLES[v], "Искажение": kind, "Величина": val,
                 "Метод": METHOD_TITLES[m], "Сходство с исходной картой, %": nanmean(xs)}
                for (v, kind, val, m), xs in sorted(agg.items(), key=lambda t: (t[0][0], t[0][1], t[0][2], t[0][3]))]
        write_csv(out / "e5_distortion.csv", rows)

    print(f"Результаты сохранены в {out.resolve()}")


if __name__ == "__main__":
    main()
