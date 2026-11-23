import json, re, random
import numpy as np

def parse_and_eval(raw_str, gt):
    cleaned = re.sub(r"```json|```", "", raw_str).strip()
    try:
        obj = json.loads(cleaned)
        if not isinstance(obj, dict) or "subject" not in obj or "object" not in obj:
            return 0, 1, 0.0
    except Exception:
        return 0, 1, 0.0

    vals = f"{obj['subject']} {obj['object']}".strip()
    has_thai = bool(re.search(r"[\u0E00-\u0E7F]", vals))
    has_leakage = bool(re.search(r"[\uAC00-\uD7A3a-zA-Z]", vals))
    lang_violation = 1 if (not has_thai or has_leakage) else 0

    score = 0.0
    for k in ["subject", "object"]:
        pred_v, gt_v = str(obj[k]).strip(), str(gt[k]).strip()
        if pred_v and gt_v and (pred_v == gt_v or pred_v in gt_v or gt_v in pred_v):
            score += 0.5
    return 1, lang_violation, score

def bootstrap_ci(arr_a, arr_b, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.array(arr_a) - np.array(arr_b)
    boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    return np.mean(diffs) * 100, np.percentile(boot_means, 2.5) * 100, np.percentile(boot_means, 97.5) * 100

def main():
    with open("experiment_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    conditions = ["en_direct", "ko_direct", "th_direct", "en_pivot", "ko_control_step"]
    metrics = {c: {"fcr": [], "lvr": [], "acc": []} for c in conditions}

    for row in results:
        for c in conditions:
            fcr, lvr, acc = parse_and_eval(row[c], row["gt_th"])
            metrics[c]["fcr"].append(fcr)
            metrics[c]["lvr"].append(lvr)
            metrics[c]["acc"].append(acc)

    print("=== Table 1: Main Results (%) ===")
    for c in conditions:
        fcr_m = np.mean(metrics[c]["fcr"]) * 100
        lvr_m = np.mean(metrics[c]["lvr"]) * 100
        acc_all = np.mean(metrics[c]["acc"]) * 100
        valid_accs = [a for f, a in zip(metrics[c]["fcr"], metrics[c]["acc"]) if f == 1]
        acc_cond = (np.mean(valid_accs) * 100) if valid_accs else 0.0
        print(f"{c:<16} | FCR: {fcr_m:5.1f}% | Lang_Viol: {lvr_m:5.1f}% | Acc(All): {acc_all:5.1f}% | Acc(Cond): {acc_cond:5.1f}%")

    print("\n=== Table 2: Paired Differences & 95% Bootstrap CI (%) ===")
    tests = [
        ("H1: LVR (KO_prompt - EN_prompt)", metrics["ko_direct"]["lvr"], metrics["en_direct"]["lvr"]),
        ("H1: Acc (KO_prompt - EN_prompt)", metrics["ko_direct"]["acc"], metrics["en_direct"]["acc"]),
        ("H2: Acc (Pivot_EN - Direct_EN)", metrics["en_pivot"]["acc"], metrics["en_direct"]["acc"]),
        ("Ablation: Acc (Pivot_EN - Ctrl_KO)", metrics["en_pivot"]["acc"], metrics["ko_control_step"]["acc"]),
    ]
    for name, a, b in tests:
        mean_d, low, high = bootstrap_ci(a, b, n_boot=1000, seed=42)
        print(f"{name:<35} | Diff: {mean_d:+6.2f}% | 95% CI: [{low:+6.2f}%, {high:+6.2f}%]")

    random.seed(42)
    failures = [
        r for r in results
        if parse_and_eval(r["ko_direct"], r["gt_th"])[1] == 1 or parse_and_eval(r["en_pivot"], r["gt_th"])[2] < 1.0
    ]
    print(f"\n=== Sampled Failure Cases (Total failures: {len(failures)}/100) ===")
    for sample in random.sample(failures, min(3, len(failures))):
        print(json.dumps({
            "id": sample["id"],
            "gt_th": sample["gt_th"],
            "ko_direct": sample["ko_direct"],
            "en_pivot": sample["en_pivot"],
        }, ensure_ascii=False))

if __name__ == "__main__":
    main()
