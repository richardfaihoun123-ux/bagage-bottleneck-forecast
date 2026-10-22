"""Lance toute l'analyse : goulots -> prévision -> simulation -> anomalies.

Usage :
    python src/run_analysis.py --data data/csv
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import anomalies
import bottlenecks
import data
import forecast
import simulation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/csv", help="dossier des CSV générés par le projet 1")
    ap.add_argument("--out", default="results", help="dossier de sortie")
    ap.add_argument("--reps", type=int, default=20, help="répétitions par scénario de simulation")
    args = ap.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    bags, flights, conv, cal, inc = data.load(args.data)
    print(f"[1/4] Données chargées : {len(bags):,} bagages")

    summary = {}
    summary["bottlenecks"], _ = bottlenecks.run(bags, conv, fig_dir)
    print("[1/4] Goulots : OK")
    summary["forecast"] = forecast.run(bags, cal, fig_dir, flights=flights)
    print("[2/4] Prévision : OK")
    summary["simulation"] = simulation.run(bags, conv, fig_dir, n_rep=args.reps)
    print("[3/4] Simulation : OK")
    summary["anomalies"] = anomalies.run(bags, inc, fig_dir)
    print("[4/4] Anomalies : OK")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    b, f, s, a = summary["bottlenecks"], summary["forecast"], summary["simulation"], summary["anomalies"]
    print("\n=== Résultats clés ===")
    print(f"Dépôts tardifs : {b['lead_time']['share_late_bags_lead_lt60']:.1%} des retards viennent de bagages déposés < 60 min avant le départ "
          f"({b['lead_time']['share_bags_lead_lt60']:.1%} des bagages)")
    print(f"Convoyeurs C01+C02 : {b['top2_share_of_issues']:.1%} des anomalies pour {b['top2_share_of_bags']:.1%} des bagages")
    print(f"Prévision J+7 : MAE {f['daily_metrics']['Gradient boosting']['MAE']:.0f} bagages/jour "
          f"(naïf : {f['daily_metrics']['Naïf (J-7)']['MAE']:.0f})")
    for sc in s["scenarios"]:
        print(f"Simulation - {sc['scenario']:<40} retards {sc['late_rate_%']:.2f} %  P95 {sc['p95_proc']:.1f} min")
    z = a["robust_zscore"]
    print(f"Anomalies (z >= {a['z_threshold']:g}) : précision {z['precision']:.0%}, "
          f"{z['incidents_detected']}/{z['incidents_in_evaluable_windows']} pannes détectables retrouvées")
    print(f"\nTerminé en {time.time() - t0:.0f} s. Figures : {fig_dir}   Détails : {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
