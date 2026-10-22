"""Détection d'anomalies sur le temps de traitement, validée contre les pannes réellement injectées."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest

BLUE, GREEN, RED, ORANGE, GREY = "#1F4E79", "#2E8B57", "#C0392B", "#F2A900", "#7F8C8D"
WINDOW = "30min"
MIN_BAGS = 3        # une fenêtre avec moins de 3 bagages n'est pas évaluée
Z_THRESHOLD = 4.0


def window_table(bags):
    b = bags.assign(logp=np.log(bags.processing_min), win=bags.checkin_ts.dt.floor(WINDOW), h=bags.checkin_ts.dt.hour)
    # Référence robuste par convoyeur x heure : médiane et écart robuste (MAD) du log du temps de traitement
    ref = b.groupby(["conveyor_id", "h"]).logp.agg(
        mu="median", sigma=lambda s: 1.4826 * np.median(np.abs(s - np.median(s)))).reset_index()
    ref["sigma"] = ref.sigma.clip(lower=0.1)
    w = (b.groupby(["conveyor_id", "win"])
         .agg(n=("bag_id", "size"), mean_log=("logp", "mean"), mean_proc=("processing_min", "mean"),
              p90_proc=("processing_min", lambda s: s.quantile(0.9)), issue_rate=("issue", "mean"))
         .reset_index())
    w["h"] = w.win.dt.hour
    w = w.merge(ref, on=["conveyor_id", "h"], how="left")
    w = w[w.n >= MIN_BAGS].copy()
    w["z"] = (w.mean_log - w.mu) / (w.sigma / np.sqrt(w.n))
    return w


def label_windows(w, inc):
    """Une fenêtre est « vraie anomalie » si elle chevauche une panne du même convoyeur."""
    w = w.copy()
    w["incident"] = 0
    w["win_end"] = w.win + pd.Timedelta(WINDOW)
    inc = inc.assign(end=inc.start_ts + pd.to_timedelta(inc.duration_min, unit="m"))
    for r in inc.itertuples():
        m = (w.conveyor_id == r.conveyor_id) & (w.win < r.end) & (w.win_end > r.start_ts)
        w.loc[m, "incident"] = r.incident_id
    return w, inc


def evaluate(w, flag, inc):
    pos = w.incident > 0
    tp = int((flag & pos).sum())
    detected = set(w.loc[flag & pos, "incident"])
    return {"flagged_windows": int(flag.sum()), "true_positive_windows": tp,
            "precision": tp / max(int(flag.sum()), 1), "window_recall": tp / max(int(pos.sum()), 1),
            "incidents_in_evaluable_windows": int(w.loc[pos, "incident"].nunique()),
            "incidents_detected": len(detected)}


def run(bags, inc, fig_dir):
    w = window_table(bags)
    w, inc = label_windows(w, inc)
    out = {"windows_evaluated": int(len(w)), "windows_overlapping_incident": int((w.incident > 0).sum()),
           "z_threshold": Z_THRESHOLD, "window": WINDOW, "min_bags": MIN_BAGS}

    flag_z = (w.z >= Z_THRESHOLD)
    out["robust_zscore"] = {k: round(v, 3) for k, v in evaluate(w, flag_z, inc).items()}

    # Balayage du seuil : le seuil est un choix métier (budget d'alertes), on montre le compromis
    sweep = []
    for z in (3, 4, 5, 6, 8):
        e = evaluate(w, w.z >= z, inc)
        sweep.append({"z": z, "alerts": e["flagged_windows"], "precision": round(e["precision"], 3),
                      "window_recall": round(e["window_recall"], 3), "incidents_detected": e["incidents_detected"]})
    out["threshold_sweep"] = sweep

    # Isolation Forest avec le même budget d'alertes (même nombre de fenêtres signalées)
    feats = w[["mean_proc", "p90_proc", "issue_rate", "n", "h"]]
    contamination = float(np.clip(flag_z.mean(), 0.001, 0.5))
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=0).fit(feats)
    flag_if = pd.Series(iso.predict(feats) == -1, index=w.index)
    out["isolation_forest"] = {k: round(v, 3) for k, v in evaluate(w, flag_if, inc).items()}
    out["base_rate_incident_windows"] = round(float((w.incident > 0).mean()), 4)

    # Détection par incident (au moins une fenêtre signalée pendant la panne)
    inc_eval = inc[inc.incident_id.isin(w.loc[w.incident > 0, "incident"].unique())]
    out["incidents_total"] = int(len(inc))

    # --- Figures ---
    ex = (w[w.incident > 0].groupby("incident").agg(n=("n", "sum")).n.idxmax())
    r = inc[inc.incident_id == ex].iloc[0]
    sub = w[(w.conveyor_id == r.conveyor_id) & (w.win >= r.start_ts - pd.Timedelta(hours=6)) &
            (w.win <= r.end + pd.Timedelta(hours=6))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.axvspan(r.start_ts, r.end, color=RED, alpha=0.15, label=f"Panne réelle ({int(r.duration_min)} min)")
    ax.plot(sub.win, sub.mean_proc, color=BLUE, marker="o", ms=3, label="Temps moyen par fenêtre de 30 min")
    fl = sub[sub.z >= Z_THRESHOLD]
    ax.scatter(fl.win, fl.mean_proc, color=RED, zorder=3, s=40, label=f"Alerte (z ≥ {Z_THRESHOLD:g})")
    ax.set_ylabel("Temps de traitement moyen (min)"); ax.set_title(f"Détection sur {r.conveyor_id}")
    ax.legend(fontsize=8); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh%M")); ax.tick_params(axis="x", rotation=30)
    ax = axes[1]
    names = ["z-score robuste", "Isolation Forest"]
    prec = [out["robust_zscore"]["precision"], out["isolation_forest"]["precision"]]
    rec = [out["robust_zscore"]["window_recall"], out["isolation_forest"]["window_recall"]]
    x = np.arange(2)
    ax.bar(x - 0.2, np.array(prec) * 100, 0.4, color=GREEN, label="Précision")
    ax.bar(x + 0.2, np.array(rec) * 100, 0.4, color=ORANGE, label="Rappel")
    ax.set_xticks(x, names); ax.set_ylabel("%"); ax.set_title("Qualité de détection"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "08_detection_anomalies.png", dpi=150); plt.close(fig)
    return out
