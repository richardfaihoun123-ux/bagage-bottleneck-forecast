"""Analyse des goulots d'étranglement : quels convoyeurs, quelles heures, quels facteurs ?"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, GREY = "#1F4E79", "#2E8B57", "#C0392B", "#F2A900", "#7F8C8D"


def wilson(k, n, z=1.96):
    """Intervalle de confiance de Wilson pour une proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def conveyor_table(bags, conv):
    g = bags.groupby("conveyor_id").agg(
        bags=("bag_id", "size"),
        issues=("issue", "sum"),
        late=("status", lambda s: (s == "late").sum()),
        mishandled=("status", lambda s: (s == "mishandled").sum()),
        avg_proc=("processing_min", "mean"),
        p95_proc=("processing_min", lambda s: s.quantile(0.95)),
    ).reset_index().merge(conv, on="conveyor_id")
    g["issue_rate"] = g.issues / g.bags
    ci = g.apply(lambda r: wilson(r.issues, r.bags), axis=1)
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    g["share_of_issues"] = g.issues / g.issues.sum()
    g["share_of_bags"] = g.bags / g.bags.sum()
    return g.sort_values("issues", ascending=False).reset_index(drop=True)


def lead_time_table(bags):
    """Délai entre le dépôt et le départ du vol : facteur clé des retards."""
    lead = (bags.scheduled_dep - bags.checkin_ts).dt.total_seconds() / 60
    bins = [0, 45, 60, 90, 120, 180, 10_000]
    labels = ["≤ 45 min", "45-60", "60-90", "90-120", "120-180", "> 180"]
    t = (bags.assign(lead=lead, lead_bin=pd.cut(lead, bins=bins, labels=labels))
         .groupby("lead_bin", observed=True)
         .agg(bags=("bag_id", "size"), late=("status", lambda s: (s == "late").sum()),
              mishandled=("status", lambda s: (s == "mishandled").sum()))
         .reset_index())
    t["late_rate"] = t.late / t.bags
    t["mishandled_rate"] = t.mishandled / t.bags
    late_mask = bags.status == "late"
    stats = {
        "share_bags_lead_lt60": float((lead < 60).mean()),
        "share_late_bags_lead_lt60": float((lead[late_mask] < 60).mean()),
        "late_rate_lead_le45": float(t.loc[t.lead_bin == "≤ 45 min", "late_rate"].iloc[0]),
        "late_rate_lead_60_90": float(t.loc[t.lead_bin == "60-90", "late_rate"].iloc[0]),
    }
    return t, stats


def network_load_table(bags):
    """Charge du réseau (bagages déposés par heure) et temps de traitement."""
    h = (bags.assign(slot=bags.checkin_ts.dt.floor("h"))
         .groupby("slot").agg(bags=("bag_id", "size"), avg_proc=("processing_min", "mean"),
                              p95_proc=("processing_min", lambda s: s.quantile(0.95))).reset_index())
    h = h[(h.slot.dt.hour >= 5) & (h.slot.dt.hour <= 21)]   # heures d'exploitation avec volume significatif
    cap = h.bags.quantile(0.95)
    h["load_ratio"] = h.bags / cap
    labels = ["< 25 %", "25-50 %", "50-75 %", "75-100 %", "> 100 %"]
    h["load_bin"] = pd.cut(h.load_ratio, bins=[0, .25, .5, .75, 1.0, 10], labels=labels)
    t = h.groupby("load_bin", observed=True).agg(hours=("bags", "size"), avg_proc=("avg_proc", "mean"),
                                                  p95_proc=("p95_proc", "mean")).reset_index()
    return t, float(cap)


def hour_profile(bags):
    g = bags.groupby("hour").agg(bags=("bag_id", "size"), issues=("issue", "sum")).reset_index()
    g["issue_rate"] = g.issues / g.bags
    return g


def hour_conveyor_heatmap(bags):
    pv = bags.groupby(["conveyor_id", "hour"]).agg(n=("bag_id", "size"), k=("issue", "sum")).reset_index()
    pv["rate"] = pv.k / pv.n
    pv.loc[pv.n < 200, "rate"] = np.nan   # trop peu de bagages pour conclure
    return pv.pivot(index="conveyor_id", columns="hour", values="rate")


def run(bags, conv, fig_dir):
    out = {}
    ct = conveyor_table(bags, conv)
    lt, lead_stats = lead_time_table(bags)
    nl, cap = network_load_table(bags)
    hp = hour_profile(bags)
    hm = hour_conveyor_heatmap(bags)

    out["network_issue_rate"] = round(float(bags.issue.mean()), 4)
    out["conveyors"] = ct.round(4).to_dict(orient="records")
    out["corr_length_issue_rate"] = round(float(np.corrcoef(ct.length_m, ct.issue_rate)[0, 1]), 3)
    out["top2_share_of_issues"] = round(float(ct.share_of_issues.head(2).sum()), 3)
    out["top2_share_of_bags"] = round(float(ct.share_of_bags.head(2).sum()), 3)
    out["lead_time"] = {k: round(v, 4) for k, v in lead_stats.items()}
    out["lead_time_table"] = lt.round(4).to_dict(orient="records")
    out["network_load"] = nl.round(3).to_dict(orient="records")
    out["network_capacity_bags_per_hour"] = round(cap, 1)
    out["peak_issue_hours"] = hp[hp.bags > 1000].nlargest(3, "issue_rate").hour.tolist()

    # Figure 1 : Pareto avec intervalles de confiance
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(ct))
    ax.bar(x, ct.issues, color=BLUE)
    ax.set_xticks(x, ct.conveyor_id)
    ax.set_ylabel("Bagages en retard ou mal acheminés")
    ax.set_title("Pareto des anomalies par convoyeur (12 mois)")
    ax2 = ax.twinx()
    err = np.array([ct.issue_rate - ct.ci_low, ct.ci_high - ct.issue_rate]) * 100
    ax2.errorbar(x, ct.issue_rate * 100, yerr=err, color=GREEN, marker="o", ls="none", capsize=3, label="Taux d'anomalie (IC 95 %)")
    ax2.axhline(out["network_issue_rate"] * 100, color=GREY, ls="--", lw=1, label="Moyenne réseau")
    ax2.set_ylabel("Taux d'anomalie (%)"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(fig_dir / "01_pareto_convoyeurs.png", dpi=150); plt.close(fig)

    # Figure 2 : délai de dépôt avant départ -> retards
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(lt.lead_bin.astype(str), lt.late_rate * 100, color=RED)
    for i, (r, n) in enumerate(zip(lt.late_rate, lt.bags)):
        ax.text(i, r * 100 + 0.5, f"{n:,} bag.".replace(",", " "), ha="center", fontsize=8)
    ax.set_xlabel("Délai entre le dépôt du bagage et le départ du vol")
    ax.set_ylabel("Taux de bagages en retard (%)")
    ax.set_title("Les retards viennent des dépôts tardifs")
    fig.tight_layout(); fig.savefig(fig_dir / "02_retards_vs_delai_depot.png", dpi=150); plt.close(fig)

    # Figure 3 : carte thermique convoyeur x heure
    fig, ax = plt.subplots(figsize=(10, 4.8))
    hm = hm.reindex(ct.conveyor_id.tolist())
    im = ax.imshow(hm.values * 100, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=np.nanpercentile(hm.values * 100, 95))
    ax.set_xticks(range(hm.shape[1]), hm.columns); ax.set_yticks(range(hm.shape[0]), hm.index)
    ax.set_xlabel("Heure de dépôt")
    ax.set_title("Taux d'anomalie (%) par convoyeur et heure (cases avec < 200 bagages masquées)")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout(); fig.savefig(fig_dir / "03_heatmap_convoyeur_heure.png", dpi=150); plt.close(fig)

    # Figure 4 : charge réseau -> temps de traitement
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(nl.load_bin.astype(str), nl.avg_proc, marker="o", color=BLUE, label="Temps moyen")
    ax.plot(nl.load_bin.astype(str), nl.p95_proc, marker="s", color=ORANGE, label="P95 moyen par heure")
    ax.set_xlabel("Charge du réseau (bagages déposés par heure / P95 des heures)")
    ax.set_ylabel("Temps de traitement (min)")
    ax.set_title("Effet de la charge sur le temps de traitement"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "04_charge_vs_temps.png", dpi=150); plt.close(fig)
    return out, ct
