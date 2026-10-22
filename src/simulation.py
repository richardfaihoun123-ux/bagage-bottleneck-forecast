"""Simulation à événements discrets (SimPy) d'une journée de tri : scénarios « et si ? ».

Modèle : chaque convoyeur a un poste d'injection (un serveur, file FIFO) puis un temps de transport.
Les paramètres sont calibrés sur les données (capacité = charge horaire maximale observée x marge ;
transport = temps médian hors heures chargées). Les bagages de la journée choisie sont rejoués tels quels.
"""
import numpy as np
import pandas as pd
import simpy
import matplotlib.pyplot as plt

BLUE, GREEN, RED, ORANGE, GREY = "#1F4E79", "#2E8B57", "#C0392B", "#F2A900", "#7F8C8D"
SERVICE_SIGMA = 0.5       # dispersion (log-normale) du temps d'injection
HEADROOM = 1.15           # capacité = 1,15 x charge horaire maximale observée sur le convoyeur


def calibrate(bags, conv):
    slots = bags.assign(slot=bags.checkin_ts.dt.floor("h"))
    cap = slots.groupby(["conveyor_id", "slot"]).size().groupby("conveyor_id").max() * HEADROOM   # bagages/heure
    mean_service = 60.0 / cap                                                                      # minutes/bagage

    # Transport : temps médian en heures peu chargées (réseau < 25e centile de charge, heures 5-21)
    net = slots.groupby("slot").size()
    net = net[(net.index.hour >= 5) & (net.index.hour <= 21)]
    quiet = set(net[net <= net.quantile(0.25)].index)
    q = slots[slots.slot.isin(quiet)]
    med = q.groupby("conveyor_id").processing_min.median()
    sig = np.log(q.processing_min).groupby(q.conveyor_id).std()
    params = pd.DataFrame({"capacity_per_h": cap, "mean_service": mean_service,
                           "transport_median": (med - mean_service).clip(lower=3.0), "sigma": sig})
    return params.join(conv.set_index("conveyor_id"))


def simulate_day(day, params, rng, volume_factor=1.0, breakdown=None, reroute=False, transport_scale=1.0):
    """day : DataFrame des bagages du jour (conveyor_id, terminal, arrival, cutoff).
    breakdown : (conveyor_id, début en minutes, durée en minutes) ou None."""
    d = day.copy()
    if volume_factor > 1.0:                               # volume supplémentaire : bagages tirés au hasard et dupliqués
        extra = d.sample(frac=volume_factor - 1.0, replace=True, random_state=int(rng.integers(1e9))).copy()
        extra["arrival"] = np.clip(extra.arrival + rng.normal(0, 5, len(extra)), 0, None)
        d = pd.concat([d, extra], ignore_index=True)
    d = d.sort_values("arrival").reset_index(drop=True)

    env = simpy.Environment()
    stations = {c: simpy.PriorityResource(env, capacity=1) for c in params.index}
    term_of = params.terminal.to_dict()
    by_term = {t: [c for c in params.index if term_of[c] == t] for t in set(term_of.values())}
    loaded = np.zeros(len(d))
    max_queue = [0]

    def breakdown_proc(c, start, dur):
        yield env.timeout(start)
        with stations[c].request(priority=0) as req:       # priorité absolue : passe devant les bagages en attente
            yield req
            yield env.timeout(dur)

    def bag(i, c, t_arr):
        yield env.timeout(t_arr)
        if reroute and breakdown and c == breakdown[0] and breakdown[1] <= env.now < breakdown[1] + breakdown[2]:
            options = [x for x in by_term[term_of[c]] if x != c]
            c = min(options, key=lambda x: len(stations[x].queue) + stations[x].count)
        st, p = stations[c], params.loc[c]
        with st.request(priority=1) as req:
            yield req
            max_queue[0] = max(max_queue[0], len(st.queue))
            svc = rng.lognormal(np.log(p.mean_service) - SERVICE_SIGMA**2 / 2, SERVICE_SIGMA)
            yield env.timeout(svc)
        yield env.timeout(rng.lognormal(np.log(p.transport_median * transport_scale), p.sigma))
        loaded[i] = env.now

    if breakdown:
        env.process(breakdown_proc(*breakdown))
    for i, r in enumerate(d.itertuples()):
        env.process(bag(i, r.conveyor_id, r.arrival))
    env.run()

    proc = loaded - d.arrival.values
    return {"bags": len(d), "late_rate": float(np.mean(loaded > d.cutoff.values)),
            "mean_proc": float(proc.mean()), "p95_proc": float(np.quantile(proc, 0.95)),
            "max_queue": int(max_queue[0])}


def run(bags, conv, fig_dir, day=None, n_rep=20, seed=7):
    params = calibrate(bags, conv)
    if day is None:
        day = bags.groupby("date").size().idxmax()           # jour de pic
    day = pd.Timestamp(day)
    d = bags[bags.date == day]
    base = pd.DataFrame({
        "conveyor_id": d.conveyor_id.values, "terminal": d.terminal.values,
        "arrival": (d.checkin_ts - day).dt.total_seconds().values / 60,
        "cutoff": (d.cutoff - day).dt.total_seconds().values / 60})
    real = {"bags": int(len(d)), "late_rate": float((d.loaded_ts > d.cutoff).mean()),
            "mean_proc": float(d.processing_min.mean()), "p95_proc": float(d.processing_min.quantile(0.95))}

    # Calibrage sur le jour réel : les jours chargés ralentissent tout le réseau (facteur multiplicatif sur le transport)
    probe = [simulate_day(base, params, np.random.default_rng(1000 + k)) for k in range(6)]
    k_scale = real["mean_proc"] / float(np.mean([r["mean_proc"] for r in probe]))
    for _ in range(2):                                    # deux itérations pour affiner
        probe = [simulate_day(base, params, np.random.default_rng(2000 + k), transport_scale=k_scale) for k in range(6)]
        k_scale *= real["mean_proc"] / float(np.mean([r["mean_proc"] for r in probe]))

    # Panne : le convoyeur « problématique » n°2 (C02 par défaut) à son heure la plus chargée du jour
    target = "C02"
    hourly = d[d.conveyor_id == target].groupby(d.checkin_ts.dt.hour).size()
    start_min = float(hourly.idxmax() * 60)
    outage = (target, start_min, 60.0)

    scenarios = [
        ("Référence", dict()),
        ("Volume +20 %", dict(volume_factor=1.2)),
        (f"Panne {target} 1 h\n(file d'attente)", dict(breakdown=outage)),
        (f"Panne {target} 1 h\n+ reroutage", dict(breakdown=outage, reroute=True)),
        (f"Volume +20 %\n+ panne + reroutage", dict(volume_factor=1.2, breakdown=outage, reroute=True)),
    ]
    results = []
    for name, kw in scenarios:
        reps = [simulate_day(base, params, np.random.default_rng(seed + k), transport_scale=k_scale, **kw) for k in range(n_rep)]
        df = pd.DataFrame(reps)
        results.append({"scenario": name.replace("\n", " "), "bags": float(df.bags.mean()),
                        "late_rate_%": float(df.late_rate.mean() * 100), "late_rate_sd": float(df.late_rate.std() * 100),
                        "mean_proc": float(df.mean_proc.mean()), "p95_proc": float(df.p95_proc.mean()),
                        "p95_sd": float(df.p95_proc.std()), "max_queue": float(df.max_queue.mean())})
    res = pd.DataFrame(results)

    out = {"day": str(day.date()), "replications": n_rep, "outage": {"conveyor": target, "start_hour": start_min / 60, "duration_min": 60},
           "real_day": {k: round(v, 3) for k, v in real.items()}, "transport_scale": round(float(k_scale), 3),
           "scenarios": res.round(2).to_dict(orient="records"),
           "calibration": params.round(2).reset_index().to_dict(orient="records")}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = [n for n, _ in scenarios]
    cols = [GREY, ORANGE, RED, GREEN, BLUE]
    axes[0].bar(range(len(res)), res["late_rate_%"], yerr=res["late_rate_sd"], color=cols, capsize=3)
    axes[0].axhline(real["late_rate"] * 100, color="black", ls=":", lw=1, label="Réel observé")
    axes[0].set_xticks(range(len(res)), labels, fontsize=7); axes[0].set_ylabel("Bagages chargés trop tard (%)")
    axes[0].set_title("Retards"); axes[0].legend(fontsize=8)
    axes[1].bar(range(len(res)), res.p95_proc, yerr=res.p95_sd, color=cols, capsize=3)
    axes[1].axhline(real["p95_proc"], color="black", ls=":", lw=1, label="Réel observé")
    axes[1].set_xticks(range(len(res)), labels, fontsize=7); axes[1].set_ylabel("P95 du temps de traitement (min)")
    axes[1].set_title("Temps de traitement (P95)"); axes[1].legend(fontsize=8)
    for i, (a, b) in enumerate(zip(res["late_rate_%"], res.p95_proc)):
        axes[0].text(i, a / 2, f"{a:.2f} %", ha="center", color="white", fontsize=9, fontweight="bold")
        axes[1].text(i, b / 2, f"{b:.1f}", ha="center", color="white", fontsize=9, fontweight="bold")
    fig.suptitle(f"Simulation du {day:%d/%m/%Y} (jour de pic), moyenne de {n_rep} répétitions", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / "07_simulation_scenarios.png", dpi=150); plt.close(fig)
    return out
