"""Prévision des volumes à J+7 (quotidien) puis déclinaison horaire pour dimensionner les équipes."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

BLUE, GREEN, RED, ORANGE, GREY = "#1F4E79", "#2E8B57", "#C0392B", "#F2A900", "#7F8C8D"
TEST_DAYS = 60           # derniers jours réservés à l'évaluation (pas de fuite temporelle)
HORIZON = 7              # on prévoit J+7 : seules les données jusqu'à J-7 sont utilisées


def daily_frame(bags, cal):
    d = bags.groupby("date").size().rename("bags").reset_index()
    d = d[(d.date >= cal.full_date.min()) & (d.date <= cal.full_date.max())].iloc[:-1]  # dernier jour incomplet
    d = d.merge(cal, left_on="date", right_on="full_date")
    d["is_summer"] = (d.season == "ete").astype(int)
    s = d.set_index("date").bags
    d["lag_7"] = s.shift(HORIZON).values
    d["lag_14"] = s.shift(HORIZON + 7).values
    d["roll_28"] = s.shift(HORIZON).rolling(28).mean().values   # moyenne des 28 jours disponibles à J-7
    return d.dropna(subset=["lag_7", "lag_14", "roll_28"]).reset_index(drop=True)


FEATURES = ["day_of_week", "is_school_holiday", "is_summer", "month", "lag_7", "lag_14", "roll_28"]


def design_ridge(df):
    X = pd.get_dummies(df.day_of_week, prefix="dow").astype(float)
    X["holiday"] = df.is_school_holiday.values
    X["summer"] = df.is_summer.values
    X["roll_28"] = df.roll_28.values
    return X


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": float(np.mean(np.abs(y - p))), "MAPE_%": float(np.mean(np.abs(y - p) / y) * 100),
            "WAPE_%": float(np.sum(np.abs(y - p)) / np.sum(y) * 100)}


def peak_detection(y, p, threshold):
    act, pred = y >= threshold, p >= threshold
    tp = int((act & pred).sum())
    return {"threshold": float(threshold), "actual_peak_days": int(act.sum()),
            "recall": tp / max(int(act.sum()), 1), "precision": tp / max(int(pred.sum()), 1)}


def hourly_matrix(bags):
    m = bags.groupby(["date", "hour"]).size().unstack(fill_value=0)
    return m.reindex(columns=range(24), fill_value=0)


def schedule_hourly(flights, bags_train, flights_train, days):
    """Prévision horaire à partir du programme de vols (connu à l'avance).

    Chaque vol programmé apporte en moyenne N bagages (moyenne observée à l'entraînement) répartis selon la distribution
    empirique des délais de dépôt avant départ (observée à l'entraînement)."""
    lead = (bags_train.scheduled_dep - bags_train.checkin_ts).dt.total_seconds().values / 60
    lead_q = np.quantile(lead, np.linspace(0.0005, 0.9995, 1000))          # 1000 quantiles du délai de dépôt
    bags_per_flight = len(bags_train) / flights_train.flight_id.nunique()
    f = flights[flights.scheduled_dep.dt.normalize().between(days.min() - pd.Timedelta(days=1), days.max() + pd.Timedelta(days=1))]
    dep_min = (f.scheduled_dep.values.astype('datetime64[m]').astype('int64'))   # minutes depuis 1970
    t0 = days.min().to_datetime64().astype('datetime64[m]').astype('int64')
    n_hours = int((days.max() - days.min()).days + 1) * 24
    out = np.zeros(n_hours)
    for chunk in np.array_split(dep_min, max(1, len(dep_min) // 200)):
        slots = ((chunk[:, None] - lead_q[None, :]).astype('int64') - t0) // 60
        slots = slots[(slots >= 0) & (slots < n_hours)]
        out += np.bincount(slots, minlength=n_hours)[:n_hours] * bags_per_flight / len(lead_q)
    idx = pd.date_range(days.min(), periods=n_hours, freq='h')
    m = pd.Series(out, index=idx)
    mat = m.groupby([m.index.normalize(), m.index.hour]).sum().unstack()
    return mat.reindex(index=days, columns=range(24), fill_value=0)


def run(bags, cal, fig_dir, flights=None):
    out = {}
    d = daily_frame(bags, cal)
    train, test = d.iloc[:-TEST_DAYS].copy(), d.iloc[-TEST_DAYS:].copy()
    out["train_period"] = [str(train.date.min().date()), str(train.date.max().date())]
    out["test_period"] = [str(test.date.min().date()), str(test.date.max().date())]

    # --- Modèles ---
    test["pred_naive"] = test.lag_7                                           # même jour la semaine passée
    ridge = Ridge(alpha=1.0).fit(design_ridge(train), train.bags)
    test["pred_ridge"] = ridge.predict(design_ridge(test).reindex(columns=design_ridge(train).columns, fill_value=0))
    gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)
    gbm.fit(train[FEATURES], train.bags)
    test["pred_gbm"] = gbm.predict(test[FEATURES])

    res = {name: metrics(test.bags, test[col]) for name, col in
           [("Naïf (J-7)", "pred_naive"), ("Régression Ridge", "pred_ridge"), ("Gradient boosting", "pred_gbm")]}
    out["daily_metrics"] = {k: {m: round(v, 2) for m, v in r.items()} for k, r in res.items()}
    thr = 1.3 * train.bags.mean()
    out["peak_detection"] = {name: {k: round(v, 3) for k, v in peak_detection(test.bags.values, test[col].values, thr).items()}
                             for name, col in [("Naïf (J-7)", "pred_naive"), ("Régression Ridge", "pred_ridge"),
                                               ("Gradient boosting", "pred_gbm")]}
    # Biais sur les jours de pic : le modèle sous-estime-t-il les plus gros pics ?
    pk_mask = test.bags >= thr
    out["peak_bias_%"] = {name: round(float(((test.loc[pk_mask, col] - test.loc[pk_mask, "bags"]) / test.loc[pk_mask, "bags"]).mean() * 100), 2)
                          for name, col in [("Naïf (J-7)", "pred_naive"), ("Régression Ridge", "pred_ridge"), ("Gradient boosting", "pred_gbm")]}
    top5 = test.nlargest(5, "bags")
    out["top5_days_bias_%_gbm"] = round(float(((top5.pred_gbm - top5.bags) / top5.bags).mean() * 100), 2)
    best = min(res, key=lambda k: res[k]["MAE"])
    out["best_daily_model"] = best
    col_best = {"Naïf (J-7)": "pred_naive", "Régression Ridge": "pred_ridge", "Gradient boosting": "pred_gbm"}[best]

    # --- Déclinaison horaire : prévision journalière x profil horaire du type de jour ---
    hm = hourly_matrix(bags)
    hm = hm.loc[hm.index.isin(d.date)]
    share = hm.div(hm.sum(axis=1), axis=0)
    gkey = d.set_index("date")[["is_weekend", "is_school_holiday"]]
    train_share = share.loc[share.index.isin(train.date)]
    prof = train_share.groupby([gkey.loc[train_share.index, "is_weekend"].values,
                                gkey.loc[train_share.index, "is_school_holiday"].values]).mean()
    rows = []
    for r in test.itertuples():
        key = (r.is_weekend, r.is_school_holiday)
        p = prof.loc[key] if key in prof.index else prof.mean()
        rows.append(getattr(r, col_best) * p.values)
    hp = pd.DataFrame(rows, index=test.date, columns=range(24))
    ha = hm.loc[test.date]
    hb = hm.shift(HORIZON).loc[test.date]            # même heure la semaine passée
    hb = hb.fillna(0)

    # Heure de pointe : le profil moyen lisse les pics -> on prévoit directement la part de l'heure la plus chargée
    peak_share = train_share.max(axis=1)
    pk = peak_share.groupby([gkey.loc[train_share.index, "is_weekend"].values,
                             gkey.loc[train_share.index, "is_school_holiday"].values]).mean()
    peak_pred = pd.Series([getattr(r, col_best) * (pk.loc[(r.is_weekend, r.is_school_holiday)]
                                                   if (r.is_weekend, r.is_school_holiday) in pk.index else pk.mean())
                           for r in test.itertuples()], index=test.date)
    # Modèle fondé sur le programme de vols (connu à l'avance : aucune fuite d'information)
    hs = schedule_hourly(flights, bags[bags.date.isin(train.date)], flights[flights.scheduled_dep.dt.normalize().isin(train.date)], pd.DatetimeIndex(test.date))
    out["hourly_schedule"] = {
        "WAPE_%": round(float(np.abs(ha - hs).values.sum() / ha.values.sum() * 100), 2),
        "peak_hour_MAPE_%": round(float((np.abs(ha.max(axis=1) - hs.max(axis=1)) / ha.max(axis=1)).mean() * 100), 2),
        "peak_hour_MAE": round(float(np.abs(ha.max(axis=1) - hs.max(axis=1)).mean()), 1),
        "hour_of_peak_hit_%": round(float((ha.idxmax(axis=1) == hs.idxmax(axis=1)).mean() * 100), 1),
        "hour_of_peak_within_1h_%": round(float(((ha.idxmax(axis=1) - hs.idxmax(axis=1)).abs() <= 1).mean() * 100), 1),
    }
    out["hourly"] = {
        "model": best,
        "WAPE_%_model": round(float(np.abs(ha - hp).values.sum() / ha.values.sum() * 100), 2),
        "WAPE_%_naive": round(float(np.abs(ha - hb).values.sum() / ha.values.sum() * 100), 2),
        "peak_hour_MAPE_%_profile": round(float((np.abs(ha.max(axis=1) - hp.max(axis=1)) / ha.max(axis=1)).mean() * 100), 2),
        "peak_hour_MAPE_%_model": round(float((np.abs(ha.max(axis=1) - peak_pred) / ha.max(axis=1)).mean() * 100), 2),
        "peak_hour_MAPE_%_naive": round(float((np.abs(ha.max(axis=1) - hb.max(axis=1)) / ha.max(axis=1)).mean() * 100), 2),
        "peak_hour_actual_mean": round(float(ha.max(axis=1).mean()), 1),
    }

    # --- Figures ---
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(test.date, test.bags, color="black", lw=1.8, label="Réel")
    ax.plot(test.date, test.pred_naive, color=GREY, lw=1.2, ls="--", label=f"Naïf J-7 (MAE {res['Naïf (J-7)']['MAE']:.0f})")
    ax.plot(test.date, test.pred_ridge, color=ORANGE, lw=1.2, label=f"Ridge (MAE {res['Régression Ridge']['MAE']:.0f})")
    ax.plot(test.date, test.pred_gbm, color=GREEN, lw=1.5, label=f"Gradient boosting (MAE {res['Gradient boosting']['MAE']:.0f})")
    ax.axhline(thr, color=RED, lw=0.8, ls=":", label="Seuil de pic (1,3 x moyenne)")
    ax.set_ylabel("Bagages par jour"); ax.set_title(f"Prévision à J+7 sur les {TEST_DAYS} derniers jours (jamais vus à l'entraînement)")
    ax.legend(ncol=2, fontsize=8); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(fig_dir / "05_prevision_quotidienne.png", dpi=150); plt.close(fig)

    peak_day = test.loc[test.bags.idxmax(), "date"]
    out["hourly_example_day"] = str(peak_day.date())
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(24)
    ax.bar(x, ha.loc[peak_day].values, width=0.8, color=BLUE, alpha=0.85, label="Réel")
    ax.plot(x, hs.loc[peak_day].values, color=GREEN, marker="o", lw=2, label="Prévu (programme de vols)")
    ax.plot(x, hp.loc[peak_day].values, color=ORANGE, marker="s", lw=1.2, ls="--", label="Prévu (profil moyen)")
    ax.set_xlabel("Heure de dépôt"); ax.set_ylabel("Bagages")
    ax.set_title(f"Prévision horaire le jour de pic ({peak_day:%d/%m/%Y})"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "06_prevision_horaire_jour_de_pic.png", dpi=150); plt.close(fig)
    return out
