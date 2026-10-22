"""Chargement des données du projet 1 (CSV générés par generate_data.py)."""
from pathlib import Path

import pandas as pd

CUTOFF_MIN = 20  # un bagage doit être chargé 20 min avant le départ (règle du projet 1)


def load(data_dir):
    d = Path(data_dir)
    if not (d / "fact_baggage.csv").exists():
        raise SystemExit(
            f"Fichiers introuvables dans '{d}'.\n"
            "Générez d'abord les données avec le projet 1 (python generate_data.py), "
            "puis copiez le dossier data/csv ici, ou utilisez --data <chemin>."
        )
    bags = pd.read_csv(d / "fact_baggage.csv", parse_dates=["checkin_ts", "loaded_ts"])
    flights = pd.read_csv(d / "dim_flight.csv", parse_dates=["scheduled_dep"])
    conv = pd.read_csv(d / "dim_conveyor.csv")
    cal = pd.read_csv(d / "dim_date.csv", parse_dates=["full_date"])
    inc = pd.read_csv(d / "fact_incident.csv", parse_dates=["start_ts"])

    bags = bags.merge(flights[["flight_id", "scheduled_dep", "terminal"]], on="flight_id")
    bags["cutoff"] = bags.scheduled_dep - pd.Timedelta(minutes=CUTOFF_MIN)
    bags["date"] = bags.checkin_ts.dt.normalize()   # jour de dépôt
    bags["hour"] = bags.checkin_ts.dt.hour
    bags["issue"] = (bags.status != "on_time").astype(int)
    return bags, flights, conv, cal, inc
