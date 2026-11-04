import numpy as np
import pandas as pd

from eval_lib import SPLITS, bootstrap_wape_diff, metrics


def test_prevision_parfaite():
    m = metrics([3, 0, 5], [3, 0, 5])
    assert m["MAE"] == 0 and m["WAPE"] == 0 and m["bias"] == 0


def test_wape_et_biais():
    m = metrics([10, 10], [12, 8])          # erreurs +2 et -2
    assert m["MAE"] == 2
    assert abs(m["WAPE"] - 0.2) < 1e-9      # 4 / 20
    assert m["bias"] == 0                   # les erreurs se compensent


def test_bootstrap_previsions_identiques():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10).repeat(24),
                       "n_pickups": np.arange(240) % 7 + 1})
    df["a"] = df["n_pickups"] + 1
    df["b"] = df["a"]
    lo, hi = bootstrap_wape_diff(df, "a", "b")
    assert lo == 0 and hi == 0


def test_periodes_ordonnees_sans_chevauchement():
    t, v, s = (pd.Timestamp(SPLITS[k][0]) for k in ("train", "val", "test"))
    assert pd.Timestamp(SPLITS["train"][1]) < v < pd.Timestamp(SPLITS["val"][1]) < s