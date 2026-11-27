import numpy as np
import pandas as pd
import pytest

from apexgrid.models.dnf_hazard import DNFHazardModel


def _training():
    rows = []
    for round_num in range(1, 5):
        for idx in range(6):
            rows.append({
                "Round": round_num,
                "DNFTarget": int(idx == 0),
                "driver_dnf_rate": 0.30 if idx == 0 else 0.02,
                "GridPosition": idx + 1,
                "rain_probability": 0.1,
                "wet_conditions_indicator": 0.0,
            })
    return pd.DataFrame(rows)


def test_hazard_model_produces_bounded_driver_probabilities():
    model = DNFHazardModel().fit(_training())
    target = _training().iloc[:3].copy()
    probs = model.predict_proba(target)
    assert probs.shape == (3,)
    assert np.isfinite(probs).all()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_hazard_model_rejects_unlabelled_training():
    with pytest.raises(ValueError, match="DNFTarget"):
        DNFHazardModel().fit(pd.DataFrame({"driver_dnf_rate": [0.1]}))


def test_chronological_hazard_evaluation_does_not_use_future_rounds():
    result = DNFHazardModel().evaluate_chronological(_training())
    assert result["status"] in {"VALIDATED", "UNAVAILABLE"}
    if result["folds"]:
        assert all(fold["round"] >= 2 for fold in result["folds"])
