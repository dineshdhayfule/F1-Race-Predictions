"""Chronological driver retirement-risk model."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.utils.logging_utils import get_logger

logger = get_logger("dnf_hazard")


class DNFHazardModel:
    """Estimates in-race retirement probability from pre-race features.

    The label is deliberately narrow: ``1`` means canonical RETIRED and ``0``
    means canonical FINISHED. DNS, DSQ, NC, and UNKNOWN are excluded.
    """

    feature_cols = (
        "driver_dnf_rate",
        "GridPosition",
        "rain_probability",
        "wet_conditions_indicator",
    )

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[Pipeline] = None
        self.training_rows = 0
        self.positive_rate: Optional[float] = None

    @classmethod
    def _eligible_rows(cls, df: pd.DataFrame) -> pd.DataFrame:
        if "DNFTarget" not in df.columns:
            raise ValueError("DNFTarget is required to train the DNF hazard model.")
        rows = df[df["DNFTarget"].isin([0, 1])].copy()
        if rows.empty:
            raise ValueError("No explicitly labelled FINISHED/RETIRED rows are available.")
        return rows

    @classmethod
    def _features(cls, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in cls.feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing DNF hazard features: {missing}")
        return df[list(cls.feature_cols)].apply(pd.to_numeric, errors="coerce")

    def fit(self, training_df: pd.DataFrame) -> "DNFHazardModel":
        rows = self._eligible_rows(training_df)
        labels = rows["DNFTarget"].astype(int)
        if labels.nunique() < 2:
            raise ValueError("DNF hazard training requires both FINISHED and RETIRED labels.")
        if len(rows) < 6 or labels.value_counts().min() < 2:
            raise ValueError(
                "Insufficient labelled rows for calibrated DNF hazard training "
                f"(rows={len(rows)}, class_counts={labels.value_counts().to_dict()})."
            )

        base = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", CalibratedClassifierCV(base, method="sigmoid", cv=2)),
        ])
        self.model.fit(self._features(rows), labels)
        self.training_rows = len(rows)
        self.positive_rate = float(labels.mean())
        return self

    def predict_proba(self, target_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DNF hazard model has not been fitted.")
        probabilities = self.model.predict_proba(self._features(target_df))[:, 1]
        if not np.isfinite(probabilities).all():
            raise ValueError("DNF hazard model produced non-finite probabilities.")
        return np.clip(probabilities.astype(float), 0.0, 1.0)

    def evaluate_chronological(self, training_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate later rounds against earlier rounds and report calibration."""
        if "Round" not in training_df.columns:
            raise ValueError("Round is required for chronological DNF validation.")
        rounds = sorted(pd.to_numeric(training_df["Round"], errors="coerce").dropna().unique())
        fold_metrics: List[Dict[str, float]] = []
        for current_round in rounds[1:]:
            train = training_df[training_df["Round"] < current_round]
            test_source = training_df[training_df["Round"] == current_round]
            try:
                test = self._eligible_rows(test_source)
                fold = DNFHazardModel(self.random_state).fit(train)
            except ValueError:
                continue
            if test.empty:
                continue
            y_true = test["DNFTarget"].astype(int).to_numpy()
            probs = fold.predict_proba(test)
            predictions = (probs >= 0.5).astype(int)
            baseline = float(fold.positive_rate or 0.0)
            fold_metrics.append({
                "round": float(current_round),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "brier_score": float(brier_score_loss(y_true, probs)),
                "baseline_brier_score": float(brier_score_loss(
                    y_true, np.full(len(y_true), baseline)
                )),
                "positive_rate": float(y_true.mean()),
            })

        if not fold_metrics:
            return {"status": "UNAVAILABLE", "folds": [], "positive_rate": None}
        return {
            "status": "VALIDATED",
            "folds": fold_metrics,
            "positive_rate": float(training_df["DNFTarget"].dropna().mean()),
        }
