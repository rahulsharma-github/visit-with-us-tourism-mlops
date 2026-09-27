'''Single scoring contract for the app and automated deployment tests.'''

import json

import joblib
import numpy as np
import pandas as pd

from tourism_project.model_building.schema import DEPLOYMENT, FEATURES, NUMERIC, clean_features, sha256


def load_bundle():
    metadata = json.loads((DEPLOYMENT / "model_metadata.json").read_text())
    path = DEPLOYMENT / "model.joblib"
    if sha256(path) != metadata["model_sha256"]:
        raise ValueError("Model checksum differs from the committed metadata.")
    if not metadata["quality_gate_passed"]:
        raise ValueError("This model did not pass its promotion gate.")
    return joblib.load(path), metadata


def predict(frame, model, metadata):
    if frame.empty:
        raise ValueError("At least one customer is required.")
    cleaned = clean_features(frame)
    for column in NUMERIC:
        observed = cleaned[column].dropna()
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError(f"{column} must contain finite nonnegative values or blanks.")
    if (cleaned["NumberOfPersonVisiting"] < 1).any():
        raise ValueError("Party size must be at least one.")
    if (cleaned["NumberOfChildrenVisiting"] > cleaned["NumberOfPersonVisiting"]).any():
        raise ValueError("Children cannot exceed the total party size.")
    scores = model.predict_proba(frame[FEATURES])[:, 1]
    return pd.DataFrame({"purchase_score": scores,
                         "priority_contact": (scores >= metadata["threshold"]).astype(int)}, index=frame.index)
