'''Shared data contract; no fitted statistics or target information enter cleaning.'''

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "tourism_project"
DATA = PROJECT / "data"
REPORTS = PROJECT / "reports"
DEPLOYMENT = PROJECT / "deployment"
TARGET = "ProdTaken"
SEED = 42
EXCLUDED = ["TypeofContact", "DurationOfPitch", "NumberOfFollowups", "ProductPitched", "PitchSatisfactionScore"]
CATEGORICAL = ["CityTier", "Occupation", "Gender", "PreferredPropertyStar", "MaritalStatus", "Passport", "OwnCar", "Designation"]
NUMERIC = ["Age", "NumberOfPersonVisiting", "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome"]
FEATURES = NUMERIC + CATEGORICAL
DERIVED = ["AdultsVisiting", "IncomePerTraveller"]
EXPECTED = ["CustomerID", TARGET, *FEATURES, *EXCLUDED]
CHOICES = {
    "CityTier": [1, 2, 3], "PreferredPropertyStar": [3, 4, 5], "Passport": [0, 1], "OwnCar": [0, 1],
    "Occupation": ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    "Gender": ["Female", "Male"], "MaritalStatus": ["Single", "Unmarried", "Married", "Divorced"],
    "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
}


def write_json(path, payload):
    '''Persist machine-readable evidence without NumPy serialization ambiguity.'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_features(frame):
    '''Apply deterministic corrections; imputation stays inside fitted CV pipelines.'''
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    result = frame[FEATURES].copy()
    # Convert numeric strings consistently in training, batch scoring, and the app.
    for column in NUMERIC + ["CityTier", "PreferredPropertyStar", "Passport", "OwnCar"]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    for column in CATEGORICAL:
        result[column] = result[column].map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
    for column in ["CityTier", "PreferredPropertyStar", "Passport", "OwnCar"]:
        result[column] = result[column].map(lambda value: str(int(float(value))) if pd.notna(value) else np.nan)
    result["Gender"] = result["Gender"].replace({"Fe Male": "Female"})
    # Single and Unmarried remain separate because their business equivalence is unconfirmed.
    return result


def profile_groups(frame):
    '''Put identical pre-contact profiles in one split, independent of their outcomes.'''
    return pd.util.hash_pandas_object(clean_features(frame), index=False).astype(str)


class PrepareFeatures(TransformerMixin, BaseEstimator):
    '''Serializable, stateless cleaning and feature engineering shared with production.'''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = clean_features(X)
        result["AdultsVisiting"] = result["NumberOfPersonVisiting"] - result["NumberOfChildrenVisiting"]
        result["IncomePerTraveller"] = result["MonthlyIncome"] / result["NumberOfPersonVisiting"].replace(0, np.nan)
        return result
