'''Validate and register the source CSV before downstream jobs can consume it.'''

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tourism_project.model_building.schema import DATA, EXPECTED, FEATURES, NUMERIC, REPORTS, TARGET, profile_groups, sha256, write_json


def register(path=DATA / "tourism.csv"):
    frame = pd.read_csv(path)
    missing = sorted(set(EXPECTED) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(EXPECTED) - {"Unnamed: 0"})
    if missing or unexpected:
        raise ValueError(f"Schema mismatch: missing={missing}, unexpected={unexpected}")
    if frame.empty or frame["CustomerID"].isna().any() or frame["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be non-null and unique in a non-empty dataset.")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("ProdTaken must contain both binary classes without missing outcomes.")
    for column in NUMERIC:
        values = pd.to_numeric(frame[column], errors="raise").dropna()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid nonnegative numeric values in {column}.")
    if (frame["NumberOfPersonVisiting"] < 1).any():
        raise ValueError("Party size must be at least one.")
    if (frame["NumberOfChildrenVisiting"] > frame["NumberOfPersonVisiting"]).any():
        raise ValueError("Child count cannot exceed party size.")
    groups = profile_groups(frame)
    summary = {
        "source_file": Path(path).name, "sha256": sha256(path), "rows": len(frame), "columns": len(frame.columns),
        "expected_columns_present": True, "missing_values": {k: int(v) for k, v in frame.isna().sum().items()},
        "dtypes": frame.dtypes.astype(str).to_dict(), "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_ids": int(frame.CustomerID.duplicated().sum()),
        "repeated_full_profiles_without_ids": int(frame.drop(columns=["CustomerID", "Unnamed: 0"], errors="ignore").duplicated().sum()),
        "repeated_precontact_profiles": int(groups.duplicated().sum()), "unique_precontact_profiles": int(groups.nunique()),
        "gender_typo_count": int(frame.Gender.eq("Fe Male").sum()),
        "target_counts": {str(k): int(v) for k, v in frame[TARGET].value_counts().items()},
        "purchase_rate": float(frame[TARGET].mean()), "model_input_columns": FEATURES,
    }
    write_json(REPORTS / "registration.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DATA / "tourism.csv")
    register(parser.parse_args().csv)
