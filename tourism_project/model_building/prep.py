'''Make deterministic, stratified, profile-disjoint train/test workflow artifacts.'''

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tourism_project.model_building.schema import DATA, EXCLUDED, FEATURES, REPORTS, SEED, TARGET, clean_features, profile_groups, sha256, write_json


def prepare(path=DATA / "tourism.csv", output=DATA / "splits"):
    frame = pd.read_csv(path)
    X = clean_features(frame)
    y = frame[TARGET].astype(int)
    groups = profile_groups(X)
    # The first seeded group-stratified fold is the holdout, without searching for a favorable score.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_index, test_index = next(splitter.split(X, y, groups))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, indexes in [("train", train_index), ("test", test_index)]:
        X.iloc[indexes].to_csv(output / f"X_{name}.csv", index=False)
        y.iloc[indexes].to_csv(output / f"y_{name}.csv", index=False)
        groups.iloc[indexes].rename("profile_group").to_csv(output / f"groups_{name}.csv", index=False)
        frame.iloc[indexes][["CustomerID"]].to_csv(output / f"ids_{name}.csv", index=False)
        artifacts[name] = {"rows": len(indexes), "positive_count": int(y.iloc[indexes].sum()), "positive_rate": float(y.iloc[indexes].mean())}
    overlap = set(groups.iloc[train_index]) & set(groups.iloc[test_index])
    if overlap:
        raise AssertionError("Profile leakage between train and test splits.")
    summary = {
        "source_sha256": sha256(path), "seed": SEED, "split_method": "first fold of StratifiedGroupKFold(n_splits=5)",
        "model_input_count": len(FEATURES), "excluded_post_contact": EXCLUDED,
        "excluded_identifiers": ["Unnamed: 0", "CustomerID"], "profile_overlap": len(overlap),
        "splits": artifacts, "file_hashes": {p.name: sha256(p) for p in sorted(output.glob("*.csv"))},
    }
    write_json(output / "split_manifest.json", summary)
    write_json(REPORTS / "preparation.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DATA / "tourism.csv")
    parser.add_argument("--output", type=Path, default=DATA / "splits")
    args = parser.parse_args()
    prepare(args.csv, args.output)
