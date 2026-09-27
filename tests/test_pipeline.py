'''Meaningful data, artifact, serialization, and app contract tests.'''

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from tourism_project.deployment.inference import load_bundle, predict
from tourism_project.model_building.data_register import register
from tourism_project.model_building.schema import DATA, EXCLUDED, FEATURES, ROOT, clean_features


def test_registration_rejects_missing_target(tmp_path):
    frame = pd.read_csv(DATA / "tourism.csv").drop(columns="ProdTaken")
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Schema mismatch"):
        register(path)


def test_cleaner_is_idempotent_and_excludes_future_fields():
    frame = pd.read_csv(DATA / "tourism.csv")
    cleaned = clean_features(frame)
    pd.testing.assert_frame_equal(cleaned, clean_features(cleaned))
    assert not set(EXCLUDED + ["CustomerID", "ProdTaken", "Unnamed: 0"]) & set(cleaned)
    assert "Fe Male" not in set(cleaned.Gender)


def test_artifact_profile_and_customer_separation():
    split = DATA / "splits"
    for kind in ["groups", "ids"]:
        train = pd.read_csv(split / f"{kind}_train.csv", dtype=str).iloc[:, 0]
        test = pd.read_csv(split / f"{kind}_test.csv", dtype=str).iloc[:, 0]
        assert not set(train) & set(test)


def test_saved_predictions_reproduce_holdout_and_threshold():
    model, metadata = load_bundle()
    X = pd.read_csv(DATA / "splits" / "X_test.csv")
    expected = pd.read_csv(ROOT / "tourism_project/reports/test_predictions.csv")
    actual = predict(X, model, metadata)
    np.testing.assert_allclose(actual.purchase_score, expected.score, atol=1e-12)
    np.testing.assert_array_equal(actual.priority_contact, expected.predicted)
    assert metadata["quality_gate_passed"]


def test_unknown_category_and_missing_numeric_are_supported():
    model, metadata = load_bundle()
    X = pd.read_csv(DATA / "splits" / "X_test.csv").head(2)
    X.loc[0, "Occupation"] = "New occupation"
    X.loc[1, "MonthlyIncome"] = np.nan
    result = predict(X, model, metadata)
    assert result.purchase_score.between(0, 1).all()


def test_invalid_party_and_schema_fail_clearly():
    model, metadata = load_bundle()
    X = pd.read_csv(DATA / "splits" / "X_test.csv").head(1)
    with pytest.raises(ValueError, match="Missing input columns"):
        predict(X.drop(columns=FEATURES[0]), model, metadata)
    X.loc[0, "NumberOfChildrenVisiting"] = 100
    with pytest.raises(ValueError, match="Children cannot"):
        predict(X, model, metadata)


def test_streamlit_page_and_single_customer_prediction():
    app = AppTest.from_file(str(ROOT / "tourism_project/deployment/app.py"), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == "Purchase score" for metric in app.metric)


def test_workflow_artifact_contract_and_promotion_guard():
    config = yaml.load((ROOT / ".github/workflows/pipeline.yml").read_text(), Loader=yaml.BaseLoader)
    assert "push" in config["on"] and "workflow_dispatch" in config["on"]
    assert config["jobs"]["data-prep"]["needs"] == "register-dataset"
    assert config["jobs"]["model-training"]["needs"] == "data-prep"
    source = (ROOT / ".github/workflows/pipeline.yml").read_text()
    assert "actions/download-artifact@v4" in source and "actions/upload-artifact@v4" in source
    assert "contents: write" in source and "[skip ci]" in source
    assert "git push origin HEAD:main" in source
