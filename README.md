# Visit with Us: Wellness Tourism Purchase Prediction

This project ranks prospective buyers before a sales contact. A serializable pipeline
cleans customer attributes, engineers two features, imputes and encodes within training
folds, and trains a classifier. GitHub Actions transfers validated data and splits as
artifacts, logs hyperparameter trials to MLflow, tests the application, and commits
the accepted model to `main` for Streamlit Community Cloud.

## Repository layout

```text
.github/workflows/pipeline.yml
.streamlit/config.toml
tourism_project/
  data/tourism.csv
  data/sample_customers.csv
  model_building/schema.py
  model_building/data_register.py
  model_building/prep.py
  model_building/train.py
  deployment/app.py
  deployment/inference.py
  deployment/model.joblib
  deployment/model_metadata.json
  deployment/requirements.txt
  reports/
  requirements.txt
tests/test_pipeline.py
publish_github.py
collect_evidence.py
```

## Local execution

Python 3.12 is the tested version and is used in CI and deployment.

```bash
python -m pip install -r tourism_project/requirements.txt
python -m tourism_project.model_building.data_register
python -m tourism_project.model_building.prep
python -m tourism_project.model_building.train
python -m pytest tests -q
python -m streamlit run tourism_project/deployment/app.py
```

## Modeling decisions

- Remove `CustomerID` and the exported row index from predictors.
- Correct `Fe Male` to `Female`, retaining `Single` and `Unmarried` as separate categories.
- Exclude pitch duration, follow-ups, pitched product, satisfaction and contact type,
  because they are not assumed available before the future contact.
- Keep identical pre-contact profiles in the same partition and CV fold, even if
  they belong to different customer IDs. Do not silently delete legitimate customers.
- Select the model using training grouped-CV average precision. Optimize F2 on
  training out-of-fold scores to favor recall, acknowledging that actual contact
  costs and purchase margins are not provided.
- Evaluate the selected model once on the holdout; compare the fixed threshold with
  0.5 for transparency. Neither comparison chooses a threshold using test labels.
- Scores are uncalibrated ranking scores, not assured purchase probabilities.
- The historical target is a proxy for the new Wellness offer: the data has no
  prospective Wellness campaign outcomes or dates to establish temporal validity.

## CI/CD and experiment evidence

`registered-data`, `data-splits`, `experiment-tracking`, and `trained-model` are
workflow artifacts. The first three jobs have read-only repository permissions.
Only the promotion job can write model artifacts to `main`. It rejects stale runs
and uses a normal fast-forward push, without force pushes. Promotion requires model
quality checks and eight automated tests. `[skip ci]` and path exclusions avoid
retraining on generated model commits.

MLflow runs are stored in `tourism_project/mlruns` locally and retained as a
workflow artifact for 30 days. Every hyperparameter combination has a child run,
with per-fold scores, means, standard deviations, parameters and algorithm name.
The parent run includes the winner, threshold, test metrics, data hash and model.
The CSV trial ledger is also committed in reports for convenient inspection.

## Public deployment status

- [Public project repository](https://github.com/rahulsharma-github/visit-with-us-tourism-mlops)
- [Hosted workflow runs](https://github.com/rahulsharma-github/visit-with-us-tourism-mlops/actions/workflows/pipeline.yml)
- [Streamlit application URL supplied by the project owner](https://visit-with-us-tourism-mlops.streamlit.app/)

The initial four-job hosted workflow passed, including automatic model promotion
to `main`. Use the workflow link to inspect the latest execution and its artifacts.
On 2026-09-27, the project owner confirmed that the Streamlit app produces a
prediction in an incognito/private browser without login. This owner-reported
check is recorded separately from the automated HTTP health probe. An HTML
response from the health URL is not treated as proof of success or failure.
Genuine deployment screenshots are still required for submission. The notebook records
live run status, the model's source commit, the actual app URL and genuine service
screenshots in its Output Evaluation section. Missing app evidence remains pending.
See `RUN_AND_SUBMIT.md` for the deployment and evidence requirements.

## References

- [scikit-learn: preventing data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/tracking-api)
- [GitHub workflow artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data)
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
