# Run and Submit

The notebook and HTML contain actual local outputs. Hosted execution evidence is
required before claiming all deployment-related rubric points.

## Colab

1. Open `Visit_With_Us_AML_MLOps_Project.ipynb` in Colab. A CPU runtime is sufficient.
2. Store `tourism.csv` in MyDrive, or set `DATASET_PATH` to its exact Drive path in
   the configuration cell. Drive is mounted automatically when needed.
3. Run all cells. The notebook creates the complete repository and executes its
   registration, preparation, tuning, tracking, serialization and application tests.
4. For publishing, store a GitHub token in the Colab secret `GITHUB_TOKEN`, grant
   notebook access, and set `PUBLISH_TO_GITHUB = True`. The token needs permission
   to create/update your repository and write Actions workflows. It is never printed.
   A classic token needs `repo` and `workflow`; a suitable fine-grained token must
   grant Contents and Workflows write access to the destination repository.
5. Run the GitHub publication cell once. It infers your username from the token and
   publishes one atomic commit to `visit-with-us-tourism-mlops` (or your chosen name).
6. Open the printed Actions URL. Allow the four-job pipeline to finish successfully.
   If repository policy blocks the bot's commit, enable Actions write permissions
   or use a permitted repository; do not bypass branch protection rules.

## Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository.
2. Create an app from the project repository, branch `main`.
3. Set the main file to `tourism_project/deployment/app.py` and Python to **3.12**.
4. Deploy, make the app public, and test one customer using **Assess customer**.
5. Put the actual `https://...streamlit.app` URL in `STREAMLIT_APP_URL` in the notebook.

The template and scored rubric specify Streamlit Community Cloud; the generic
submission paragraph mentions Hugging Face. This project implements the explicit
Streamlit requirement. Confirm with the program manager only if an additional
Hugging Face mirror is explicitly required by your course.

## Capture real evidence and export

Capture these from the actual service pages, with results legible:

- `evidence/github_repository.png`: the public repository URL and folder structure.
- `evidence/github_workflow.png`: the completed four-job Actions run showing success.
- `evidence/streamlit_app.png`: the public app URL and a displayed customer result.

Place them in the generated workspace's `evidence` folder. They must be genuine
screen captures; charts or constructed tables do not replace deployed-service screenshots.
The notebook embeds these files as image outputs when its Output Evaluation cell runs.
It also queries the actual GitHub workflow and app health endpoint when URLs are supplied.
No screenshot or status is manufactured when evidence is absent.

Rerun only the Output Evaluation and final export cells after adding evidence; there
is no need to retrain solely to insert the screenshots. Download the updated Colab
notebook, place it in the generated workspace (or set `NOTEBOOK_TO_EXPORT` in the
last cell to its path), and run that cell to export the final HTML with visible outputs.
Submit the HTML file, as required by the assignment.
