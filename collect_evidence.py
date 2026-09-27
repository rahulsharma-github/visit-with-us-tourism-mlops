'''Retrieve real public deployment evidence; never synthesize workflow success.'''

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent
REQUIRED_JOBS = {"register-dataset", "data-prep", "model-training", "promote-model"}


def collect(repo, streamlit_url=""):
    if len(repo.split("/")) != 2 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. /" for c in repo):
        raise ValueError("Repository must be owner/name.")
    api = "https://api.github.com"
    session = requests.Session()

    def get(path):
        response = session.get(api + path, timeout=30)
        response.raise_for_status()
        return response.json()

    repository = get(f"/repos/{repo}")
    runs = get(f"/repos/{repo}/actions/workflows/pipeline.yml/runs?per_page=10")["workflow_runs"]
    main_runs = [r for r in runs if r["head_branch"] == "main" and r["event"] in {"push", "workflow_dispatch"}]
    run = main_runs[0] if main_runs else None
    jobs = get(f"/repos/{repo}/actions/runs/{run['id']}/jobs")["jobs"] if run else []
    result = {
        "github_repo": repo, "repository_url": repository["html_url"], "repository_public": not repository["private"],
        "workflow_url": run["html_url"] if run else "", "workflow_conclusion": run["conclusion"] if run else "not_run",
        "workflow_source_commit": run["head_sha"] if run else "",
        "jobs": [{"name": job["name"], "conclusion": job["conclusion"]} for job in jobs],
        "all_required_jobs_passed": REQUIRED_JOBS <= {j["name"] for j in jobs if j["conclusion"] == "success"},
        "streamlit_url": streamlit_url, "streamlit_health_ok": False,
    }
    if streamlit_url:
        parsed = urlparse(streamlit_url)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".streamlit.app"):
            raise ValueError("Expected the public HTTPS Streamlit Community Cloud app URL.")
        response = session.get(streamlit_url.rstrip("/") + "/_stcore/health", timeout=60)
        result["streamlit_health_ok"] = response.status_code == 200 and response.text.strip().lower() == "ok"
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    result["screenshots_present"] = {name: (evidence / name).is_file() for name in
                                     ["github_repository.png", "github_workflow.png", "streamlit_app.png"]}
    (evidence / "deployment.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--streamlit-url", default="")
    args = parser.parse_args()
    collect(args.repo, args.streamlit_url)
