'''Retrieve real public deployment evidence; never synthesize workflow success.'''

import argparse
import base64
import json
import os
import subprocess
from datetime import datetime, timezone
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
    # Optional authentication avoids shared public API rate limits; credentials never leave GitHub.
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            token = subprocess.run(["gh", "auth", "token", "--hostname", "github.com"],
                                   capture_output=True, text=True, check=True).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            token = None
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    def get(path):
        response = session.get(api + path, timeout=30)
        response.raise_for_status()
        return response.json()

    repository = get(f"/repos/{repo}")
    runs = get(f"/repos/{repo}/actions/workflows/pipeline.yml/runs?per_page=10")["workflow_runs"]
    main_runs = [r for r in runs if r["head_branch"] == "main" and r["event"] in {"push", "workflow_dispatch"}]
    run = main_runs[0] if main_runs else None
    jobs = get(f"/repos/{repo}/actions/runs/{run['id']}/jobs")["jobs"] if run else []
    # Inspect the last model-metadata commit, not a later documentation-only commit.
    model_commits = get(f"/repos/{repo}/commits?path=tourism_project/deployment/model_metadata.json&per_page=1")
    model_commit = model_commits[0] if model_commits else {}
    model_metadata = {}
    hosted_preparation = {}
    if model_commit:
        content = get(f"/repos/{repo}/contents/tourism_project/deployment/model_metadata.json?ref={model_commit['sha']}")
        model_metadata = json.loads(base64.b64decode(content["content"]))
        preparation = get(f"/repos/{repo}/contents/tourism_project/reports/preparation.json?ref={model_commit['sha']}")
        hosted_preparation = json.loads(base64.b64decode(preparation["content"]))
    promotion_verified = bool(
        run and (model_commit.get("author") or {}).get("login") == "github-actions[bot]"
        and model_metadata.get("source_commit") == run["head_sha"]
        and model_metadata.get("quality_gate_passed")
    )
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "github_repo": repo, "repository_url": repository["html_url"], "repository_public": not repository["private"],
        "workflow_url": run["html_url"] if run else "", "workflow_conclusion": run["conclusion"] if run else "not_run",
        "workflow_source_commit": run["head_sha"] if run else "",
        "jobs": [{"name": job["name"], "conclusion": job["conclusion"]} for job in jobs],
        "all_required_jobs_passed": REQUIRED_JOBS <= {j["name"] for j in jobs if j["conclusion"] == "success"},
        "automatic_model_commit_verified": promotion_verified,
        "model_commit_url": model_commit.get("html_url", ""),
        "hosted_model_source_commit": model_metadata.get("source_commit", ""),
        "hosted_test_metrics": model_metadata.get("test_metrics", {}),
        "hosted_threshold": model_metadata.get("threshold"),
        "hosted_best_parameters": model_metadata.get("best_parameters", {}),
        "hosted_source_sha256": model_metadata.get("source_sha256", ""),
        "hosted_split_hashes": hosted_preparation.get("file_hashes", {}),
        "hosted_model_sha256": model_metadata.get("model_sha256", ""),
        "streamlit_url": streamlit_url, "streamlit_health_ok": False,
        "streamlit_health_check": {},
    }
    if streamlit_url:
        parsed = urlparse(streamlit_url)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".streamlit.app"):
            raise ValueError("Expected the public HTTPS Streamlit Community Cloud app URL.")
        # A cloud HTML shell is not a successful health response or a verified prediction.
        try:
            response = requests.get(streamlit_url.rstrip("/") + "/_stcore/health", timeout=60)
            result["streamlit_health_ok"] = response.status_code == 200 and response.text.strip().lower() == "ok"
            result["streamlit_health_check"] = {
                "http_status": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "redirect_count": len(response.history),
                "result": "verified" if result["streamlit_health_ok"] else "unverified_response",
            }
        except requests.RequestException as exc:
            result["streamlit_health_check"] = {"result": "request_failed", "error_type": type(exc).__name__}
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    confirmation_path = evidence / "browser_confirmation.json"
    confirmation = json.loads(confirmation_path.read_text()) if confirmation_path.exists() else {}
    # Retain provenance: an owner's dated browser report is not an automated interaction test.
    result["owner_browser_confirmation"] = confirmation if (
        confirmation.get("streamlit_url", "").rstrip("/") == streamlit_url.rstrip("/")
        and confirmation.get("public_prediction_confirmed") is True
    ) else {}
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
