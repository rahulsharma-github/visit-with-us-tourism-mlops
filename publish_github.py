'''Publish the prepared project in one GitHub commit using an environment-held token.'''

import argparse
import base64
import os
import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


def publish(repository_name="visit-with-us-tourism-mlops", expected_owner="rahulsharma-github", include_paths=None):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        # Retrieve an already-authorized CLI credential without printing or persisting it.
        try:
            credential = subprocess.run(["gh", "auth", "token", "--hostname", "github.com"], capture_output=True, text=True, check=True)
            token = credential.stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            token = None
    if not token:
        raise RuntimeError("A GitHub token with repository and workflow permissions is required in GH_TOKEN.")
    if not repository_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in repository_name):
        raise ValueError("Supply a repository name, without slashes or spaces.")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                            "X-GitHub-Api-Version": "2022-11-28"})

    def api(method, path, **kwargs):
        response = session.request(method, f"https://api.github.com{path}", timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()

    owner = api("GET", "/user")["login"]
    if owner.lower() != expected_owner.lower():
        raise RuntimeError(f"Signed-in account is {owner}; the requested owner is {expected_owner}.")
    repo = f"{owner}/{repository_name}"
    response = session.get(f"https://api.github.com/repos/{repo}", timeout=30)
    if response.status_code == 404:
        api("POST", "/user/repos", json={"name": repository_name, "private": False, "auto_init": True,
            "description": "Visit with Us: pre-contact purchase propensity, MLflow and GitHub Actions."})
    else:
        response.raise_for_status()
        # Refuse to overwrite an unrelated existing repository.
        marker = session.get(f"https://api.github.com/repos/{repo}/contents/tourism_project/model_building/schema.py", timeout=30)
        if marker.status_code != 200:
            raise RuntimeError("Existing repository is not this project. Use a new repository name.")
    info = api("GET", f"/repos/{repo}")
    branch = info["default_branch"]
    if branch != "main":
        raise RuntimeError("This workflow expects main; choose a new repository with main as default.")
    reference = api("GET", f"/repos/{repo}/git/ref/heads/main")
    parent_sha = reference["object"]["sha"]
    parent = api("GET", f"/repos/{repo}/git/commits/{parent_sha}")
    allowed_roots = [ROOT / "tourism_project", ROOT / "tests", ROOT / ".github", ROOT / ".streamlit"]
    files = []
    for folder in allowed_roots:
        for path in folder.rglob("*"):
            if path.is_file() and not set(path.parts) & {"__pycache__", "mlruns", "splits", ".pytest_cache"}:
                if path.name != "secrets.toml":
                    files.append(path)
    for name in ["README.md", "RUN_AND_SUBMIT.md", ".gitignore", "publish_github.py", "collect_evidence.py"]:
        if (ROOT / name).exists():
            files.append(ROOT / name)
    if include_paths is not None:
        # Scoped updates preserve models and reports already promoted by the hosted workflow.
        requested = {str(Path(name)) for name in include_paths}
        allowed = {str(path.relative_to(ROOT)) for path in files}
        if not requested or not requested <= allowed:
            raise ValueError("Every scoped update must name an allowed project file.")
        files = [path for path in files if str(path.relative_to(ROOT)) in requested]
    entries = []
    for path in sorted(set(files)):
        blob = api("POST", f"/repos/{repo}/git/blobs", json={"content": base64.b64encode(path.read_bytes()).decode(), "encoding": "base64"})
        entries.append({"path": str(path.relative_to(ROOT)), "mode": "100644", "type": "blob", "sha": blob["sha"]})
    tree = api("POST", f"/repos/{repo}/git/trees", json={"base_tree": parent["tree"]["sha"], "tree": entries})
    commit = api("POST", f"/repos/{repo}/git/commits", json={"message": "Implement tourism MLOps pipeline and application",
                 "tree": tree["sha"], "parents": [parent_sha]})
    api("PATCH", f"/repos/{repo}/git/refs/heads/main", json={"sha": commit["sha"], "force": False})
    print(f"Published {len(entries)} files in one commit: https://github.com/{repo}/commit/{commit['sha']}")
    return {"github_repo": repo, "repository_url": f"https://github.com/{repo}", "source_commit": commit["sha"],
            "actions_url": f"https://github.com/{repo}/actions", "streamlit_url": ""}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="visit-with-us-tourism-mlops")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    result = publish(args.repo, include_paths=args.only)
    (ROOT / "evidence").mkdir(exist_ok=True)
    evidence = ROOT / "evidence/deployment.json"
    previous = json.loads(evidence.read_text()) if evidence.exists() else {}
    result["streamlit_url"] = previous.get("streamlit_url", "")
    evidence.write_text(json.dumps(result, indent=2))
