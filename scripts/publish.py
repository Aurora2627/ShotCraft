"""Publish this research repository using existing GitHub authorization.

Credentials stay in memory and are never printed or written into project files.
"""

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "Aurora2627"
NAME = "ShotCraft"
URL = "https://github.com/" + OWNER + "/" + NAME + ".git"


def main():
    credential = subprocess.run(
        ["git", "credential", "fill"],
        cwd=ROOT,
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never"),
        timeout=45,
    )
    fields = dict(
        line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line
    )
    token = fields.get("password")
    if credential.returncode or not token:
        raise RuntimeError("Existing GitHub login needs to be restored in VS Code")
    proxy = subprocess.run(
        ["git", "config", "--get", "http.proxy"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    handler = (
        urllib.request.ProxyHandler({"https": proxy, "http": proxy})
        if proxy
        else urllib.request.ProxyHandler()
    )
    opener = urllib.request.build_opener(handler)

    def request(path, body=None):
        req = urllib.request.Request(
            "https://api.github.com" + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
                "User-Agent": "ShotCraft-project-publisher",
                "Content-Type": "application/json",
            },
        )
        with opener.open(req, timeout=30) as response:
            return json.load(response)

    if request("/user")["login"].casefold() != OWNER.casefold():
        raise RuntimeError("The authorized GitHub account is not the project owner")
    try:
        existing = request("/repos/" + OWNER + "/" + NAME)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise RuntimeError(
                "Repository lookup failed, HTTP " + str(error.code)
            ) from None
        existing = request(
            "/user/repos",
            {
                "name": NAME,
                "private": False,
                "description": "Constrained local video editing research system with a VideoPainter backend",
                "has_wiki": False,
            },
        )
    if existing["owner"]["login"].casefold() != OWNER.casefold() or existing["private"]:
        raise RuntimeError("Unexpected repository owner or visibility")
    remotes = subprocess.run(
        ["git", "remote"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    if "origin" not in remotes:
        subprocess.run(["git", "remote", "add", "origin", URL], cwd=ROOT, check=True)
    else:
        actual = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if actual != URL:
            raise RuntimeError("Refusing to change an existing origin URL")
    subprocess.run(
        ["git", "push", "--atomic", "-u", "origin", "main", "v0.1.0"],
        cwd=ROOT,
        check=True,
    )
    print("PUBLISHED https://github.com/" + OWNER + "/" + NAME, flush=True)


if __name__ == "__main__":
    main()
