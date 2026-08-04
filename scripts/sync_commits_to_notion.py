#!/usr/bin/env python3
"""Add commits from a GitHub push to a Notion database."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request


NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2025-09-03"
ZERO_SHA = "0" * 40
REQUIRED_PROPERTIES = {
    "Name": "title",
    "Commit SHA": "rich_text",
    "Repository": "rich_text",
    "Branch": "rich_text",
    "Author": "rich_text",
    "Committed At": "date",
    "URL": "url",
}


class NotionClient:
    def __init__(self, token: str) -> None:
        self.token = token

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{NOTION_API}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"Notion API {error.code}: {detail}") from error

    def resolve_data_source(self, database_id: str) -> str:
        database = self.request("GET", f"/databases/{database_id}")
        data_sources = database.get("data_sources", [])
        if len(data_sources) != 1:
            raise RuntimeError(
                "The Notion database must contain exactly one data source; "
                f"found {len(data_sources)}"
            )
        return data_sources[0]["id"]

    def validate_data_source(self, data_source_id: str) -> None:
        data_source = self.request("GET", f"/data_sources/{data_source_id}")
        properties = data_source.get("properties", {})
        errors = []
        for name, expected_type in REQUIRED_PROPERTIES.items():
            actual_type = properties.get(name, {}).get("type")
            if actual_type != expected_type:
                errors.append(f"{name!r}: expected {expected_type}, got {actual_type or 'missing'}")
        if errors:
            raise RuntimeError("Invalid Notion database schema: " + "; ".join(errors))

    def commit_exists(self, data_source_id: str, sha: str) -> bool:
        result = self.request(
            "POST",
            f"/data_sources/{data_source_id}/query",
            {
                "page_size": 1,
                "filter": {
                    "property": "Commit SHA",
                    "rich_text": {"equals": sha},
                },
            },
        )
        return bool(result.get("results"))

    def create_commit(self, data_source_id: str, commit: dict, repository: str, branch: str) -> None:
        message = commit["message"][:2000]
        self.request(
            "POST",
            "/pages",
            {
                "parent": {"type": "data_source_id", "data_source_id": data_source_id},
                "properties": {
                    "Name": {"title": [{"text": {"content": message}}]},
                    "Commit SHA": {"rich_text": [{"text": {"content": commit["sha"]}}]},
                    "Repository": {"rich_text": [{"text": {"content": repository}}]},
                    "Branch": {"rich_text": [{"text": {"content": branch}}]},
                    "Author": {"rich_text": [{"text": {"content": commit["author"]}}]},
                    "Committed At": {"date": {"start": commit["committed_at"]}},
                    "URL": {"url": f"https://github.com/{repository}/commit/{commit['sha']}"},
                },
            },
        )


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def commits_for_push(before: str, after: str) -> list[dict]:
    if not before or before == ZERO_SHA:
        revisions = [after]
    else:
        revisions = git_output("rev-list", "--reverse", f"{before}..{after}").splitlines()

    commits = []
    for sha in revisions:
        fields = git_output("show", "-s", "--format=%H%x00%an%x00%aI%x00%B", sha).split("\0", 3)
        commits.append(
            {"sha": fields[0], "author": fields[1], "committed_at": fields[2], "message": fields[3].strip()}
        )
    return commits


def main() -> int:
    token = required_env("NOTION_TOKEN")
    database_id = required_env("NOTION_DATABASE_ID")
    repository = required_env("REPOSITORY")
    branch = required_env("BRANCH")
    after = os.environ.get("AFTER_SHA", "").strip() or git_output("rev-parse", "HEAD")
    before = os.environ.get("BEFORE_SHA", "").strip()

    client = NotionClient(token)
    data_source_id = client.resolve_data_source(database_id)
    client.validate_data_source(data_source_id)
    created = 0
    for commit in commits_for_push(before, after):
        if client.commit_exists(data_source_id, commit["sha"]):
            print(f"Already synced: {commit['sha']}")
            continue
        client.create_commit(data_source_id, commit, repository, branch)
        created += 1
        print(f"Synced: {commit['sha']}")
    print(f"Created {created} Notion page(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
