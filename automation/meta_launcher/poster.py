"""Mock posting clients for Tessrax meta launcher.

All clients default to dry-run behavior and avoid live network calls unless
explicitly invoked with ``live=True``. Even in live mode, the clients only
emit placeholder text to demonstrate where API integrations would occur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PostPayload:
    """Represents a prepared post payload."""

    channel: str
    title: str
    body: str
    metadata: Dict[str, str]


class MockClient:
    """Base mock client that prints dry-run or live actions."""

    name: str = "generic"

    def post(self, payload: PostPayload, *, live: bool = False) -> str:
        mode = "LIVE" if live else "DRY-RUN"
        return (
            f"[{mode}] {self.name} client queued post to {payload.channel}: "
            f"{payload.title}"
        )


class RedditClient(MockClient):
    name = "reddit"


class HackerNewsClient(MockClient):
    name = "hacker_news"


class ForumClient(MockClient):
    name = "forum"


class GithubClient(MockClient):
    name = "github"


class LinkedinClient(MockClient):
    name = "linkedin"


class XClient(MockClient):
    name = "x"


class NewsletterClient(MockClient):
    name = "newsletter"


class BlogClient(MockClient):
    name = "blog"


class AcademicClient(MockClient):
    name = "academic"


CLIENTS = {
    "reddit": RedditClient(),
    "hacker_news": HackerNewsClient(),
    "forum": ForumClient(),
    "github": GithubClient(),
    "linkedin": LinkedinClient(),
    "x": XClient(),
    "newsletter": NewsletterClient(),
    "blog": BlogClient(),
    "academic": AcademicClient(),
}


def get_client(client_id: str) -> MockClient:
    """Return a mock client for the requested channel."""

    try:
        return CLIENTS[client_id]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown client: {client_id}") from exc


def dispatch_post(payload: PostPayload, *, live: bool = False) -> str:
    """Route the payload to the appropriate mock client."""

    client = get_client(payload.metadata["client"])
    return client.post(payload, live=live)
