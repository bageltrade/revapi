"""
Simple API client for the Todo Fixture service used in revapi.

This client reproduces the behaviour of the JavaScript app.
It uses a permanent session cookie `sid=abc123` that the service
accepts unconditionally.

Example usage:
>>> from api_client import TodoClient
>>> client = TodoClient(base_url="http://127.0.0.1:8765")
>>> client.get_todos()
[{"id": 1, "title": "first"}, {"id": 2, "title": "from agent"}]
>>> client.add_todo("new one")
{"id": 3, "title": "new one"}
"""

import requests
from dataclasses import dataclass
from typing import List, Dict, Any

# The client has no fancy auth. The service uses a cookie `sid=abc123`.
# The X-Client header signals the type of client.

@dataclass
class Todo:
    id: int
    title: str

class TodoClient:
    """Client for the Todo REST API.

    The API is tiny: GET /api/todos and POST /api/todos.
    Authentication is performed via the `sid` cookie (value ``abc123``)
    and a custom ``X-Client: fixture`` header.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8765"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TodoClient/1.0 (Python requests)",
            "X-Client": "fixture",
            "Accept": "application/json",
        })
        # Set the session cookie manually; no auth flow.
        self.session.cookies.set("sid", "abc123", domain="127.0.0.1", path="/")

    def _handle_response(self, r: requests.Response) -> Dict[str, Any]:
        try:
            r.raise_for_status()
        except requests.HTTPError as exc:
            # Provide a useful error message
            raise RuntimeError(f"API request failed: {exc.response.status_code} {exc.response.reason}") from exc
        return r.json()

    def get_todos(self) -> List[Todo]:
        """Retrieve the list of todos.

        Returns a list of :class:`Todo` objects.
        """
        url = f"{self.base_url}/api/todos"
        r = self.session.get(url, headers={"Referer": f"{self.base_url}/"})
        data = self._handle_response(r)
        return [Todo(**item) for item in data.get("items", [])]

    def add_todo(self, title: str) -> Todo:
        """Add a new todo with the supplied *title*.

        Returns the newly created :class:`Todo`.
        """
        url = f"{self.base_url}/api/todos"
        payload = {"title": title}
        r = self.session.post(url, json=payload, headers={"Referer": f"{self.base_url}/"})
        data = self._handle_response(r)
        return Todo(**data)


if __name__ == "__main__":
    # Demonstration of the client.
    client = TodoClient()
    print("Current todos:")
    for t in client.get_todos():
        print(f"{t.id}: {t.title}")
    new_todo = client.add_todo("example from script")
    print("Added todo:", new_todo)
    print("Todos after addition:")
    for t in client.get_todos():
        print(f"{t.id}: {t.title}")
