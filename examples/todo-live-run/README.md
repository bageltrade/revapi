# TodoClient

A tiny Python client for the experimental Todo Fixture API used in the revapi exercises.

## Installation

This project has no external dependencies beyond the Python standard library. The client uses the `requests` package, which is part of the standard any modern Python installation.

```bash
python3 -m pip install requests
```

## Usage

```python
from api_client import TodoClient

client = TodoClient(base_url="http://127.0.0.1:8765")

# List existing todos
for todo in client.get_todos():
    print(todo.id, todo.title)

# Create a new todo
new_todo = client.add_todo("Hello World")
print("Created todo", new_todo)
```

The client automatically handles the session cookie (`sid=abc123`) and required custom headers.

## API Reference

### `TodoClient.get_todos() -> List[Todo]`

Return the current list of todos.

### `TodoClient.add_todo(title: str) -> Todo`

Create a new todo with the supplied title.

## Notes

- No authentication flow is required for this service; the cookie `sid=abc123` is accepted by the server.
- The `X-Client: fixture` header identifies the non‑browser client.
- The implementation is intentionally straightforward; it should work immediately after copying to a local environment.

---

Happy coding!
