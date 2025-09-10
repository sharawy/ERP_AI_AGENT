"""
SQL helpers for domain agents.

This module defines a simple interface for executing read and write queries
against the SQLite database.  Agents can instantiate a `SQLTool` and use
its `read()` and `write()` methods to interact with their respective
tables.  The implementation delegates to the `database` module.

Usage example:

```python
from erp_app.tools.sql_tool import SQLTool

sales_sql = SQLTool("sales")
rows = sales_sql.read("SELECT * FROM customers WHERE id = ?", (customer_id,))
sales_sql.write(
    "INSERT INTO leads (customer_name, contact_email, message) VALUES (?, ?, ?)",
    (name, email, message),
)
```
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union, Dict

import database


class SQLTool:
    """A helper for executing SQL queries.

    The `name` argument is informational and can be used for logging or
    registration.  It has no effect on query execution.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def read(self, sql: str, params: Union[Sequence[Any], None] = None) -> List[dict]:
        """Execute a SELECT query and return the results as a list of dicts."""
        rows = database.query(sql, params)
        return database.to_dicts(rows)

    def write(self, sql: str, params: Union[Sequence[Any], None] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return the last row ID.

        For statements that do not insert rows, SQLite sets `lastrowid` to
        the row ID of the most recent successful INSERT on the connection.
        """
        with database.transaction() as cur:
            cur.execute(sql, params or [])
            # `lastrowid` will be meaningful for INSERTs; for updates it will
            # reflect the last inserted row in the transaction.
            return cur.lastrowid

    def execute_sql(self, sql: str, params: Dict[str, Any] | None = None):
        """Execute a SQL query and return the results or row count."""
        # Naive safety: block DDL and cross-table wild writes
        bad = ["DROP ", "ALTER ", "TRUNCATE ", "ATTACH ", "DETACH ", "PRAGMA "]
        if any(b in sql.upper() for b in bad):
            raise ValueError("Unsafe SQL blocked")

        try:
            rows = database.query(sql, params)
            return database.to_dicts(rows)
        except Exception:
            return {"rowcount": database.rowcount(sql, params)}