"""Long-term memory Store entrypoint for LangGraph platform deployment.

Declared in ``langgraph.json`` under the ``store`` key. The LangGraph runtime
imports ``store`` from this module and injects it into every graph compiled with
a ``store=`` argument, making ``langgraph.config.get_store()`` return this instance.

Backend is selected via the ``MEMORY_STORE`` environment variable (see
``api._build_store`` for semantics): ``memory`` (InMemoryStore, default),
``postgres`` (PostgresStore, requires ``MEMORY_STORE_POSTGRES_DSN``), or
``none`` (no store).
"""

import os

from langgraph.store.base import BaseStore


def _build() -> BaseStore | None:
    mode = os.getenv("MEMORY_STORE", "memory").lower()

    if mode in ("none", "off", "false"):
        return None

    if mode == "postgres":
        dsn = os.getenv("MEMORY_STORE_POSTGRES_DSN") or os.getenv("POSTGRES_DSN")
        if dsn:
            from langgraph.store.postgres import PostgresStore

            s = PostgresStore.from_conn_string(dsn)
            # Synchronous setup is acceptable at import time in the platform runtime.
            try:
                s.setup()
            except Exception:
                # Table may already exist; ignore.
                pass
            return s

    from langgraph.store.memory import InMemoryStore

    return InMemoryStore(index=None)


store: BaseStore | None = _build()
