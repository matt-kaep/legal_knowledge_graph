"""Helpers OVH PostgreSQL — credentials via .env.local racine 05-Technique/."""
from __future__ import annotations
import os
from pathlib import Path
from . import config


def load_env_local() -> None:
    """Lit `.env.local` et exporte les vars dans os.environ."""
    env_path = config.ENV_LOCAL
    if not env_path.exists():
        raise FileNotFoundError(f".env.local introuvable : {env_path}")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def connect():
    """Renvoie une connexion psycopg2 (SSL). load_env_local() implicite."""
    import psycopg2
    load_env_local()
    required = ["OVH_DB_HOST", "OVH_DB_PORT", "OVH_DB_NAME",
                "OVH_DB_USER", "OVH_DB_PASSWORD"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Variables manquantes dans .env.local : {missing}")
    return psycopg2.connect(
        host=os.environ["OVH_DB_HOST"],
        port=os.environ["OVH_DB_PORT"],
        dbname=os.environ["OVH_DB_NAME"],
        user=os.environ["OVH_DB_USER"],
        password=os.environ["OVH_DB_PASSWORD"],
        sslmode="require",
        connect_timeout=15,
    )
