"""SQLite connection, schema creation, FTS preparation, and integrity helpers."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "1.0.0-phase6"
PHASE7_SCHEMA_VERSION = "1.1.0-phase7"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = PROJECT_ROOT / "sql"
REQUIRED_TABLES: tuple[str, ...] = (
    "conditions", "condition_aliases", "symptoms", "condition_symptoms",
    "doshas", "condition_doshas", "recommendations", "recommendation_categories",
    "formulations", "ingredients", "formulation_ingredients", "contraindications",
    "safety_claims", "evidence_sources", "knowledge_claims", "claim_evidence",
    "knowledge_rules", "rule_evidence", "rule_validation", "knowledge_base_versions",
    "model_predictions", "retrieval_results", "final_recommendations", "stale_items",
    "final_recommendation_claims", "audit_log",
)


def connect_database(path: Path | str) -> sqlite3.Connection:
    """Open SQLite with foreign-key enforcement enabled for every connection."""

    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def file_sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest without loading the whole file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _execute_sql_file(connection: sqlite3.Connection, path: Path) -> None:
    connection.executescript(path.read_text(encoding="utf-8"))


def fts5_available(connection: sqlite3.Connection) -> bool:
    """Probe FTS5 without leaving any persistent object behind."""

    try:
        connection.execute(
            "CREATE VIRTUAL TABLE temp.phase6_fts5_probe USING fts5(content)"
        )
        connection.execute("DROP TABLE temp.phase6_fts5_probe")
    except sqlite3.OperationalError:
        return False
    return True


def prepare_fts(connection: sqlite3.Connection) -> bool:
    """Create lightweight search indexes and synchronization triggers when supported."""

    if not fts5_available(connection):
        return False
    connection.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS symptom_search_fts USING fts5(
            symptom_text, normalized_text,
            content='symptoms', content_rowid='symptom_id'
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS recommendation_search_fts USING fts5(
            recommendation_text, normalized_text,
            content='recommendations', content_rowid='recommendation_id'
        );

        CREATE TRIGGER IF NOT EXISTS trg_symptoms_fts_ai AFTER INSERT ON symptoms BEGIN
            INSERT INTO symptom_search_fts(rowid, symptom_text, normalized_text)
            VALUES (NEW.symptom_id, NEW.symptom_text, NEW.normalized_text);
        END;
        CREATE TRIGGER IF NOT EXISTS trg_symptoms_fts_ad AFTER DELETE ON symptoms BEGIN
            INSERT INTO symptom_search_fts(symptom_search_fts, rowid, symptom_text, normalized_text)
            VALUES ('delete', OLD.symptom_id, OLD.symptom_text, OLD.normalized_text);
        END;
        CREATE TRIGGER IF NOT EXISTS trg_symptoms_fts_au AFTER UPDATE ON symptoms BEGIN
            INSERT INTO symptom_search_fts(symptom_search_fts, rowid, symptom_text, normalized_text)
            VALUES ('delete', OLD.symptom_id, OLD.symptom_text, OLD.normalized_text);
            INSERT INTO symptom_search_fts(rowid, symptom_text, normalized_text)
            VALUES (NEW.symptom_id, NEW.symptom_text, NEW.normalized_text);
        END;

        CREATE TRIGGER IF NOT EXISTS trg_recommendations_fts_ai AFTER INSERT ON recommendations BEGIN
            INSERT INTO recommendation_search_fts(rowid, recommendation_text, normalized_text)
            VALUES (NEW.recommendation_id, NEW.recommendation_text, NEW.normalized_text);
        END;
        CREATE TRIGGER IF NOT EXISTS trg_recommendations_fts_ad AFTER DELETE ON recommendations BEGIN
            INSERT INTO recommendation_search_fts(
                recommendation_search_fts, rowid, recommendation_text, normalized_text
            ) VALUES ('delete', OLD.recommendation_id, OLD.recommendation_text, OLD.normalized_text);
        END;
        CREATE TRIGGER IF NOT EXISTS trg_recommendations_fts_au AFTER UPDATE ON recommendations BEGIN
            INSERT INTO recommendation_search_fts(
                recommendation_search_fts, rowid, recommendation_text, normalized_text
            ) VALUES ('delete', OLD.recommendation_id, OLD.recommendation_text, OLD.normalized_text);
            INSERT INTO recommendation_search_fts(rowid, recommendation_text, normalized_text)
            VALUES (NEW.recommendation_id, NEW.recommendation_text, NEW.normalized_text);
        END;
        """
    )
    return True


def rebuild_fts(connection: sqlite3.Connection) -> None:
    """Rebuild existing FTS external-content indexes after idempotent seeding."""

    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "symptom_search_fts" in tables:
        connection.execute(
            "INSERT INTO symptom_search_fts(symptom_search_fts) VALUES ('rebuild')"
        )
    if "recommendation_search_fts" in tables:
        connection.execute(
            "INSERT INTO recommendation_search_fts(recommendation_search_fts) VALUES ('rebuild')"
        )


def initialize_database(
    connection: sqlite3.Connection,
    *,
    source_manifest_sha256: str,
) -> bool:
    """Create all idempotent database objects and record the current schema version."""

    for filename in ("schema.sql", "indexes.sql", "triggers.sql"):
        _execute_sql_file(connection, SQL_DIR / filename)
    connection.execute(
        "UPDATE knowledge_base_versions SET is_current = 0 "
        "WHERE schema_version <> ? AND is_current = 1",
        (SCHEMA_VERSION,),
    )
    connection.execute(
        """
        INSERT INTO knowledge_base_versions(
            schema_version, description, source_manifest_sha256, is_current
        ) VALUES (?, ?, ?, 1)
        ON CONFLICT(schema_version) DO NOTHING
        """,
        (
            SCHEMA_VERSION,
            "Phase 6 provenance-aware five-condition coursework knowledge base",
            source_manifest_sha256,
        ),
    )
    recorded_hash = connection.execute(
        "SELECT source_manifest_sha256 FROM knowledge_base_versions WHERE schema_version = ?",
        (SCHEMA_VERSION,),
    ).fetchone()[0]
    if recorded_hash != source_manifest_sha256:
        raise ValueError(
            "Existing schema version was built from a different Phase 5 manifest."
        )
    available = prepare_fts(connection)
    connection.commit()
    return available


def apply_phase7_migration(
    connection: sqlite3.Connection,
    *,
    parent_manifest_sha256: str,
) -> bool:
    """Apply the non-destructive Phase 7 rule-metadata migration once."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            migration_sha256 TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    migrations = (
        ("002_phase7_rule_lifecycle", "002_phase7_rule_lifecycle.sql"),
        ("003_phase7_rule_validation_status", "003_phase7_rule_validation_status.sql"),
        ("004_phase7_rule_staleness_audit", "004_phase7_rule_staleness_audit.sql"),
    )
    changed = False
    for migration_id, filename in migrations:
        migration_path = SQL_DIR / "migrations" / filename
        migration_hash = file_sha256(migration_path)
        existing = connection.execute(
            "SELECT migration_sha256 FROM schema_migrations WHERE migration_id = ?",
            (migration_id,),
        ).fetchone()
        if existing:
            if existing[0] != migration_hash:
                raise ValueError(
                    f"Applied migration {migration_id} no longer matches its source file."
                )
            continue
        _execute_sql_file(connection, migration_path)
        connection.execute(
            "INSERT INTO schema_migrations(migration_id, schema_version, migration_sha256) VALUES (?, ?, ?)",
            (migration_id, PHASE7_SCHEMA_VERSION, migration_hash),
        )
        changed = True
    if changed:
        connection.execute(
            "UPDATE knowledge_base_versions SET is_current = 0 WHERE is_current = 1"
        )
        connection.execute(
            """
            INSERT INTO knowledge_base_versions(
                schema_version, description, source_manifest_sha256, is_current
            ) VALUES (?, ?, ?, 1)
            ON CONFLICT(schema_version) DO UPDATE SET
                description=excluded.description,
                source_manifest_sha256=excluded.source_manifest_sha256,
                is_current=1
            """,
            (
                PHASE7_SCHEMA_VERSION,
                "Phase 7 structured rule lifecycle and evidence-status separation",
                parent_manifest_sha256,
            ),
        )
    connection.commit()
    return changed


def list_user_indexes(connection: sqlite3.Connection) -> set[str]:
    """Return explicitly named indexes, excluding SQLite auto-indexes."""

    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex_%'"
        )
    }


def table_row_counts(
    connection: sqlite3.Connection, tables: Iterable[str] = REQUIRED_TABLES
) -> dict[str, int]:
    """Count records in a trusted, fixed table allowlist."""

    counts: dict[str, int] = {}
    for table in tables:
        counts[table] = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    return counts


def foreign_key_violations(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return any orphaned relationship rows reported by SQLite."""

    return list(connection.execute("PRAGMA foreign_key_check"))
