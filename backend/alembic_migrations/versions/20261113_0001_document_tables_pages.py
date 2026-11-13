"""document_tables: ensure pages_json + llm_understanding_json

Revision ID: 20261113_0001
Revises:
Create Date: 2026-11-13

Replaces manual ``_migrate_schema`` DDL alterations for ``document_tables``.
Idempotent: creates the table when missing and adds additive columns when
an older install already has the table.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "20261113_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _is_sqlite() -> bool:
    bind = op.get_bind()
    return bind.dialect.name == "sqlite"


def _text_type():
    return sa.Text()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    text_type = _text_type()

    if "document_tables" not in tables:
        op.create_table(
            "document_tables",
            sa.Column("id", sa.String(160), primary_key=True),
            sa.Column("document_id", sa.CHAR(36), nullable=False),
            sa.Column("user_id", sa.CHAR(36), nullable=False),
            sa.Column("table_id", sa.String(128), nullable=False),
            sa.Column("filename", sa.String(255), nullable=True),
            sa.Column("page_start", sa.Integer(), nullable=True),
            sa.Column("page_end", sa.Integer(), nullable=True),
            sa.Column("pages_json", text_type, nullable=True),
            sa.Column("section", sa.String(512), nullable=True),
            sa.Column("subsection", sa.String(512), nullable=True),
            sa.Column("caption", text_type, nullable=True),
            sa.Column("extraction_method", sa.String(64), nullable=True),
            sa.Column("n_rows", sa.Integer(), nullable=True),
            sa.Column("n_cols", sa.Integer(), nullable=True),
            sa.Column("headers_json", text_type, nullable=True),
            sa.Column("rows_json", text_type, nullable=True),
            sa.Column("bbox_json", text_type, nullable=True),
            sa.Column("merged_cells_json", text_type, nullable=True),
            sa.Column("cells_json", text_type, nullable=True),
            sa.Column("header_rows_json", text_type, nullable=True),
            sa.Column("low_confidence_json", text_type, nullable=True),
            sa.Column("table_json", text_type, nullable=True),
            sa.Column("summary_text", text_type, nullable=True),
            sa.Column("llm_understanding_json", text_type, nullable=True),
            sa.Column(
                "needs_verification",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("index_version", sa.String(32), nullable=True),
            sa.Column(
                "ordinal", sa.Integer(), nullable=False, server_default=sa.text("0")
            ),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint(
                "document_id",
                "table_id",
                name="uq_document_tables_document_table_id",
            ),
        )
        op.create_index("ix_document_tables_document_id", "document_tables", ["document_id"])
        op.create_index("ix_document_tables_user_id", "document_tables", ["user_id"])
        op.create_index("ix_document_tables_table_id", "document_tables", ["table_id"])
        op.create_index(
            "ix_document_tables_document_page",
            "document_tables",
            ["document_id", "page_start"],
        )
        return

    cols = {c["name"] for c in inspector.get_columns("document_tables")}
    if "llm_understanding_json" not in cols:
        op.add_column(
            "document_tables",
            sa.Column("llm_understanding_json", text_type, nullable=True),
        )
    if "pages_json" not in cols:
        op.add_column(
            "document_tables",
            sa.Column("pages_json", text_type, nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "document_tables" not in set(inspector.get_table_names()):
        return
    cols = {c["name"] for c in inspector.get_columns("document_tables")}
    if "pages_json" in cols:
        op.drop_column("document_tables", "pages_json")
    # Keep llm_understanding_json on downgrade — dropping it would lose data.
