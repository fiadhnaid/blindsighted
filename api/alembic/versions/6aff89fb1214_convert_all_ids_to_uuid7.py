"""Convert all IDs to UUID7

Revision ID: 6aff89fb1214
Revises: 604e0a843889
Create Date: 2026-01-16 11:09:33.423621

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6aff89fb1214"
down_revision: Union[str, None] = "604e0a843889"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop segments table (has FK to recordings and stream_sessions)
    op.drop_index("ix_segments_session_id", table_name="segments", if_exists=True)
    op.drop_table("segments", if_exists=True)

    # Drop recordings table (has FK to stream_sessions)
    op.drop_table("recordings", if_exists=True)

    # Drop and recreate stream_sessions with UUID primary key
    op.drop_index("ix_stream_sessions_agent_id", table_name="stream_sessions", if_exists=True)
    op.drop_index("ix_stream_sessions_room_name", table_name="stream_sessions", if_exists=True)
    op.drop_index("ix_stream_sessions_user_id", table_name="stream_sessions", if_exists=True)
    op.drop_table("stream_sessions", if_exists=True)

    # Recreate stream_sessions with UUID primary key
    op.create_table(
        "stream_sessions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("room_name", sa.String(length=255), nullable=False),
        sa.Column("room_sid", sa.String(length=255), nullable=True),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("device_id", sa.String(length=255), nullable=True),
        sa.Column("agent_id", sa.String(length=255), nullable=True),
        sa.Column("session_metadata", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("room_name"),
    )
    op.create_index("ix_stream_sessions_room_name", "stream_sessions", ["room_name"], unique=True)
    op.create_index("ix_stream_sessions_user_id", "stream_sessions", ["user_id"], unique=False)
    op.create_index("ix_stream_sessions_agent_id", "stream_sessions", ["agent_id"], unique=False)

    # Recreate recordings with UUID primary key and FK
    op.create_table(
        "recordings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("room_name", sa.String(length=255), nullable=False),
        sa.Column("r2_key", sa.String(length=512), nullable=False),
        sa.Column("r2_url", sa.String(length=512), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("format", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("egress_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["stream_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("r2_key"),
    )
    op.create_index("ix_recordings_session_id", "recordings", ["session_id"], unique=False)
    op.create_index("ix_recordings_room_name", "recordings", ["room_name"], unique=False)

    # Recreate segments with UUID primary key and FKs
    op.create_table(
        "segments",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("start_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("video_frame_count", sa.Integer(), nullable=False),
        sa.Column("audio_frame_count", sa.Integer(), nullable=False),
        sa.Column("recording_id", sa.UUID(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("agent_response", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["stream_sessions.id"]),
        sa.ForeignKeyConstraint(["recording_id"], ["recordings.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_segments_session_id", "segments", ["session_id"], unique=False)


def downgrade() -> None:
    # Drop segments table
    op.drop_index("ix_segments_session_id", table_name="segments")
    op.drop_table("segments")

    # Drop recordings table
    op.drop_index("ix_recordings_room_name", table_name="recordings")
    op.drop_index("ix_recordings_session_id", table_name="recordings")
    op.drop_table("recordings")

    # Drop stream_sessions table
    op.drop_index("ix_stream_sessions_agent_id", table_name="stream_sessions")
    op.drop_index("ix_stream_sessions_user_id", table_name="stream_sessions")
    op.drop_index("ix_stream_sessions_room_name", table_name="stream_sessions")
    op.drop_table("stream_sessions")

    # Recreate stream_sessions with Integer primary key
    op.create_table(
        "stream_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("room_name", sa.String(length=255), nullable=False),
        sa.Column("room_sid", sa.String(length=255), nullable=True),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("device_id", sa.String(length=255), nullable=True),
        sa.Column("agent_id", sa.String(length=255), nullable=True),
        sa.Column("session_metadata", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("room_name"),
    )
    op.create_index("ix_stream_sessions_room_name", "stream_sessions", ["room_name"], unique=True)
    op.create_index("ix_stream_sessions_user_id", "stream_sessions", ["user_id"], unique=False)
    op.create_index("ix_stream_sessions_agent_id", "stream_sessions", ["agent_id"], unique=False)

    # Recreate recordings with Integer primary key and FK
    op.create_table(
        "recordings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("room_name", sa.String(length=255), nullable=False),
        sa.Column("r2_key", sa.String(length=512), nullable=False),
        sa.Column("r2_url", sa.String(length=512), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("format", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("egress_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("r2_key"),
    )

    # Recreate segments with Integer primary key
    op.create_table(
        "segments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("start_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("video_frame_count", sa.Integer(), nullable=False),
        sa.Column("audio_frame_count", sa.Integer(), nullable=False),
        sa.Column("recording_id", sa.Integer(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("agent_response", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_segments_session_id", "segments", ["session_id"], unique=False)
