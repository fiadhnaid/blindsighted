"""Initial migration with stream_sessions and recordings tables

Revision ID: ca4b539aef90
Revises: 
Create Date: 2026-01-15 21:48:01.415344

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ca4b539aef90'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create stream_sessions table
    op.create_table(
        'stream_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('room_name', sa.String(length=255), nullable=False),
        sa.Column('room_sid', sa.String(length=255), nullable=True),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('device_id', sa.String(length=255), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('room_name')
    )
    op.create_index(op.f('ix_stream_sessions_room_name'), 'stream_sessions', ['room_name'], unique=True)
    op.create_index(op.f('ix_stream_sessions_user_id'), 'stream_sessions', ['user_id'], unique=False)

    # Create recordings table
    op.create_table(
        'recordings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('room_name', sa.String(length=255), nullable=False),
        sa.Column('r2_key', sa.String(length=512), nullable=False),
        sa.Column('r2_url', sa.String(length=512), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('format', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('egress_id', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('r2_key')
    )
    op.create_index(op.f('ix_recordings_session_id'), 'recordings', ['session_id'], unique=False)
    op.create_index(op.f('ix_recordings_room_name'), 'recordings', ['room_name'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_recordings_room_name'), table_name='recordings')
    op.drop_index(op.f('ix_recordings_session_id'), table_name='recordings')
    op.drop_table('recordings')
    op.drop_index(op.f('ix_stream_sessions_user_id'), table_name='stream_sessions')
    op.drop_index(op.f('ix_stream_sessions_room_name'), table_name='stream_sessions')
    op.drop_table('stream_sessions')
