"""Add agent_id and segments table

Revision ID: e4e6f88cabb0
Revises: ca4b539aef90
Create Date: 2026-01-15 22:52:28.319839

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4e6f88cabb0'
down_revision: Union[str, None] = 'ca4b539aef90'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add agent_id column to stream_sessions table
    op.add_column('stream_sessions', sa.Column('agent_id', sa.String(length=255), nullable=True))
    op.create_index(op.f('ix_stream_sessions_agent_id'), 'stream_sessions', ['agent_id'], unique=False)

    # Create segments table
    op.create_table(
        'segments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('turn_number', sa.Integer(), nullable=False),
        sa.Column('start_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('video_frame_count', sa.Integer(), nullable=False),
        sa.Column('audio_frame_count', sa.Integer(), nullable=False),
        sa.Column('recording_id', sa.Integer(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('agent_response', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_segments_session_id'), 'segments', ['session_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_segments_session_id'), table_name='segments')
    op.drop_table('segments')
    op.drop_index(op.f('ix_stream_sessions_agent_id'), table_name='stream_sessions')
    op.drop_column('stream_sessions', 'agent_id')
