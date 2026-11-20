"""Initial schema for photo-indexer-web

Revision ID: 20261120_000000
Revises: 
Create Date: 2026-11-20 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20261120_000000'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Core assets table
    op.create_table(
        'enrich_assets',
        sa.Column('asset_id', sa.Text(), primary_key=True),
        sa.Column('content_sha256', sa.Text()),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('file_name', sa.Text(), nullable=False),
        sa.Column('file_size', sa.Integer()),
        sa.Column('mime_type', sa.Text()),
        sa.Column('status', sa.Text(), default='pending'),
        sa.Column('indexed_at', sa.Text()),
        sa.Column('analyzed_at', sa.Text()),
        sa.Column('enriched_at', sa.Text()),
        sa.Column('error_message', sa.Text()),
    )

    # EXIF data
    op.create_table(
        'enrich_exif',
        sa.Column('asset_id', sa.Text(), primary_key=True),
        sa.Column('raw_json', sa.Text(), nullable=False),
        sa.Column('structured_json', sa.Text()),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )

    # Faces
    op.create_table(
        'enrich_faces',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('asset_id', sa.Text(), nullable=False),
        sa.Column('bbox_json', sa.Text()),
        sa.Column('landmarks_json', sa.Text()),
        sa.Column('landmarks_468_json', sa.Text()),
        sa.Column('landmarks_3d_json', sa.Text()),
        sa.Column('embedding_json', sa.Text(), nullable=False),
        sa.Column('blendshapes_json', sa.Text()),
        sa.Column('expression_vector_json', sa.Text()),
        sa.Column('identity_vector_json', sa.Text()),
        sa.Column('head_pose_json', sa.Text()),
        sa.Column('eye_gaze_json', sa.Text()),
        sa.Column('iris_landmarks_json', sa.Text()),
        sa.Column('det_score', sa.Float()),
        sa.Column('age', sa.Integer()),
        sa.Column('gender', sa.Integer()),
        sa.Column('model_name', sa.Text()),
        sa.Column('detected_at', sa.Text()),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_enrich_faces_asset', 'enrich_faces', ['asset_id'])

    # VLM results
    op.create_table(
        'enrich_vlm',
        sa.Column('asset_id', sa.Text(), primary_key=True),
        sa.Column('model', sa.Text()),
        sa.Column('caption', sa.Text()),
        sa.Column('description', sa.Text()),
        sa.Column('raw_response', sa.Text()),
        sa.Column('processing_ms', sa.Integer()),
        sa.Column('analyzed_at', sa.Text()),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )

    # NER entities
    op.create_table(
        'enrich_entities',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('asset_id', sa.Text(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('entity_type', sa.Text(), nullable=False),
        sa.Column('label', sa.Text()),
        sa.Column('confidence', sa.Float()),
        sa.Column('bbox_json', sa.Text()),
        sa.Column('source', sa.Text(), default='vlm'),
        sa.Column('start_char', sa.Integer()),
        sa.Column('end_char', sa.Integer()),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_enrich_entities_asset', 'enrich_entities', ['asset_id'])
    op.create_index('idx_enrich_entities_type', 'enrich_entities', ['entity_type'])

    # Sidecars
    op.create_table(
        'enrich_sidecars',
        sa.Column('asset_id', sa.Text(), primary_key=True),
        sa.Column('sidecar_path', sa.Text()),
        sa.Column('status', sa.Text()),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )

    # Tags
    op.create_table(
        'enrich_tags',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('asset_id', sa.Text(), nullable=False),
        sa.Column('tag_name', sa.Text(), nullable=False),
        sa.Column('source', sa.Text(), default='manual'),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_enrich_tags_asset', 'enrich_tags', ['asset_id'])
    op.create_index('idx_enrich_tags_name', 'enrich_tags', ['tag_name'])

    # Persons (simple face recognition)
    op.create_table(
        'enrich_persons',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('name', sa.Text(), nullable=False, unique=True),
        sa.Column('embedding_json', sa.Text(), nullable=False),
        sa.Column('created_at', sa.Text()),
    )

    # Face matches
    op.create_table(
        'enrich_face_matches',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('face_id', sa.Text(), nullable=False),
        sa.Column('person_id', sa.Text(), nullable=False),
        sa.Column('similarity', sa.Float(), nullable=False),
        sa.Column('matched_at', sa.Text()),
        sa.ForeignKeyConstraint(['face_id'], ['enrich_faces.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['person_id'], ['enrich_persons.id'], ondelete='CASCADE'),
    )
    op.create_index('idx_enrich_face_matches_face', 'enrich_face_matches', ['face_id'])
    op.create_index('idx_enrich_face_matches_person', 'enrich_face_matches', ['person_id'])

    # Body poses
    op.create_table(
        'enrich_body_poses',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('asset_id', sa.Text(), nullable=False),
        sa.Column('landmark_set', sa.Text(), nullable=False),
        sa.Column('landmarks_2d_json', sa.Text(), nullable=False),
        sa.Column('landmarks_3d_json', sa.Text()),
        sa.Column('bbox_json', sa.Text()),
        sa.Column('det_score', sa.Float()),
        sa.Column('model_name', sa.Text()),
        sa.Column('detected_at', sa.Text()),
        sa.Column('raw_response_json', sa.Text()),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_enrich_body_poses_asset', 'enrich_body_poses', ['asset_id'])
    op.create_index('idx_enrich_body_poses_model', 'enrich_body_poses', ['model_name'])

    # Body measurements
    op.create_table(
        'enrich_body_measurements',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('asset_id', sa.Text()),
        sa.Column('performer_id', sa.Text()),
        sa.Column('measurement_type', sa.Text(), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('unit', sa.Text(), default='cm'),
        sa.Column('confidence', sa.Float(), default=0.0),
        sa.Column('method', sa.Text(), default='landmark_2d'),
        sa.Column('reference_scale', sa.Float()),
        sa.Column('landmarks_used_json', sa.Text()),
        sa.Column('computed_at', sa.Text()),
        sa.Column('notes', sa.Text()),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_enrich_body_measurements_asset', 'enrich_body_measurements', ['asset_id'])
    op.create_index('idx_enrich_body_measurements_performer', 'enrich_body_measurements', ['performer_id'])
    op.create_index('idx_enrich_body_measurements_type', 'enrich_body_measurements', ['measurement_type'])

    # Facial biometrics
    op.create_table(
        'enrich_facial_biometrics',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('performer_id', sa.Text(), nullable=False),
        sa.Column('identity_embedding_json', sa.Text(), nullable=False),
        sa.Column('identity_embedding_std_json', sa.Text()),
        sa.Column('identity_quality_mean', sa.Float(), default=0.0),
        sa.Column('shape_coefficients_json', sa.Text()),
        sa.Column('expression_coefficients_mean_json', sa.Text()),
        sa.Column('expression_coefficients_std_json', sa.Text()),
        sa.Column('canonical_landmarks_68_json', sa.Text()),
        sa.Column('canonical_landmarks_68_cov_json', sa.Text()),
        sa.Column('canonical_landmarks_468_json', sa.Text()),
        sa.Column('facial_measurements_json', sa.Text(), default='{}'),
        sa.Column('skin_tone_lab_json', sa.Text()),
        sa.Column('skin_texture_descriptor_json', sa.Text()),
        sa.Column('source_image_count', sa.Integer(), default=0),
        sa.Column('pose_coverage_json', sa.Text(), default='{}'),
        sa.Column('expression_coverage_json', sa.Text(), default='{}'),
        sa.Column('updated_at', sa.Text()),
        sa.Column('model_versions_json', sa.Text(), default='{}'),
        sa.ForeignKeyConstraint(['performer_id'], ['enrich_performers.id'], ondelete='CASCADE'),
    )
    op.create_index('idx_enrich_facial_biometrics_performer', 'enrich_facial_biometrics', ['performer_id'])

    # Performers (extended)
    op.create_table(
        'enrich_performers',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('name', sa.Text(), nullable=False, unique=True),
        sa.Column('aliases_json', sa.Text(), default='[]'),
        sa.Column('face_embedding_json', sa.Text()),
        sa.Column('face_embedding_quality', sa.Float(), default=0.0),
        sa.Column('measurements_json', sa.Text(), default='{}'),
        sa.Column('measurement_confidence_json', sa.Text(), default='{}'),
        sa.Column('body_type', sa.Text()),
        sa.Column('height_cm', sa.Float()),
        sa.Column('height_confidence', sa.Float(), default=0.0),
        sa.Column('weight_kg_estimate', sa.Float()),
        sa.Column('facial_biometrics_id', sa.Text()),
        sa.Column('asset_count', sa.Integer(), default=0),
        sa.Column('first_seen', sa.Text()),
        sa.Column('last_seen', sa.Text()),
        sa.Column('source_tags_json', sa.Text(), default='[]'),
        sa.Column('notes', sa.Text()),
        sa.Column('created_at', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.Text(), nullable=False),
    )
    op.create_index('idx_enrich_performers_name', 'enrich_performers', ['name'])

    # Performer-asset linkage
    op.create_table(
        'enrich_performer_assets',
        sa.Column('performer_id', sa.Text(), nullable=False),
        sa.Column('asset_id', sa.Text(), nullable=False),
        sa.Column('matched_at', sa.Text(), nullable=False),
        sa.Column('match_method', sa.Text()),
        sa.Column('match_score', sa.Float()),
        sa.PrimaryKeyConstraint('performer_id', 'asset_id'),
        sa.ForeignKeyConstraint(['performer_id'], ['enrich_performers.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_enrich_performer_assets_asset', 'enrich_performer_assets', ['asset_id'])

    # Semantic tags (new)
    op.create_table(
        'enrich_semantic_tags',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('asset_id', sa.Text(), nullable=False),
        sa.Column('namespace', sa.Text(), nullable=False),
        sa.Column('key', sa.Text(), nullable=False),
        sa.Column('value', sa.Text()),
        sa.Column('confidence', sa.Float(), default=1.0),
        sa.Column('source', sa.Text(), nullable=False),
        sa.Column('model_version', sa.Text()),
        sa.Column('created_at', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['asset_id'], ['enrich_assets.asset_id'], ondelete='CASCADE'),
    )
    op.create_index('idx_semantic_tags_asset_ns', 'enrich_semantic_tags', ['asset_id', 'namespace'])
    op.create_index('idx_semantic_tags_ns_key', 'enrich_semantic_tags', ['namespace', 'key'])
    op.create_index('idx_semantic_tags_confidence', 'enrich_semantic_tags', ['confidence'])

    # Tag relations (ontology)
    op.create_table(
        'enrich_tag_relations',
        sa.Column('from_namespace', sa.Text(), nullable=False),
        sa.Column('from_key', sa.Text(), nullable=False),
        sa.Column('to_namespace', sa.Text(), nullable=False),
        sa.Column('to_key', sa.Text(), nullable=False),
        sa.Column('relation_type', sa.Text(), nullable=False),
        sa.Column('weight', sa.Float(), default=1.0),
        sa.PrimaryKeyConstraint('from_namespace', 'from_key', 'to_namespace', 'to_key', 'relation_type'),
    )


def downgrade() -> None:
    op.drop_table('enrich_tag_relations')
    op.drop_table('enrich_semantic_tags')
    op.drop_table('enrich_performer_assets')
    op.drop_table('enrich_performers')
    op.drop_table('enrich_facial_biometrics')
    op.drop_table('enrich_body_measurements')
    op.drop_table('enrich_body_poses')
    op.drop_table('enrich_face_matches')
    op.drop_table('enrich_persons')
    op.drop_table('enrich_tags')
    op.drop_table('enrich_sidecars')
    op.drop_table('enrich_entities')
    op.drop_table('enrich_vlm')
    op.drop_table('enrich_faces')
    op.drop_table('enrich_exif')
    op.drop_table('enrich_assets')