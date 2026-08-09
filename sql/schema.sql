PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS knowledge_base_versions (
    version_id INTEGER PRIMARY KEY,
    schema_version TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    source_manifest_sha256 TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conditions (
    condition_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    normalized_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    provenance_status TEXT NOT NULL CHECK (provenance_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    record_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS condition_aliases (
    alias_id INTEGER PRIMARY KEY,
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    alias_text TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    alias_type TEXT NOT NULL CHECK (alias_type IN
        ('canonical', 'abbreviation', 'spelling_variant', 'common_name', 'dataset_original')),
    source_profile_id TEXT NOT NULL DEFAULT '',
    provenance_status TEXT NOT NULL CHECK (provenance_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    record_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (condition_id, normalized_alias, source_profile_id)
);

CREATE TABLE IF NOT EXISTS recommendation_categories (
    category_id INTEGER PRIMARY KEY,
    category_code TEXT NOT NULL UNIQUE,
    category_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evidence_sources (
    source_id TEXT PRIMARY KEY,
    publisher TEXT NOT NULL,
    page_title TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    access_date TEXT NOT NULL,
    publication_or_review_date TEXT,
    jurisdiction TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN
        ('official_health', 'peer_reviewed', 'ayurveda_reference', 'dataset', 'expert_review')),
    source_version TEXT NOT NULL,
    validation_status TEXT NOT NULL CHECK (validation_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    reviewer TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_claims (
    claim_id TEXT PRIMARY KEY,
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    claim_type TEXT NOT NULL CHECK (claim_type IN
        ('symptom', 'general_self_care', 'referral_consideration',
         'dataset_dosha_association', 'dataset_symptom_profile',
         'dataset_lifestyle_text', 'dataset_prevention_text',
         'dataset_complication_text', 'contraindication', 'other')),
    claim_summary TEXT NOT NULL,
    original_text TEXT,
    normalized_text TEXT NOT NULL,
    source_profile_id TEXT,
    evidence_status TEXT NOT NULL CHECK (evidence_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    claim_version TEXT NOT NULL,
    safety_relevance TEXT NOT NULL CHECK (safety_relevance IN
        ('none', 'general', 'caution', 'referral', 'urgent', 'emergency')),
    limitations TEXT NOT NULL,
    phase7_eligible INTEGER NOT NULL DEFAULT 0 CHECK (phase7_eligible IN (0, 1)),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS symptoms (
    symptom_id INTEGER PRIMARY KEY,
    symptom_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    provenance_status TEXT NOT NULL CHECK (provenance_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    record_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (normalized_text, provenance_status)
);

CREATE TABLE IF NOT EXISTS condition_symptoms (
    condition_symptom_id INTEGER PRIMARY KEY,
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    symptom_id INTEGER NOT NULL REFERENCES symptoms(symptom_id) ON DELETE RESTRICT,
    claim_id TEXT REFERENCES knowledge_claims(claim_id) ON DELETE SET NULL,
    source_profile_id TEXT NOT NULL DEFAULT '',
    source_text TEXT,
    relationship_status TEXT NOT NULL CHECK (relationship_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    relationship_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (condition_id, symptom_id, source_profile_id, relationship_status)
);

CREATE TABLE IF NOT EXISTS doshas (
    dosha_id INTEGER PRIMARY KEY,
    dosha_name TEXT NOT NULL UNIQUE CHECK (dosha_name IN ('Vata', 'Pitta', 'Kapha')),
    description TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS condition_doshas (
    condition_dosha_id INTEGER PRIMARY KEY,
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    dosha_id INTEGER NOT NULL REFERENCES doshas(dosha_id) ON DELETE RESTRICT,
    claim_id TEXT NOT NULL REFERENCES knowledge_claims(claim_id) ON DELETE CASCADE,
    source_profile_id TEXT NOT NULL,
    original_dosha_text TEXT NOT NULL,
    relationship_status TEXT NOT NULL CHECK (relationship_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    relationship_version TEXT NOT NULL,
    conflict_note TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (condition_id, dosha_id, source_profile_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id INTEGER PRIMARY KEY,
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    category_id INTEGER NOT NULL REFERENCES recommendation_categories(category_id) ON DELETE RESTRICT,
    claim_id TEXT REFERENCES knowledge_claims(claim_id) ON DELETE SET NULL,
    recommendation_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    source_profile_id TEXT NOT NULL DEFAULT '',
    provenance_status TEXT NOT NULL CHECK (provenance_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    recommendation_version TEXT NOT NULL,
    is_generated INTEGER NOT NULL DEFAULT 0 CHECK (is_generated IN (0, 1)),
    is_stale INTEGER NOT NULL DEFAULT 0 CHECK (is_stale IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (condition_id, category_id, normalized_text, source_profile_id, provenance_status)
);

CREATE TABLE IF NOT EXISTS formulations (
    formulation_id INTEGER PRIMARY KEY,
    formulation_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    source_profile_id TEXT NOT NULL DEFAULT '',
    provenance_status TEXT NOT NULL CHECK (provenance_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    formulation_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingredients (
    ingredient_id INTEGER PRIMARY KEY,
    ingredient_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    provenance_status TEXT NOT NULL CHECK (provenance_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    ingredient_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS formulation_ingredients (
    formulation_ingredient_id INTEGER PRIMARY KEY,
    formulation_id INTEGER NOT NULL REFERENCES formulations(formulation_id) ON DELETE CASCADE,
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(ingredient_id) ON DELETE RESTRICT,
    provenance_status TEXT NOT NULL CHECK (provenance_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    relationship_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (formulation_id, ingredient_id)
);

CREATE TABLE IF NOT EXISTS contraindications (
    contraindication_id INTEGER PRIMARY KEY,
    condition_id INTEGER REFERENCES conditions(condition_id) ON DELETE CASCADE,
    recommendation_id INTEGER REFERENCES recommendations(recommendation_id) ON DELETE CASCADE,
    formulation_id INTEGER REFERENCES formulations(formulation_id) ON DELETE CASCADE,
    claim_id TEXT NOT NULL REFERENCES knowledge_claims(claim_id) ON DELETE CASCADE,
    contraindication_summary TEXT NOT NULL,
    provenance_status TEXT NOT NULL CHECK (provenance_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    contraindication_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (condition_id IS NOT NULL OR recommendation_id IS NOT NULL OR formulation_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS safety_claims (
    safety_claim_id INTEGER PRIMARY KEY,
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    claim_id TEXT NOT NULL UNIQUE REFERENCES knowledge_claims(claim_id) ON DELETE CASCADE,
    safety_level TEXT NOT NULL CHECK (safety_level IN
        ('information', 'caution', 'referral', 'urgent', 'emergency')),
    safety_summary TEXT NOT NULL,
    provenance_status TEXT NOT NULL CHECK (provenance_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    safety_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_evidence_id INTEGER PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES knowledge_claims(claim_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES evidence_sources(source_id) ON DELETE RESTRICT,
    source_locator TEXT NOT NULL,
    supports_complete_claim INTEGER NOT NULL CHECK (supports_complete_claim IN (0, 1)),
    validation_status TEXT NOT NULL CHECK (validation_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    reviewer TEXT NOT NULL,
    notes TEXT NOT NULL,
    evidence_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (claim_id, source_id, source_locator)
);

CREATE TABLE IF NOT EXISTS knowledge_rules (
    rule_id TEXT PRIMARY KEY,
    condition_id INTEGER REFERENCES conditions(condition_id) ON DELETE CASCADE,
    rule_version TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    conditions_json TEXT NOT NULL CHECK (json_valid(conditions_json)),
    action_json TEXT NOT NULL CHECK (json_valid(action_json)),
    priority INTEGER NOT NULL CHECK (priority BETWEEN 1 AND 5),
    explanation TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    last_validation_date TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rule_evidence (
    rule_evidence_id INTEGER PRIMARY KEY,
    rule_id TEXT NOT NULL REFERENCES knowledge_rules(rule_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES evidence_sources(source_id) ON DELETE RESTRICT,
    claim_id TEXT REFERENCES knowledge_claims(claim_id) ON DELETE RESTRICT,
    validation_status TEXT NOT NULL CHECK (validation_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (rule_id, source_id, claim_id)
);

CREATE TABLE IF NOT EXISTS rule_validation (
    rule_validation_id INTEGER PRIMARY KEY,
    rule_id TEXT NOT NULL REFERENCES knowledge_rules(rule_id) ON DELETE CASCADE,
    validation_status TEXT NOT NULL CHECK (validation_status IN
        ('dataset_derived', 'reference_checked', 'expert_reviewed', 'draft', 'inactive')),
    validator TEXT NOT NULL,
    validation_date TEXT NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id INTEGER PRIMARY KEY,
    input_reference_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dosha_id INTEGER NOT NULL REFERENCES doshas(dosha_id) ON DELETE RESTRICT,
    uncalibrated_score REAL NOT NULL CHECK (uncalibrated_score BETWEEN 0.0 AND 1.0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    retrieval_result_id INTEGER PRIMARY KEY,
    query_reference_hash TEXT NOT NULL,
    rank INTEGER NOT NULL CHECK (rank > 0),
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id) ON DELETE CASCADE,
    source_profile_id TEXT NOT NULL,
    similarity_score REAL NOT NULL CHECK (similarity_score BETWEEN 0.0 AND 1.0),
    retrieval_method TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (query_reference_hash, rank, retrieval_method)
);

CREATE TABLE IF NOT EXISTS final_recommendations (
    final_recommendation_id INTEGER PRIMARY KEY,
    condition_id INTEGER REFERENCES conditions(condition_id) ON DELETE SET NULL,
    input_reference_hash TEXT NOT NULL,
    recommendation_text TEXT NOT NULL,
    recommendation_version TEXT NOT NULL,
    generation_status TEXT NOT NULL CHECK (generation_status IN
        ('draft', 'generated', 'excluded', 'inactive')),
    is_stale INTEGER NOT NULL DEFAULT 0 CHECK (is_stale IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS final_recommendation_claims (
    final_recommendation_claim_id INTEGER PRIMARY KEY,
    final_recommendation_id INTEGER NOT NULL REFERENCES final_recommendations(final_recommendation_id) ON DELETE CASCADE,
    claim_id TEXT NOT NULL REFERENCES knowledge_claims(claim_id) ON DELETE RESTRICT,
    claim_version_snapshot TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (final_recommendation_id, claim_id)
);

CREATE TABLE IF NOT EXISTS stale_items (
    stale_item_id INTEGER PRIMARY KEY,
    item_type TEXT NOT NULL CHECK (item_type IN
        ('recommendation', 'final_recommendation', 'retrieval_result', 'knowledge_rule')),
    item_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (item_type, item_id, status)
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE', 'STALE')),
    details_json TEXT NOT NULL CHECK (json_valid(details_json)),
    event_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
