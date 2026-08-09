BEGIN;

ALTER TABLE knowledge_rules
    ADD COLUMN rule_key TEXT NOT NULL DEFAULT '';
ALTER TABLE knowledge_rules
    ADD COLUMN rule_name TEXT NOT NULL DEFAULT '';
ALTER TABLE knowledge_rules
    ADD COLUMN lifecycle_status TEXT NOT NULL DEFAULT 'draft'
        CHECK (lifecycle_status IN ('draft', 'active', 'inactive'));
ALTER TABLE knowledge_rules
    ADD COLUMN evidence_status TEXT NOT NULL DEFAULT 'dataset_derived'
        CHECK (evidence_status IN ('dataset_derived', 'reference_checked', 'expert_reviewed'));
ALTER TABLE knowledge_rules
    ADD COLUMN structural_validation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (structural_validation_status IN ('pending', 'valid', 'invalid'));
ALTER TABLE knowledge_rules
    ADD COLUMN limitations TEXT NOT NULL DEFAULT '';
ALTER TABLE knowledge_rules
    ADD COLUMN safety_notes TEXT NOT NULL DEFAULT '';
ALTER TABLE knowledge_rules
    ADD COLUMN is_stale INTEGER NOT NULL DEFAULT 0
        CHECK (is_stale IN (0, 1));

CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_rules_key_version
    ON knowledge_rules(rule_key, rule_version);
CREATE INDEX IF NOT EXISTS idx_knowledge_rules_active_priority
    ON knowledge_rules(condition_id, lifecycle_status, is_stale,
                       structural_validation_status, priority, rule_key, rule_version);
CREATE INDEX IF NOT EXISTS idx_knowledge_rules_evidence_status
    ON knowledge_rules(evidence_status);

COMMIT;
