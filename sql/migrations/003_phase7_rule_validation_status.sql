BEGIN;

ALTER TABLE rule_validation
    ADD COLUMN structural_validation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (structural_validation_status IN ('pending', 'valid', 'invalid'));
ALTER TABLE rule_validation
    ADD COLUMN validation_type TEXT NOT NULL DEFAULT 'structural_and_evidence_link'
        CHECK (validation_type IN
            ('structural', 'evidence_link', 'structural_and_evidence_link'));

CREATE INDEX IF NOT EXISTS idx_rule_validation_structural_status
    ON rule_validation(structural_validation_status, validation_date, rule_id);

COMMIT;
