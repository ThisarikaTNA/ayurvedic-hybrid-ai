PRAGMA foreign_keys = ON;

CREATE TRIGGER IF NOT EXISTS trg_conditions_ai_audit
AFTER INSERT ON conditions
BEGIN
    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES ('condition', CAST(NEW.condition_id AS TEXT), 'INSERT',
            json_object('canonical_name', NEW.canonical_name, 'version', NEW.record_version));
END;

CREATE TRIGGER IF NOT EXISTS trg_conditions_au_audit
AFTER UPDATE ON conditions
BEGIN
    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES ('condition', CAST(NEW.condition_id AS TEXT), 'UPDATE',
            json_object('canonical_name', NEW.canonical_name, 'old_version', OLD.record_version,
                        'new_version', NEW.record_version));
END;

CREATE TRIGGER IF NOT EXISTS trg_evidence_sources_ai_audit
AFTER INSERT ON evidence_sources
BEGIN
    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES ('evidence_source', NEW.source_id, 'INSERT',
            json_object('url', NEW.url, 'status', NEW.validation_status,
                        'version', NEW.source_version));
END;

CREATE TRIGGER IF NOT EXISTS trg_knowledge_claims_ai_audit
AFTER INSERT ON knowledge_claims
BEGIN
    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES ('knowledge_claim', NEW.claim_id, 'INSERT',
            json_object('claim_type', NEW.claim_type, 'status', NEW.evidence_status,
                        'version', NEW.claim_version));
END;

CREATE TRIGGER IF NOT EXISTS trg_knowledge_claims_au_stale
AFTER UPDATE OF claim_summary, evidence_status, claim_version, is_active ON knowledge_claims
WHEN OLD.claim_summary IS NOT NEW.claim_summary
  OR OLD.evidence_status IS NOT NEW.evidence_status
  OR OLD.claim_version IS NOT NEW.claim_version
  OR OLD.is_active IS NOT NEW.is_active
BEGIN
    UPDATE recommendations
       SET is_stale = 1, updated_at = CURRENT_TIMESTAMP
     WHERE claim_id = NEW.claim_id;

    UPDATE final_recommendations
       SET is_stale = 1, updated_at = CURRENT_TIMESTAMP
     WHERE final_recommendation_id IN (
         SELECT final_recommendation_id
           FROM final_recommendation_claims
          WHERE claim_id = NEW.claim_id
     );

    INSERT INTO stale_items(item_type, item_id, reason, status)
    SELECT 'recommendation', CAST(recommendation_id AS TEXT),
           'Supporting knowledge claim changed: ' || NEW.claim_id, 'open'
      FROM recommendations
     WHERE claim_id = NEW.claim_id
    ON CONFLICT(item_type, item_id, status) DO UPDATE SET
        reason = excluded.reason, updated_at = CURRENT_TIMESTAMP;

    INSERT INTO stale_items(item_type, item_id, reason, status)
    SELECT 'final_recommendation', CAST(final_recommendation_id AS TEXT),
           'Supporting knowledge claim changed: ' || NEW.claim_id, 'open'
      FROM final_recommendation_claims
     WHERE claim_id = NEW.claim_id
    ON CONFLICT(item_type, item_id, status) DO UPDATE SET
        reason = excluded.reason, updated_at = CURRENT_TIMESTAMP;

    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES ('knowledge_claim', NEW.claim_id, 'UPDATE',
            json_object('old_status', OLD.evidence_status, 'new_status', NEW.evidence_status,
                        'old_version', OLD.claim_version, 'new_version', NEW.claim_version));
END;

CREATE TRIGGER IF NOT EXISTS trg_evidence_sources_au_stale
AFTER UPDATE OF validation_status, source_version ON evidence_sources
WHEN OLD.validation_status IS NOT NEW.validation_status
  OR OLD.source_version IS NOT NEW.source_version
BEGIN
    UPDATE recommendations
       SET is_stale = 1, updated_at = CURRENT_TIMESTAMP
     WHERE claim_id IN (
         SELECT claim_id FROM claim_evidence WHERE source_id = NEW.source_id
     );

    UPDATE final_recommendations
       SET is_stale = 1, updated_at = CURRENT_TIMESTAMP
     WHERE final_recommendation_id IN (
         SELECT frc.final_recommendation_id
           FROM final_recommendation_claims AS frc
           JOIN claim_evidence AS ce ON ce.claim_id = frc.claim_id
          WHERE ce.source_id = NEW.source_id
     );

    INSERT INTO stale_items(item_type, item_id, reason, status)
    SELECT 'recommendation', CAST(r.recommendation_id AS TEXT),
           'Supporting evidence source changed: ' || NEW.source_id, 'open'
      FROM recommendations AS r
      JOIN claim_evidence AS ce ON ce.claim_id = r.claim_id
     WHERE ce.source_id = NEW.source_id
    ON CONFLICT(item_type, item_id, status) DO UPDATE SET
        reason = excluded.reason, updated_at = CURRENT_TIMESTAMP;

    INSERT INTO stale_items(item_type, item_id, reason, status)
    SELECT 'final_recommendation', CAST(frc.final_recommendation_id AS TEXT),
           'Supporting evidence source changed: ' || NEW.source_id, 'open'
      FROM final_recommendation_claims AS frc
      JOIN claim_evidence AS ce ON ce.claim_id = frc.claim_id
     WHERE ce.source_id = NEW.source_id
    ON CONFLICT(item_type, item_id, status) DO UPDATE SET
        reason = excluded.reason, updated_at = CURRENT_TIMESTAMP;

    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES ('evidence_source', NEW.source_id, 'UPDATE',
            json_object('old_status', OLD.validation_status, 'new_status', NEW.validation_status,
                        'old_version', OLD.source_version, 'new_version', NEW.source_version));
END;

CREATE TRIGGER IF NOT EXISTS trg_final_recommendations_ai_audit
AFTER INSERT ON final_recommendations
BEGIN
    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES ('generated_recommendation', CAST(NEW.final_recommendation_id AS TEXT), 'INSERT',
            json_object('generation_status', NEW.generation_status,
                        'version', NEW.recommendation_version));
END;

CREATE TRIGGER IF NOT EXISTS trg_final_recommendations_au_audit
AFTER UPDATE ON final_recommendations
BEGIN
    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES ('generated_recommendation', CAST(NEW.final_recommendation_id AS TEXT), 'UPDATE',
            json_object('old_status', OLD.generation_status,
                        'new_status', NEW.generation_status,
                        'old_stale', OLD.is_stale, 'new_stale', NEW.is_stale,
                        'version', NEW.recommendation_version));
END;
