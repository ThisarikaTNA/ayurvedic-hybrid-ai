BEGIN;

CREATE TRIGGER IF NOT EXISTS trg_phase7_knowledge_rules_ai_audit
AFTER INSERT ON knowledge_rules
BEGIN
    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES (
        'knowledge_rule', NEW.rule_id, 'INSERT',
        json_object(
            'rule_version', NEW.rule_version,
            'lifecycle_status', NEW.lifecycle_status,
            'evidence_status', NEW.evidence_status
        )
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_phase7_knowledge_rules_au_audit
AFTER UPDATE OF rule_version, conditions_json, action_json, lifecycle_status,
                evidence_status, structural_validation_status ON knowledge_rules
BEGIN
    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    VALUES (
        'knowledge_rule', NEW.rule_id, 'UPDATE',
        json_object(
            'old_rule_version', OLD.rule_version,
            'new_rule_version', NEW.rule_version,
            'old_lifecycle_status', OLD.lifecycle_status,
            'new_lifecycle_status', NEW.lifecycle_status,
            'old_evidence_status', OLD.evidence_status,
            'new_evidence_status', NEW.evidence_status
        )
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_phase7_claim_rules_stale
AFTER UPDATE OF claim_summary, evidence_status, claim_version, is_active ON knowledge_claims
WHEN OLD.claim_summary IS NOT NEW.claim_summary
  OR OLD.evidence_status IS NOT NEW.evidence_status
  OR OLD.claim_version IS NOT NEW.claim_version
  OR OLD.is_active IS NOT NEW.is_active
BEGIN
    UPDATE knowledge_rules
       SET is_stale = 1, updated_at = CURRENT_TIMESTAMP
     WHERE rule_id IN (
         SELECT rule_id FROM rule_evidence WHERE claim_id = NEW.claim_id
     );

    INSERT INTO stale_items(item_type, item_id, reason, status)
    SELECT 'knowledge_rule', rule_id,
           'Supporting knowledge claim changed: ' || NEW.claim_id, 'open'
      FROM rule_evidence
     WHERE claim_id = NEW.claim_id
    ON CONFLICT(item_type, item_id, status) DO UPDATE SET
        reason = excluded.reason, updated_at = CURRENT_TIMESTAMP;

    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    SELECT 'knowledge_rule', rule_id, 'STALE',
           json_object('reason', 'supporting_claim_changed', 'claim_id', NEW.claim_id)
      FROM rule_evidence
     WHERE claim_id = NEW.claim_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_phase7_source_rules_stale
AFTER UPDATE OF validation_status, source_version ON evidence_sources
WHEN OLD.validation_status IS NOT NEW.validation_status
  OR OLD.source_version IS NOT NEW.source_version
BEGIN
    UPDATE knowledge_rules
       SET is_stale = 1, updated_at = CURRENT_TIMESTAMP
     WHERE rule_id IN (
         SELECT rule_id FROM rule_evidence WHERE source_id = NEW.source_id
     );

    INSERT INTO stale_items(item_type, item_id, reason, status)
    SELECT 'knowledge_rule', rule_id,
           'Supporting evidence source changed: ' || NEW.source_id, 'open'
      FROM rule_evidence
     WHERE source_id = NEW.source_id
    ON CONFLICT(item_type, item_id, status) DO UPDATE SET
        reason = excluded.reason, updated_at = CURRENT_TIMESTAMP;

    INSERT INTO audit_log(entity_type, entity_id, action, details_json)
    SELECT 'knowledge_rule', rule_id, 'STALE',
           json_object('reason', 'supporting_source_changed', 'source_id', NEW.source_id)
      FROM rule_evidence
     WHERE source_id = NEW.source_id;
END;

-- On an upgrade from an already-seeded Phase 7 database, preserve an audit
-- record equivalent to the INSERT trigger that a clean rebuild will produce.
INSERT INTO audit_log(entity_type, entity_id, action, details_json)
SELECT 'knowledge_rule', kr.rule_id, 'INSERT',
       json_object(
           'rule_version', kr.rule_version,
           'lifecycle_status', kr.lifecycle_status,
           'evidence_status', kr.evidence_status
       )
  FROM knowledge_rules AS kr
 WHERE NOT EXISTS (
       SELECT 1 FROM audit_log AS al
        WHERE al.entity_type = 'knowledge_rule'
          AND al.entity_id = kr.rule_id
          AND al.action = 'INSERT'
 );

COMMIT;
