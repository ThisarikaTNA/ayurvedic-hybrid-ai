PRAGMA foreign_keys = ON;

CREATE INDEX IF NOT EXISTS idx_conditions_canonical_name
    ON conditions(canonical_name);
CREATE INDEX IF NOT EXISTS idx_condition_aliases_normalized_alias
    ON condition_aliases(normalized_alias);
CREATE INDEX IF NOT EXISTS idx_condition_aliases_condition_id
    ON condition_aliases(condition_id);
CREATE INDEX IF NOT EXISTS idx_symptoms_normalized_text
    ON symptoms(normalized_text);
CREATE INDEX IF NOT EXISTS idx_condition_symptoms_condition
    ON condition_symptoms(condition_id, symptom_id);
CREATE INDEX IF NOT EXISTS idx_condition_symptoms_symptom
    ON condition_symptoms(symptom_id, condition_id);
CREATE INDEX IF NOT EXISTS idx_condition_symptoms_claim
    ON condition_symptoms(claim_id);
CREATE INDEX IF NOT EXISTS idx_condition_doshas_condition
    ON condition_doshas(condition_id, dosha_id);
CREATE INDEX IF NOT EXISTS idx_condition_doshas_dosha
    ON condition_doshas(dosha_id, condition_id);
CREATE INDEX IF NOT EXISTS idx_condition_doshas_claim
    ON condition_doshas(claim_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_claims_status_type
    ON knowledge_claims(evidence_status, claim_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_claims_condition
    ON knowledge_claims(condition_id);
CREATE INDEX IF NOT EXISTS idx_evidence_sources_url
    ON evidence_sources(url);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_claim
    ON claim_evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_source
    ON claim_evidence(source_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_condition
    ON recommendations(condition_id, category_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_claim
    ON recommendations(claim_id);
CREATE INDEX IF NOT EXISTS idx_formulation_ingredients_formulation
    ON formulation_ingredients(formulation_id);
CREATE INDEX IF NOT EXISTS idx_formulation_ingredients_ingredient
    ON formulation_ingredients(ingredient_id);
CREATE INDEX IF NOT EXISTS idx_contraindications_claim
    ON contraindications(claim_id);
CREATE INDEX IF NOT EXISTS idx_safety_claims_condition
    ON safety_claims(condition_id, safety_level);
CREATE INDEX IF NOT EXISTS idx_knowledge_rules_status_priority
    ON knowledge_rules(status, priority);
CREATE INDEX IF NOT EXISTS idx_rule_evidence_rule
    ON rule_evidence(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_evidence_source
    ON rule_evidence(source_id);
CREATE INDEX IF NOT EXISTS idx_rule_validation_rule
    ON rule_validation(rule_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_dosha
    ON model_predictions(dosha_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_results_condition
    ON retrieval_results(condition_id);
CREATE INDEX IF NOT EXISTS idx_final_recommendations_condition
    ON final_recommendations(condition_id);
CREATE INDEX IF NOT EXISTS idx_final_recommendation_claims_claim
    ON final_recommendation_claims(claim_id);
CREATE INDEX IF NOT EXISTS idx_stale_items_status
    ON stale_items(status, item_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp
    ON audit_log(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_entity
    ON audit_log(entity_type, entity_id);
