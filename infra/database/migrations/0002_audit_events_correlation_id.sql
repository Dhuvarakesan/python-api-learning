ALTER TABLE audit_events
  ADD COLUMN IF NOT EXISTS correlation_id text;

UPDATE audit_events
SET correlation_id = safe_metadata ->> 'correlationId'
WHERE correlation_id IS NULL
  AND safe_metadata ? 'correlationId';

UPDATE audit_events
SET correlation_id = 'legacy-' || id::text
WHERE correlation_id IS NULL;

ALTER TABLE audit_events
  ALTER COLUMN correlation_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS audit_events_correlation_idx
  ON audit_events (correlation_id);
