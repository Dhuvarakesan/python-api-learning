# Messaging Orchestration

## Purpose

`services/messaging-orchestration` owns COMP-004, the reminder and OTP orchestration service for one-time-code delivery and appointment reminder scheduling through HIPAA-eligible SMS/email vendors or scheduling SaaS capabilities.

## Owned Contracts

- Accepts patient contact method, verification requests, appointment reminder schedules, delivery callbacks, and reminder configuration.
- Produces OTP send requests, reminder send requests, delivery status events, failed-delivery audit evidence, and vendor callback normalization.
- Coordinates with the EHR adapter when appointment state is needed to validate reminder timing.
- Emits ARD-API-006 events for reminder activity and operations failures.

## PHI Handling

- Use minimum necessary message content and templates; avoid appointment details in SMS/email where a lower-PHI alternative is viable.
- Never log OTP codes, full message bodies, secrets, tokens, or full patient contact details.
- Retain operational delivery records for 90 days unless an approved vendor workflow requires otherwise; audit summaries are retained for 6 years.
- Vendors handling PHI must be HIPAA-eligible and covered by BAAs where required.
