# Architecture Requirements Document (ARD)

> Replace all `<PLACEHOLDER>` values. Every material architecture decision or constraint should have a stable ID that feature specifications can reference.

## Document Metadata

- **Product/Application:** Online Patient Scheduling and Reminders
- **Document ID:** ARD-OPSR
- **Version:** 0.1
- **Status:** Draft
- **Architecture Owner:** TBD; see ARC-OQ-001
- **Contributors:** Product Owner TBD, Engineering Lead TBD, Security Reviewer TBD, Clinic Administrator TBD
- **Last Updated:** 2026-11-18
- **Reference PRD:** `PRD.md`

---

## 1. Architecture Summary

Online Patient Scheduling and Reminders will be a custom lightweight web application built as a Next.js full-stack TypeScript application with PostgreSQL persistence, deployed on a TBD cloud provider using HIPAA-eligible managed services under a business associate agreement before production use. The architecture prioritizes HIPAA-aligned PHI handling, small-business operational simplicity, mobile-first patient access, and safe integration with the clinic’s existing scheduling/EHR workflow.

Patients will book and reschedule eligible appointments through passwordless magic link or one-time-code access. Staff will use role-based accounts with MFA to configure appointment eligibility, configure rescheduling rules, and view the day’s schedule. The existing scheduling/EHR system is the appointment system of record when integration is available; the application database acts as an operational cache and workflow store. FHIR/REST integration is preferred, with CSV import/export as the fallback when the EHR cannot support API integration.

## 2. Architecture Drivers

- PRD-FR-001 — Authorized staff must configure which appointment types are online-bookable.
- PRD-FR-002 — Patients must be able to book available slots for eligible appointment types online.
- PRD-FR-003 — New-patient intake and insurance information must be collected before confirming an appointment.
- PRD-FR-004 — Patients must be able to reschedule eligible appointments before the clinic-configured cutoff.
- PRD-FR-005 — Authorized staff must configure the patient self-service rescheduling cutoff.
- PRD-FR-006 — SMS and email reminders must be sent only when channel consent is recorded.
- PRD-FR-007 — Reminder opt-outs and delivery/failure status must be available.
- PRD-FR-008 — Staff must see all appointments for a selected day with patient, provider, time, status, appointment type, and contact/reminder status.
- PRD-FR-009 — Existing scheduling/EHR integration is preferred, with import/export supported when integration is unavailable.
- PRD-NFR-001 — PHI requires HIPAA-aligned controls including BAAs, encryption, RBAC, audit logs, and minimum necessary handling.
- PRD-NFR-002 — Patient booking and rescheduling must be mobile-first, responsive, and WCAG 2.1 AA conformant.
- PRD-NFR-003 — Reminder content must avoid PHI beyond the minimum necessary.
- PRD-NFR-004 — Scheduling actions, consent changes, opt-outs, and staff access to appointment data must be auditable.
- CON-001 — The product must support HIPAA-aligned handling of protected health information.
- CON-002 — The clinic is a small business, so implementation must account for budget sensitivity.
- CON-003 — Existing scheduling/EHR capabilities may constrain integration scope and may require interim import/export.
- CON-004 — Target release date is not set.
- CON-005 — First release is limited to selected appointment types marked as eligible for online booking.
- CON-006 — Reminder vendors that handle PHI must support required HIPAA-aligned agreements and controls.

## 3. Architecture Principles

1. Minimize PHI collection, storage, transmission, and exposure across all components.
2. Prefer managed HIPAA-eligible services and BAA-covered vendors to reduce operational burden for a small business.
3. Reuse established repository and platform patterns where appropriate.
4. Keep presentation, orchestration, domain, integration, and persistence concerns separated.
5. Treat the existing scheduling/EHR system as the appointment system of record when integration is available.
6. Design all patient-facing flows mobile-first and accessible to WCAG 2.1 AA.

## 4. System Context

### 4.1 Actors

- Patient — Books eligible appointments, submits new-patient intake and insurance information, reschedules eligible appointments, provides reminder consent, and opts out of reminders.
- Front-Desk Staff — Reviews the day’s schedule, monitors reminder/contact status, and follows up on failed reminders or schedule changes.
- Clinic Administrator — Configures online-bookable appointment types, rescheduling cutoff rules, staff access, and rollout settings.
- Existing Scheduling/EHR System — Acts as the appointment system of record when integration is available and exchanges schedule data with the application.
- SMS Reminder Vendor — Sends consented SMS reminders and returns delivery/failure events.
- Email Reminder Vendor — Sends consented email reminders and returns delivery/failure events.
- Cloud Platform — Hosts application, database, secrets, logs, metrics, and deployment infrastructure using HIPAA-eligible services under a BAA before production.

### 4.2 External Systems

| ID | System | Purpose | Interface | Owner |
|---|---|---|---|---|
| EXT-001 | Existing scheduling/EHR system | Appointment system of record when integration is available; source/target for appointments, providers, availability, and schedule updates | REST/FHIR preferred; File fallback | Clinic/vendor TBD; see ARC-OQ-005 |
| EXT-002 | SMS reminder vendor | Send appointment reminders and receive delivery/failure status for consented SMS channels | REST/Event webhook | Vendor TBD; see ARC-OQ-007 |
| EXT-003 | Email reminder vendor | Send appointment reminders and receive delivery/failure status for consented email channels | REST/Event webhook | Vendor TBD; see ARC-OQ-007 |
| EXT-004 | Cloud provider | Host HIPAA-aligned production workload, managed database, secrets, observability, and deployment infrastructure | Managed platform APIs | Provider TBD; see ARC-OQ-002 |

## 5. Technology Stack

| Layer | Technology | Version/Constraint | Rationale |
|---|---|---|---|
| Frontend | Next.js with TypeScript | Stable supported version selected during implementation; WCAG 2.1 AA patient flows required | Supports mobile-first patient UI and staff UI in a single lightweight web application |
| Backend | Next.js server-side/API routes with TypeScript | Stable supported Node.js LTS runtime selected during implementation | Keeps architecture simple for a small business while preserving typed API/domain logic |
| Data | PostgreSQL | Managed encrypted PostgreSQL service required in production | Relational model fits appointments, consent, audit logs, configuration, and integration state |
| Hosting | TBD cloud provider | Must use HIPAA-eligible managed services under BAA before production | Cloud provider is unsettled; compliance and operational controls drive final selection |
| Authentication | Staff MFA accounts; patient passwordless magic link or one-time code | Staff RBAC required; patient access must avoid unnecessary account burden | Balances security, usability, and small-clinic operational simplicity |
| Integration | FHIR/REST preferred; CSV import/export fallback | Final EHR capabilities TBD | Supports PRD-FR-009 without blocking launch if API integration is unavailable |
| Messaging | SMS and email vendors | Must support HIPAA-aligned agreements and minimum necessary content controls when handling PHI | Required for PRD-FR-006 and PRD-FR-007 |
| Infrastructure | Infrastructure as Code | Tool TBD; required before production | Enables repeatable environments, change review, and rollback discipline |

## 6. Logical Architecture

### 6.1 Components

#### COMP-001 — Patient Web Experience

- **Responsibility:** Provide mobile-first booking, new-patient intake, insurance capture, rescheduling, consent, and opt-out flows.
- **Inputs:** Appointment type selections, patient identifiers, intake fields, insurance fields, selected slots, reminder consent, opt-out requests, passwordless access tokens.
- **Outputs:** Booking requests, rescheduling requests, consent records, opt-out records, patient-facing confirmations, validation errors.
- **Dependencies:** COMP-002, COMP-003, COMP-004, COMP-006, COMP-008.
- **Repository Area:** `app/patient` or TBD by implementation repository structure.

#### COMP-002 — Staff and Admin Web Experience

- **Responsibility:** Provide staff day schedule view, appointment type eligibility configuration, rescheduling cutoff configuration, reminder status visibility, and administrative controls.
- **Inputs:** Staff credentials, MFA challenge responses, schedule date filters, configuration changes, audit search filters.
- **Outputs:** Day schedule view, configuration updates, reminder/contact status display, audit retrieval results.
- **Dependencies:** COMP-003, COMP-004, COMP-005, COMP-006, COMP-008.
- **Repository Area:** `app/staff` or TBD by implementation repository structure.

#### COMP-003 — Scheduling Domain Service

- **Responsibility:** Enforce appointment eligibility, availability checks, booking rules, rescheduling cutoff rules, and schedule state transitions.
- **Inputs:** Booking commands, rescheduling commands, appointment type configuration, availability data, patient context.
- **Outputs:** Confirmed booking records, rescheduled appointment records, rejected booking/rescheduling attempts, domain events for reminders and audit logging.
- **Dependencies:** COMP-004, COMP-005, COMP-006, COMP-008.
- **Repository Area:** `src/domain/scheduling` or TBD by implementation repository structure.

#### COMP-004 — Persistence Layer

- **Responsibility:** Store operational cache data, configuration, patient-provided workflow data, consent state, reminder status, integration state, and audit records.
- **Inputs:** Domain entities, configuration updates, consent changes, reminder delivery events, integration sync metadata, audit events.
- **Outputs:** Retrieved appointment cache, configuration records, consent state, reminder status, audit records, integration state.
- **Dependencies:** Managed PostgreSQL service.
- **Repository Area:** `src/persistence` or TBD by implementation repository structure.

#### COMP-005 — EHR Integration Adapter

- **Responsibility:** Synchronize appointments, provider availability, appointment types, and schedule updates with the existing scheduling/EHR workflow when available; support CSV import/export fallback.
- **Inputs:** FHIR/REST API responses, CSV imports, booking updates, rescheduling updates, export requests.
- **Outputs:** EHR API calls, CSV exports, normalized schedule data, sync success/failure events.
- **Dependencies:** EXT-001, COMP-003, COMP-004, COMP-008.
- **Repository Area:** `src/integrations/ehr` or TBD by implementation repository structure.

#### COMP-006 — Reminder Service

- **Responsibility:** Schedule SMS and email reminders, enforce consent and opt-out rules, submit reminder messages, and process delivery/failure callbacks.
- **Inputs:** Appointment events, consent records, opt-out records, reminder schedules, vendor callbacks.
- **Outputs:** SMS/email send requests, reminder status records, failed reminder alerts, audit events.
- **Dependencies:** EXT-002, EXT-003, COMP-004, COMP-008.
- **Repository Area:** `src/domain/reminders` or TBD by implementation repository structure.

#### COMP-007 — Authentication and Authorization Service

- **Responsibility:** Authenticate staff with MFA, authorize staff actions using roles, issue and validate patient passwordless links or one-time codes, and enforce least privilege.
- **Inputs:** Staff login requests, MFA verification, patient link/code requests, session tokens, role assignments.
- **Outputs:** Authenticated sessions, authorization decisions, access-denied responses, security audit events.
- **Dependencies:** COMP-004, cloud identity/secrets services.
- **Repository Area:** `src/auth` or TBD by implementation repository structure.

#### COMP-008 — Audit and Observability Service

- **Responsibility:** Record audit events, structured logs, operational metrics, error events, and alerts while preventing PHI leakage into telemetry.
- **Inputs:** Scheduling actions, staff access events, consent changes, opt-outs, integration failures, reminder failures, application errors.
- **Outputs:** Audit records, structured logs, metrics, alerts, production readiness evidence.
- **Dependencies:** COMP-004, cloud logging/metrics/alerting services.
- **Repository Area:** `src/observability` or TBD by implementation repository structure.

### 6.2 Data Flow

1. Patient requests booking or rescheduling access through COMP-001, and COMP-007 validates a passwordless magic link or one-time code.
2. COMP-001 sends booking or rescheduling data to COMP-003.
3. COMP-003 validates appointment type eligibility, availability, new-patient intake requirements, insurance requirements, and rescheduling cutoff rules.
4. COMP-003 retrieves or updates operational schedule data through COMP-004 and synchronizes appointment changes through COMP-005 when EHR integration is available.
5. COMP-005 exchanges appointment, provider, availability, and schedule update data with EXT-001 through FHIR/REST when available or CSV import/export as a fallback.
6. COMP-003 emits booking and rescheduling events to COMP-006 and COMP-008.
7. COMP-006 checks consent and opt-out state in COMP-004 before sending SMS through EXT-002 or email through EXT-003.
8. EXT-002 and EXT-003 return delivery/failure callbacks to COMP-006, which records status in COMP-004 and alerts through COMP-008 for failures.
9. Staff authenticate through COMP-007 and view the day’s schedule in COMP-002, which reads appointment, reminder, and contact status from COMP-003 and COMP-004.
10. COMP-008 records audit events for staff access, booking, rescheduling, consent, opt-out, integration, and reminder status changes.

## 7. Architecture Decisions

### ARD-DEC-001 — Build a Custom Lightweight Web Application

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-003, PRD-FR-004, PRD-FR-005, PRD-FR-008, PRD-NFR-001, PRD-NFR-002

**Decision**

The first release will use a custom lightweight web application built on managed HIPAA-eligible cloud services and BAA-covered vendors.

**Rationale**

The PRD requires a combination of patient booking, rescheduling, staff schedule visibility, clinic-configurable rules, reminder consent, and EHR/import-export flexibility. A custom lightweight application supports these specific workflows while keeping scope constrained for a small business.

**Consequences**

- The product can tailor eligibility, rescheduling, reminders, and schedule views to the clinic workflow.
- The engineering team must own secure software delivery, production operations, and HIPAA-aligned controls.
- Vendor and cloud selections must be validated for BAA coverage before production.

**Alternatives Considered**

- Scheduling SaaS product — Not selected because fit with EHR fallback, custom staff view, and PHI/audit requirements is unknown.
- Existing EHR portal extension — Not selected because existing EHR capabilities are unknown and may constrain launch.
- Hybrid SaaS plus custom integration layer — Not selected for first release because it adds integration and vendor complexity.

### ARD-DEC-002 — Use Next.js TypeScript with PostgreSQL

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-004, PRD-FR-008, PRD-NFR-002, PRD-NFR-004

**Decision**

The application will use Next.js as a full-stack TypeScript web application with PostgreSQL for persistence.

**Rationale**

Next.js and TypeScript support a mobile-first web experience and typed server-side workflow logic in one lightweight stack. PostgreSQL supports transactional scheduling workflows, consent state, configuration, integration state, and audit records.

**Consequences**

- A single TypeScript stack reduces implementation complexity.
- Database schema design must protect PHI and enforce retention and audit requirements.
- The team must select stable supported framework and runtime versions before implementation.

**Alternatives Considered**

- React frontend with separate Node.js backend — Not selected because it increases service boundaries for a small first release.
- Python/Django server-rendered application — Not selected because the approved stack is TypeScript/Next.js.
- Stack TBD — Not selected because the architecture interview settled the concrete stack.

### ARD-DEC-003 — Treat Existing Scheduling/EHR as System of Record When Integrated

- **Status:** Accepted
- **Related Requirements:** PRD-FR-002, PRD-FR-004, PRD-FR-008, PRD-FR-009, PRD-NFR-004

**Decision**

When integration is available, the existing scheduling/EHR system will be the appointment system of record, and the application database will store operational cache, workflow, reminder, consent, integration, and audit data. When integration is unavailable, CSV import/export will provide an interim workflow.

**Rationale**

This reduces duplicate schedule management and aligns with PRD-FR-009 while preserving a launch path when EHR integration is unavailable or costly.

**Consequences**

- Availability and conflict checks must account for synchronization timing and EHR capabilities.
- Import/export fallback may create operational reconciliation work for staff.
- The final integration design depends on the specific EHR system and available APIs.

**Alternatives Considered**

- Application database as the appointment system of record — Not selected because it risks duplicating the EHR workflow.
- No local appointment persistence — Not selected because reminders, schedule views, audit logs, and fallback workflows need operational state.
- Manual staff reconciliation only — Not selected because it undermines online booking efficiency.

### ARD-DEC-004 — Use Passwordless Patient Access and Staff MFA/RBAC

- **Status:** Accepted
- **Related Requirements:** PRD-FR-002, PRD-FR-004, PRD-FR-005, PRD-FR-008, PRD-NFR-001, PRD-NFR-004

**Decision**

Staff will authenticate with role-based accounts and MFA. Patients will access booking and rescheduling through passwordless magic links or one-time codes.

**Rationale**

Staff actions involve PHI and administrative controls, requiring stronger identity and authorization. Patient passwordless access reduces friction while providing controlled access to booking and rescheduling flows.

**Consequences**

- Patient access tokens and one-time codes must be time-limited, single-use where appropriate, and audited.
- Staff roles must be defined before production use.
- Authentication events and authorization failures must feed security audit logs.

**Alternatives Considered**

- Username/password for all users — Not selected because patient account creation adds friction.
- Clinic SSO for staff and full patient portal accounts — Not selected because SSO and portal capabilities are not settled.
- Booking reference only with no patient authentication — Not selected because it is insufficient for PHI-sensitive rescheduling workflows.

### ARD-DEC-005 — Prefer FHIR/REST EHR Integration with CSV Fallback

- **Status:** Accepted
- **Related Requirements:** PRD-FR-009, PRD-NFR-001, PRD-NFR-004

**Decision**

The application will prefer FHIR/REST integration with the existing scheduling/EHR system and provide CSV import/export fallback when API integration is unavailable.

**Rationale**

FHIR/REST is the preferred healthcare integration model when supported, while CSV fallback preserves a launch path aligned to PRD-FR-009.

**Consequences**

- The EHR adapter must normalize data across API and file workflows.
- CSV fallback requires strict access controls, encryption, retention, and audit handling.
- Contract tests are required for API integration and file schema validation is required for CSV workflows.

**Alternatives Considered**

- Vendor-specific API first — Not selected because the EHR vendor is unknown.
- File-only first release — Not selected because it may create unnecessary manual work when APIs are available.
- Manual reconciliation only — Not selected because it does not satisfy the preferred integration approach.

### ARD-DEC-006 — Require Automated CI/CD, IaC, Security Scanning, SBOM, and Rollback

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001, PRD-NFR-004

**Decision**

Production readiness requires automated build, test, lint, security scan, SBOM generation, infrastructure as code for environments, and documented rollback.

**Rationale**

HIPAA-aligned operations require repeatability, change control evidence, vulnerability management, and recoverability without imposing unnecessary manual burden on a small business.

**Consequences**

- Critical and High security findings block release.
- Infrastructure changes must be versioned and reviewed.
- Deployment automation and rollback procedures must exist before production.

**Alternatives Considered**

- Manual deployment with checklist — Not selected because it weakens repeatability and audit evidence.
- Non-production automation only — Not selected because production changes need the strongest controls.
- CI/CD TBD — Not selected because deployment posture was settled in the architecture interview.

## 8. API and Integration Architecture

### ARD-API-001 — Patient Booking API

- **Type:** REST
- **Operation/Topic:** `POST /api/patient/bookings`
- **Provider:** COMP-003
- **Consumers:** COMP-001
- **Authentication:** Patient passwordless magic link or one-time code validated by COMP-007
- **Versioning:** URL-based API versioning when public/external contracts are introduced; internal initial route version is controlled by release version

**Request/Input Contract**

Includes patient context, existing/new patient indicator, appointment type ID, selected slot ID, contact details, reminder consent, and required new-patient intake and insurance fields when applicable. Exact intake and insurance fields are TBD; see ARC-OQ-006.

**Response/Output Contract**

Returns booking confirmation status, appointment identifier, appointment time, provider, appointment type, reminder consent status, and user-safe errors when booking cannot be completed.

**Error Handling**

Returns validation errors for missing required fields, unavailable slots, ineligible appointment types, expired patient access tokens, and integration conflicts. Booking commands must use idempotency keys to prevent duplicate appointment creation on retries.

### ARD-API-002 — Patient Rescheduling API

- **Type:** REST
- **Operation/Topic:** `POST /api/patient/appointments/{appointmentId}/reschedule`
- **Provider:** COMP-003
- **Consumers:** COMP-001
- **Authentication:** Patient passwordless magic link or one-time code validated by COMP-007
- **Versioning:** URL-based API versioning when public/external contracts are introduced; internal initial route version is controlled by release version

**Request/Input Contract**

Includes appointment ID, patient access token context, target slot ID, and idempotency key.

**Response/Output Contract**

Returns rescheduling confirmation status, updated appointment time, provider, appointment type, and user-safe errors when rescheduling cannot be completed.

**Error Handling**

Returns errors for expired access, appointment not eligible for rescheduling, cutoff violation, unavailable slot, EHR sync conflict, and duplicate command. Rescheduling must be idempotent and audited.

### ARD-API-003 — Staff Schedule API

- **Type:** REST
- **Operation/Topic:** `GET /api/staff/schedule?date={date}`
- **Provider:** COMP-003
- **Consumers:** COMP-002
- **Authentication:** Staff authenticated session with MFA and RBAC authorization through COMP-007
- **Versioning:** Internal route version controlled by release version

**Request/Input Contract**

Includes selected schedule date and staff authorization context.

**Response/Output Contract**

Returns all appointments for the selected day with patient, provider, time, status, appointment type, and contact/reminder status.

**Error Handling**

Returns access-denied errors for unauthorized roles and user-safe availability errors when schedule data cannot be loaded. Staff schedule access must be audited under PRD-NFR-004.

### ARD-API-004 — Admin Configuration API

- **Type:** REST
- **Operation/Topic:** `PUT /api/staff/config/appointment-types/{appointmentTypeId}` and `PUT /api/staff/config/rescheduling-cutoff`
- **Provider:** COMP-003
- **Consumers:** COMP-002
- **Authentication:** Staff authenticated session with MFA and administrator/authorized-staff RBAC through COMP-007
- **Versioning:** Internal route version controlled by release version

**Request/Input Contract**

Includes appointment type eligibility flag or rescheduling cutoff rule, staff authorization context, and change reason where required by clinic policy.

**Response/Output Contract**

Returns updated configuration state and effective rule metadata.

**Error Handling**

Returns validation errors for invalid rules and access-denied errors for unauthorized staff. Configuration changes must be audited.

### ARD-API-005 — EHR Scheduling Integration

- **Type:** REST | File
- **Operation/Topic:** `FHIR/REST scheduling endpoints` or `CSV schedule import/export`
- **Provider:** EXT-001 and COMP-005
- **Consumers:** COMP-005 and EXT-001
- **Authentication:** EHR-supported secure API authentication or restricted encrypted file exchange; exact method TBD by EHR capability
- **Versioning:** FHIR version or vendor API version when available; CSV schema version required for file fallback

**Request/Input Contract**

For API integration, normalized requests include appointment types, provider availability, appointment create/update, and schedule retrieval data. For file fallback, CSV schemas include appointments, providers, availability, and update/export records.

**Response/Output Contract**

For API integration, normalized responses include success/failure, external appointment IDs, conflict state, and updated schedule details. For file fallback, output includes accepted/rejected import rows and export manifests.

**Error Handling**

Integration failures create retryable sync events where safe, staff-visible alerts when reconciliation is required, and audit records. Duplicate appointment creation must be prevented through external ID mapping and idempotency keys where supported.

### ARD-API-006 — Reminder Vendor APIs and Webhooks

- **Type:** REST | Event
- **Operation/Topic:** `POST /vendor/sms`, `POST /vendor/email`, `POST /api/webhooks/reminders/{vendor}`
- **Provider:** COMP-006, EXT-002, EXT-003
- **Consumers:** EXT-002, EXT-003, COMP-006
- **Authentication:** Vendor API credentials stored in secrets manager; signed webhook validation where supported
- **Versioning:** Vendor API version; internal webhook version controlled by route and release version

**Request/Input Contract**

Send requests include destination channel, minimum necessary reminder content, template identifier, appointment reminder timing metadata, and vendor correlation ID. Webhooks include correlation ID, delivery status, failure reason, and timestamp.

**Response/Output Contract**

Send responses include accepted/rejected status and vendor message ID. Webhook processing updates reminder status as delivered, failed, opted out, or not sent due to consent/opt-out state.

**Error Handling**

Failed sends and failed callbacks are retried where safe. Persistent failures generate alerts and staff-visible reminder status. Reminder processing must not send through a channel after opt-out or without consent.

## 9. Data Architecture

### DATA-001 — Appointment

- **Purpose:** Represent scheduled visits, booking state, rescheduling changes, and staff schedule view entries.
- **System of Record:** Existing scheduling/EHR system when integrated; application database as operational cache and interim workflow store when using CSV fallback.
- **Classification:** Restricted
- **Retention:** Operational PHI retained only as long as needed for scheduling workflow; exact duration per clinic HIPAA policy is TBD; see ARC-OQ-009.
- **Key Relationships:** Patient, provider, appointment type, reminder status, audit event, EHR external ID.

### DATA-002 — Patient Scheduling Profile

- **Purpose:** Store minimum necessary patient identity/contact information needed for booking, rescheduling, reminders, and staff schedule view.
- **System of Record:** Existing scheduling/EHR system when integrated for established patient records; application database for workflow-submitted scheduling profile data until synchronized or exported.
- **Classification:** Restricted
- **Retention:** Operational PHI retained only as long as needed for scheduling workflow; exact duration per clinic HIPAA policy is TBD; see ARC-OQ-009.
- **Key Relationships:** Appointment, reminder consent, intake submission, insurance submission, audit event.

### DATA-003 — New-Patient Intake and Insurance Submission

- **Purpose:** Capture required intake and insurance information before confirming new-patient appointments.
- **System of Record:** Existing scheduling/EHR system after successful synchronization/import; application database as workflow store until transfer or retention expiry.
- **Classification:** Restricted
- **Retention:** Operational PHI retained only as long as needed for scheduling workflow; exact duration per clinic HIPAA policy is TBD; see ARC-OQ-009.
- **Key Relationships:** Patient scheduling profile, appointment, audit event, EHR export/sync record.

### DATA-004 — Reminder Consent and Opt-Out

- **Purpose:** Track SMS/email consent, opt-out state, and channel eligibility for reminders.
- **System of Record:** Application database unless the EHR supports consent synchronization; final synchronization behavior TBD by EHR capability.
- **Classification:** Restricted
- **Retention:** Retained per clinic consent and HIPAA policy; exact duration TBD; see ARC-OQ-009.
- **Key Relationships:** Patient scheduling profile, reminder event, audit event.

### DATA-005 — Reminder Event and Delivery Status

- **Purpose:** Track reminder scheduling, send attempts, vendor message IDs, delivery/failure status, and not-sent reasons.
- **System of Record:** Application database.
- **Classification:** Restricted
- **Retention:** Retained per clinic HIPAA policy; exact duration TBD; see ARC-OQ-009.
- **Key Relationships:** Appointment, patient scheduling profile, consent record, vendor callback, audit event.

### DATA-006 — Appointment Type and Scheduling Configuration

- **Purpose:** Store online-bookable eligibility flags and rescheduling cutoff rules.
- **System of Record:** Application database for online scheduling configuration; appointment type source may be EHR when integrated.
- **Classification:** Internal
- **Retention:** Retained while configuration is active and per change/audit retention policy after modification; exact duration TBD; see ARC-OQ-009.
- **Key Relationships:** Appointment type, appointment, staff user, audit event.

### DATA-007 — Staff User and Role

- **Purpose:** Manage staff authentication, MFA enrollment state, role assignments, and authorization decisions.
- **System of Record:** Application identity store or cloud identity service; final identity provider TBD.
- **Classification:** Confidential
- **Retention:** Retained while staff account is active and per clinic access/audit policy after deactivation; exact duration TBD; see ARC-OQ-009.
- **Key Relationships:** Audit event, configuration change, schedule access.

### DATA-008 — Audit Event

- **Purpose:** Record booking, rescheduling, schedule-view access, consent changes, opt-outs, configuration changes, authentication/security events, and integration events.
- **System of Record:** Application audit store in PostgreSQL and/or cloud audit logging service.
- **Classification:** Confidential; Restricted when event metadata contains PHI.
- **Retention:** Retained per clinic HIPAA policy; exact duration TBD; see ARC-OQ-009.
- **Key Relationships:** Staff user, patient scheduling profile, appointment, consent record, integration event.

### Data Migration

- **Required:** TBD; see ARC-OQ-010.
- **Approach:** If EHR integration is available, initial synchronization imports appointment types, providers, availability, and current schedule data required for launch. If EHR integration is unavailable, staff use validated CSV import/export schemas for the pilot. Historical data migration is not required unless the clinic decides existing operational schedule data must be preloaded beyond what is needed for launch.

## 10. Security Architecture

### ARD-SEC-001 — HIPAA-Aligned PHI Protection

- **Related Requirements:** PRD-NFR-001, PRD-NFR-003, PRD-NFR-004

The implementation must:

- Use BAA-covered vendors and HIPAA-eligible managed services for any production component that stores, processes, transmits, or can access PHI.
- Encrypt PHI in transit and at rest in production.
- Enforce staff RBAC with MFA and least-privilege access.
- Enforce patient passwordless access controls for booking and rescheduling.
- Record audit events for PHI access, booking, rescheduling, consent, opt-out, configuration, and integration actions.
- Apply minimum necessary PHI handling in data models, APIs, logs, reminders, and staff displays.
- Restrict reminder content to minimum necessary appointment information and avoid diagnosis, procedure details, or clinical notes.
- Store secrets only in an approved secrets manager and rotate credentials when compromised or per clinic policy.
- Generate an SBOM for each production release and run dependency, container, and IaC scanning in CI.

The implementation must not:

- Store PHI in plaintext at rest.
- Log PHI in application logs, metrics, traces, error tracking, CI logs, or deployment logs.
- Send reminders through a channel without consent or after opt-out.
- Include diagnosis, procedure details, or clinical notes in SMS or email reminders.
- Use PHI-handling vendors in production without required BAA coverage.
- Allow unauthorized staff to configure appointment eligibility or rescheduling cutoff rules.

### Security Controls

- **Authentication:** Staff authenticate with role-based accounts and MFA; patients use passwordless magic links or one-time codes for booking/rescheduling access.
- **Authorization:** RBAC enforces patient, front-desk staff, clinic administrator, and technical support access boundaries; least privilege is required for PHI and configuration actions.
- **Secrets Management:** Production secrets must be stored in the selected cloud/provider secrets manager; secrets must not be committed to source control or exposed in logs.
- **Encryption:** TLS is required for data in transit; managed encryption is required for PostgreSQL, backups, object/file storage, logs containing audit data, and integration file exchange at rest.
- **Audit Logging:** Audit events must capture actor, timestamp, action, target/resource, outcome, and correlation ID for scheduling actions, staff access, consent changes, opt-outs, configuration changes, and integration events.
- **Dependency/CVE Scanning:** Dependency, container, and IaC scans run in CI; Critical and High findings block release unless formally risk-accepted by the Security Reviewer.
- **SBOM:** Required for each production release.

## 11. Reliability, Performance, and Scalability

### ARD-OPS-001 — Availability Target

- **Target:** TBD; owner Technical Owner; monthly uptime target must be defined before production readiness approval; see ARC-OQ-011.
- **Validation Method:** Production readiness review must include uptime target, monitoring evidence, failure-mode review, and alert configuration.

### ARD-OPS-002 — Patient Booking Performance Target

- **Target:** TBD; owner Technical Owner; p95 page/API response target for patient booking and rescheduling flows must be defined before production readiness approval; see ARC-OQ-011.
- **Validation Method:** Performance tests must exercise booking and rescheduling flows with representative launch traffic and report p95 response times.

### ARD-OPS-003 — Reminder Processing Target

- **Target:** TBD; owner Technical Owner; maximum reminder job delay from scheduled send time must be defined before production readiness approval; see ARC-OQ-011.
- **Validation Method:** Scheduled job tests and production monitoring must measure reminder enqueue/send latency and alert on missed thresholds.

### ARD-OPS-004 — EHR Synchronization Target

- **Target:** TBD; owner Technical Owner; synchronization freshness and retry targets must be defined after EHR capabilities are known; see ARC-OQ-005 and ARC-OQ-011.
- **Validation Method:** Integration tests and operational dashboards must measure sync success rate, retry outcomes, stale cache age, and reconciliation events.

### ARD-OPS-005 — Scalability Constraint for Pilot Rollout

- **Target:** Initial scale target is TBD because appointment volume and launch KPIs are not quantified; see ARC-OQ-012.
- **Validation Method:** Load test scenarios must be based on pilot appointment volume assumptions approved before production readiness.

## 12. Observability

- **Logging:** Use structured application logs with correlation IDs and no PHI. Log security-relevant and operational events needed to troubleshoot booking, rescheduling, reminder, and integration workflows.
- **Metrics:** Track booking attempts/success/failure, rescheduling attempts/success/failure, reminder sends/delivery/failures/not-sent reasons, EHR sync success/failure, API latency, job latency, error rates, and staff schedule view usage.
- **Tracing:** Use request correlation across patient/staff UI, API handlers, domain services, integration adapter, reminder service, and persistence. Full distributed tracing tool selection is TBD with cloud provider choice.
- **Alerting:** Alert on failed sync jobs, persistent reminder failures, elevated API errors, job backlog, security scan failures in CI, and authentication/security anomalies where supported.
- **Sensitive Data Restrictions:** PHI must not be written to logs, metrics, traces, error tracking, deployment logs, or CI output. Audit logs may reference PHI-linked resources by identifiers and must be access-controlled.

## 13. Deployment Architecture

- **Environments:** Development, test, staging, and production are required before production launch; exact environment naming and cloud accounts/projects are TBD with provider selection.
- **Deployment Model:** Web application deployed to managed compute/serverless/container hosting on a HIPAA-eligible cloud service under BAA; exact model TBD by cloud provider selection.
- **CI/CD:** Automated build, test, lint, security scan, SBOM generation, and deployment pipeline required before production.
- **Configuration Management:** Environment-specific configuration through IaC and secrets manager; no secrets in source control.
- **Rollback:** Documented rollback required for application releases, database migrations, configuration changes, and infrastructure changes before production.
- **Infrastructure as Code:** Required; specific tool TBD with cloud provider selection; see ARC-OQ-002.

## 14. Testing Architecture

Required test levels:

- Unit tests
- Integration tests
- Contract tests
- End-to-end tests
- Security tests
- Performance tests, where applicable

Minimum quality expectations:

- **Coverage Threshold:** TBD before production readiness approval; owner Engineering Lead; see ARC-OQ-013.
- **Blocking Security Severity:** Critical and High dependency, container, application, and IaC security findings block release unless formally risk-accepted by the Security Reviewer.
- **Required Build/Lint Checks:** TypeScript type check, unit tests, integration tests, contract tests for EHR/reminder interfaces, end-to-end tests for booking/rescheduling/staff schedule flows, WCAG 2.1 AA accessibility checks for patient flows, dependency/container/IaC scans, SBOM generation, and production build validation.

## 15. Architecture Constraints

- ARC-CON-001 — The system must implement HIPAA-aligned PHI protections required by PRD-NFR-001 and project compliance requirements.
- ARC-CON-002 — Production PHI-handling vendors and managed services must have BAA coverage before production use.
- ARC-CON-003 — The clinic is a small business; the architecture must prefer managed services and avoid unnecessary operational complexity.
- ARC-CON-004 — The first release must support only appointment types marked as eligible for online booking.
- ARC-CON-005 — Existing scheduling/EHR capabilities determine whether launch uses FHIR/REST integration or CSV import/export fallback.
- ARC-CON-006 — The appointment system of record is the existing scheduling/EHR system when integration is available; otherwise the application supports interim operational workflow and export/import.
- ARC-CON-007 — Reminder vendors must support consent enforcement, opt-out handling, delivery/failure status, and HIPAA-aligned agreements when handling PHI.
- ARC-CON-008 — Patient-facing booking and rescheduling flows must be mobile-first and WCAG 2.1 AA conformant.
- ARC-CON-009 — The target release date is not set and must not be assumed by the architecture.
- ARC-CON-010 — Reminder content must use minimum necessary PHI and must not include diagnosis, procedure details, or clinical notes.
- ARC-CON-011 — CI/CD, IaC, rollback, scanning, and SBOM generation are required before production.

## 16. Risks and Trade-offs

| ID | Risk/Trade-off | Impact | Mitigation |
|---|---|---|---|
| ARC-RISK-001 | Existing scheduling/EHR integration may be unavailable, costly, or limited | Could delay launch or require staff reconciliation | Use FHIR/REST when available and CSV import/export fallback when unavailable; document final launch approach |
| ARC-RISK-002 | Cloud provider and IaC tool are TBD | Deployment design and control implementation cannot be finalized | Require HIPAA-eligible services and BAA before production; close ARC-OQ-002 before build-out |
| ARC-RISK-003 | PHI retention durations are TBD | Data lifecycle controls may be incomplete | Define clinic HIPAA retention policy before production and enforce retention jobs/configuration |
| ARC-RISK-004 | Reliability and performance targets are TBD | Production readiness cannot be objectively assessed | Define measurable uptime, latency, reminder delay, sync freshness, and scale targets before production readiness approval |
| ARC-RISK-005 | CSV fallback introduces manual handling of PHI | File exchange may increase privacy and operational risk | Encrypt files, restrict access, validate schemas, audit imports/exports, and prefer API integration when available |
| ARC-RISK-006 | Reminder vendor BAA or webhook support may be insufficient | SMS/email reminder launch could be delayed or delivery status incomplete | Select vendors early; require BAA, consent/opt-out support, delivery status, and webhook security |
| ARC-RISK-007 | Custom application increases engineering ownership | The clinic must maintain secure delivery, monitoring, and operations | Use managed services, automated CI/CD, security scans, SBOM, and documented rollback |
| ARC-RISK-008 | Minimum necessary reminder content may reduce message specificity | Patients may need to click through for details rather than seeing details in SMS/email | Use privacy-conscious templates and secure links for additional details when needed |

## 17. Open Architecture Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| ARC-OQ-001 | Who is the named Architecture Owner for this ARD? | Engineering Lead | Open | TBD |
| ARC-OQ-002 | Which cloud provider, managed hosting model, and IaC tool will be used for HIPAA-aligned production deployment? | Technical Owner | Open | TBD |
| ARC-OQ-003 | Which stable Next.js, TypeScript, Node.js, and PostgreSQL versions will be selected for implementation? | Engineering Lead | Open | TBD |
| ARC-OQ-004 | Which identity provider or authentication implementation will support staff MFA and patient passwordless access? | Technical Owner | Open | TBD |
| ARC-OQ-005 | Which existing scheduling/EHR system is used, and which FHIR/REST or file integration capabilities does it provide? | Technical Owner | Open | TBD |
| ARC-OQ-006 | Which exact new-patient intake and insurance fields are required before confirmation? | Clinic Administrator | Open | TBD |
| ARC-OQ-007 | Which SMS and email vendors will be used, and do they provide required BAAs, delivery/failure status, opt-out support, and webhook security? | Technical Owner | Open | TBD |
| ARC-OQ-008 | What exact rescheduling cutoff rule will the clinic configure for launch? | Clinic Administrator | Open | TBD |
| ARC-OQ-009 | What are the clinic-approved retention durations for operational PHI, consent records, reminder events, audit logs, backups, and integration files? | Compliance Owner | Open | TBD |
| ARC-OQ-010 | Is any initial schedule, provider, appointment type, or patient data migration required beyond EHR synchronization or CSV import for launch? | Technical Owner | Open | TBD |
| ARC-OQ-011 | What measurable production targets are required for uptime, p95 latency, reminder job delay, EHR sync freshness, and retry behavior? | Technical Owner | Open | TBD |
| ARC-OQ-012 | What pilot volume assumptions should performance and scalability tests use? | Product Owner | Open | TBD |
| ARC-OQ-013 | What automated test coverage threshold is required before production readiness approval? | Engineering Lead | Open | TBD |
| ARC-OQ-014 | Who are the named Security Reviewer and Engineering Lead approvers? | Product Owner | Open | TBD |

## 18. Feature Mapping

| Feature | Relevant Components | Decision IDs | Interface IDs | Security IDs |
|---|---|---|---|---|
| FEAT-001 | COMP-002, COMP-003, COMP-004, COMP-007, COMP-008 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-004 | ARD-API-004 | ARD-SEC-001 |
| FEAT-002 | COMP-001, COMP-003, COMP-004, COMP-005, COMP-007, COMP-008 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003, ARD-DEC-004, ARD-DEC-005 | ARD-API-001, ARD-API-005 | ARD-SEC-001 |
| FEAT-003 | COMP-001, COMP-003, COMP-004, COMP-005, COMP-007, COMP-008 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003, ARD-DEC-004, ARD-DEC-005 | ARD-API-002, ARD-API-004, ARD-API-005 | ARD-SEC-001 |
| FEAT-004 | COMP-001, COMP-004, COMP-006, COMP-008 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-006 | ARD-API-006 | ARD-SEC-001 |
| FEAT-005 | COMP-002, COMP-003, COMP-004, COMP-005, COMP-007, COMP-008 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003, ARD-DEC-004, ARD-DEC-005 | ARD-API-003, ARD-API-005 | ARD-SEC-001 |
| FEAT-006 | COMP-003, COMP-004, COMP-005, COMP-008 | ARD-DEC-003, ARD-DEC-005, ARD-DEC-006 | ARD-API-005 | ARD-SEC-001 |

## 19. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Solution Architect | TBD; see ARC-OQ-001 | Pending | TBD |
| Security Reviewer | TBD; see ARC-OQ-014 | Pending | TBD |
| Engineering Lead | TBD; see ARC-OQ-014 | Pending | TBD |