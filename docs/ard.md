# Architecture Requirements Document (ARD)

## Document Metadata

- **Product/Application:** Online Patient Scheduling and Reminders
- **Document ID:** ARD-OPSR
- **Version:** 0.1
- **Status:** Draft
- **Architecture Owner:** TBD — Product Owner to assign; see ARC-OQ-001
- **Contributors:** Product Owner, Solution Architect, Engineering Lead, Security Reviewer, Scheduling Staff
- **Last Updated:** 2026-11-16
- **Reference PRD:** `PRD.md`

---

## 1. Architecture Summary

Online Patient Scheduling and Reminders will be delivered as a single modular web application using a TypeScript React frontend, Node.js backend API, PostgreSQL database, and background workers. The system will run on HIPAA-eligible managed cloud services with signed BAAs for hosting, database, email, SMS, logging, and monitoring.

The architecture prioritizes small-business operational simplicity, HIPAA-aligned handling of scheduling data, and future integration flexibility because the clinic’s current scheduling system is not specified in the PRD. The first release will use an integration adapter layer and support manual staff-managed schedule data for the pilot until the scheduling system is identified. Once identified, the existing clinic scheduling system remains the system of record for appointments, while the application stores online booking rules, pending requests, reminder preferences, audit records, and synchronized or cached scheduling data.

Key architectural drivers are online self-service booking, secure rescheduling, HIPAA-conscious reminder delivery, role-based staff configuration, consolidated daily staff schedule visibility, accessibility, and measurable rollout instrumentation.

## 2. Architecture Drivers

- PRD-FR-001 — Staff must configure which appointment types are eligible for online booking.
- PRD-FR-002 — Staff must configure instant confirmation or manual approval by appointment type.
- PRD-FR-003 — Patients must book eligible appointments online and the system must capture submission time for measurement.
- PRD-FR-004 — Patients must be able to submit staff-reviewed booking requests when no eligible slots are available.
- PRD-FR-005 — Patients must complete secure verification before online rescheduling.
- PRD-FR-006 — Patients must choose SMS, email, or both for reminders.
- PRD-FR-007 — Reminder delivery must use content limited to appointment logistics.
- PRD-FR-008 — Staff must see all daily appointments plus pending booking and reschedule requests.
- PRD-FR-009 — Staff should review pending booking and reschedule requests from the staff schedule workflow.
- PRD-NFR-001 — Reminder templates must avoid diagnosis, treatment details, and sensitive PHI beyond appointment logistics.
- PRD-NFR-002 — Staff scheduling configuration must be protected by role-based access control.
- PRD-NFR-003 — Patient-facing booking and rescheduling flows must conform to WCAG 2.1 AA.
- PRD-NFR-004 — Failed verification must not expose appointment details.
- CON-001 — The product is for a healthcare clinic and must comply with HIPAA requirements.
- CON-002 — The product is for a small business and must avoid unnecessary first-release scope such as payments, insurance verification, and clinical intake forms.
- CON-003 — Reminder content must avoid diagnosis, treatment details, or sensitive PHI beyond appointment logistics.
- CON-004 — Emergency, urgent care, and clinically triaged appointment scenarios are excluded from launch.
- CON-005 — Scheduling system integration or data source is TBD until the current scheduling system is identified.
- CON-006 — Launch must begin with a pilot using one provider or appointment type before full rollout.

## 3. Architecture Principles

1. Protect patient scheduling data as PHI and apply least-privilege access throughout the system.
2. Prefer managed HIPAA-eligible services with signed BAAs over clinic-operated infrastructure.
3. Reuse established repository and platform patterns where appropriate.
4. Keep presentation, orchestration, domain, integration, and persistence concerns separated.
5. Isolate external scheduling, SMS, and email providers behind adapters so vendors can change without redesigning domain workflows.
6. Instrument booking, reminder, staff usage, verification, and operational flows from the pilot onward because PRD baselines and targets are TBD.
7. Design pilot workflows to support manual schedule management without blocking future scheduling-system integration.

## 4. System Context

### 4.1 Actors

- Patient — Uses public patient-facing flows to book eligible appointments, request appointments when no slot is available, choose reminder preferences, and securely reschedule eligible appointments.
- Scheduling Staff — Uses authenticated staff workflows to view the day’s schedule, review pending booking and reschedule requests, and configure booking rules where authorized.
- Clinic Administrator — Uses authenticated staff workflows to manage eligible appointment types, confirmation modes, and staff access.
- Product Owner — Owns unresolved product decisions such as scheduling source, KPI targets, office hours definition, pilot provider or appointment type, and approved communications providers.
- Security Reviewer — Reviews HIPAA-aligned controls, reminder templates, access controls, logging, dependency scanning, and SBOM outputs before production use.
- Existing Clinic Scheduling System — Once identified, remains the system of record for appointments and provides availability, appointment, provider, and status data.
- SMS Provider — Delivers appointment logistics reminders and verification codes under a signed BAA.
- Email Provider — Delivers appointment logistics reminders and verification codes under a signed BAA.
- Managed Cloud Platform — Hosts application, database, secrets, logs, metrics, alerts, backups, and monitoring under signed BAAs.

### 4.2 External Systems

| ID | System | Purpose | Interface | Owner |
|---|---|---|---|---|
| EXT-001 | Existing clinic scheduling system | System of record for appointments once identified; provides availability, appointments, providers, and status | REST/Event/File/TBD via adapter | Clinic / vendor TBD |
| EXT-002 | HIPAA-eligible SMS provider | Sends patient-selected reminders and verification codes | REST/Event | Vendor TBD |
| EXT-003 | HIPAA-eligible email provider | Sends patient-selected reminders and verification codes | REST/Event | Vendor TBD |
| EXT-004 | HIPAA-eligible managed cloud platform | Hosts application, database, secrets, logging, monitoring, backups, and CI/CD deployment targets | Managed service APIs | Cloud provider TBD |
| EXT-005 | Dependency and CVE scanning service | Blocks high and critical dependency findings and supports supply-chain review | CI/CD integration | Engineering Lead |
| EXT-006 | SBOM generation tool | Produces software bill of materials during CI/CD | CI/CD integration | Engineering Lead |

## 5. Technology Stack

| Layer | Technology | Version/Constraint | Rationale |
|---|---|---|---|
| Frontend | React with TypeScript | Current stable React and TypeScript versions selected by Engineering Lead before implementation | Supports accessible, componentized patient and staff web flows with shared type contracts. |
| Backend | Node.js with TypeScript | Current active LTS Node.js at implementation | Keeps frontend and backend in one language family and supports modular API, domain, and integration layers. |
| Data | PostgreSQL | Managed HIPAA-eligible PostgreSQL service under signed BAA | Provides relational integrity for scheduling rules, requests, audit events, reminder preferences, and integration state. |
| Background Jobs | Node.js/TypeScript worker process | Same runtime as backend | Supports reminder delivery, retry processing, scheduling sync, and asynchronous integration tasks. |
| Hosting | HIPAA-eligible managed cloud services | Signed BAAs required for hosting, database, email, SMS, logging, and monitoring | Reduces operational burden for a small business while supporting HIPAA-aligned controls. |
| CI/CD | Managed CI/CD pipeline | Must run required build, lint, test, accessibility, CVE scan, and SBOM gates | Provides repeatable deployment controls and evidence for release readiness. |
| Infrastructure | Infrastructure as Code | Tool TBD by Engineering Lead; separate dev, test, stage, and prod environments required | Enables consistent environments, reviewable changes, and repeatable recovery. |

## 6. Logical Architecture

### 6.1 Components

#### COMP-001 — Patient Web Application

- **Responsibility:** Provides patient-facing online booking, no-availability request, reminder preference, and secure rescheduling flows that conform to WCAG 2.1 AA.
- **Inputs:** Appointment type selections, patient contact and demographic details, reminder preferences, verification codes, rescheduling selections.
- **Outputs:** Booking submissions, staff-reviewed requests, reminder preferences, rescheduling requests, patient-facing confirmations or errors.
- **Dependencies:** COMP-002, COMP-006, COMP-007.
- **Repository Area:** `apps/web-patient`

#### COMP-002 — Backend API

- **Responsibility:** Exposes REST APIs, validates requests, enforces authorization, orchestrates booking, rescheduling, reminder preferences, and staff schedule workflows.
- **Inputs:** REST requests from patient and staff web applications, provider webhook events where available, worker callbacks.
- **Outputs:** REST responses, domain events, database writes, integration commands, audit events.
- **Dependencies:** COMP-004, COMP-005, COMP-006, COMP-007, COMP-008, COMP-009.
- **Repository Area:** `apps/api`

#### COMP-003 — Staff Web Application

- **Responsibility:** Provides authenticated staff views for daily schedule, pending booking and reschedule requests, appointment type eligibility, and confirmation mode configuration.
- **Inputs:** Staff credentials, MFA challenges, configuration edits, request review actions, date/provider filters.
- **Outputs:** Configuration changes, request review actions, schedule view queries, audit events.
- **Dependencies:** COMP-002, COMP-008.
- **Repository Area:** `apps/web-staff`

#### COMP-004 — Scheduling Domain Module

- **Responsibility:** Encapsulates appointment eligibility, confirmation mode, no-availability request, rescheduling cutoff, pending request, and schedule-display rules.
- **Inputs:** Appointment type configuration, availability data, appointment data, booking submissions, reschedule submissions, staff actions.
- **Outputs:** Confirmed appointment commands, pending request records, schedule view models, validation results.
- **Dependencies:** COMP-005, COMP-006, COMP-008.
- **Repository Area:** `packages/domain-scheduling`

#### COMP-005 — Scheduling Integration Adapter

- **Responsibility:** Abstracts the current scheduling data source and supports manual staff-managed schedule data during the pilot until the scheduling system is identified.
- **Inputs:** Availability queries, appointment creation/update commands, schedule sync jobs, manual availability entries.
- **Outputs:** Availability results, appointment records, provider records, appointment status updates, integration errors.
- **Dependencies:** EXT-001 when identified, COMP-006.
- **Repository Area:** `packages/integrations/scheduling`

#### COMP-006 — Application Database

- **Responsibility:** Persists booking rules, pending requests, reminder preferences, audit events, delivery attempts, user roles, verification state, and synced or cached schedule data.
- **Inputs:** Domain writes, integration sync results, audit events, delivery attempts, staff configuration.
- **Outputs:** Query results for APIs, workers, staff schedule views, audit review, and KPI measurement.
- **Dependencies:** Managed PostgreSQL service.
- **Repository Area:** `infra/database`

#### COMP-007 — Reminder Worker

- **Responsibility:** Schedules and sends patient-selected SMS and email reminders using HIPAA-conscious templates and records delivery attempts.
- **Inputs:** Reminder preferences, upcoming appointments, approved templates, retry queue items.
- **Outputs:** SMS/email provider requests, delivery attempt records, operational metrics, audit events.
- **Dependencies:** COMP-006, EXT-002, EXT-003, COMP-009.
- **Repository Area:** `apps/worker-reminders`

#### COMP-008 — Staff Identity and Access Module

- **Responsibility:** Provides clinic-managed staff accounts with MFA and application-managed roles for clinic administrator and scheduling staff.
- **Inputs:** Staff login attempts, MFA challenges, role assignments, session requests.
- **Outputs:** Authenticated sessions, authorization decisions, access-denied responses, audit events.
- **Dependencies:** COMP-006, managed secrets service.
- **Repository Area:** `packages/auth`

#### COMP-009 — Audit and Observability Module

- **Responsibility:** Produces structured logs, audit events, metrics, alerts, and PHI-redacted operational telemetry.
- **Inputs:** Application events, security events, reminder delivery attempts, integration errors, CI/CD deployment events.
- **Outputs:** Centralized logs, metrics, alerts, audit records, operational dashboards.
- **Dependencies:** EXT-004.
- **Repository Area:** `packages/observability`

#### COMP-010 — CI/CD and Infrastructure Module

- **Responsibility:** Defines infrastructure, environments, deployment pipelines, test gates, security scans, SBOM generation, and rollback automation.
- **Inputs:** Source code changes, infrastructure changes, pipeline configuration, secrets references.
- **Outputs:** Deployed environments, build artifacts, test reports, scan results, SBOM artifacts, rollback runbooks.
- **Dependencies:** EXT-004, EXT-005, EXT-006.
- **Repository Area:** `infra`

### 6.2 Data Flow

1. Patient Web Application sends selected appointment type and requested slot to Backend API.
2. Backend API validates the request through Scheduling Domain Module and queries Scheduling Integration Adapter for availability.
3. Scheduling Integration Adapter reads manual pilot schedule data or, once identified, calls the existing clinic scheduling system.
4. Scheduling Domain Module applies online eligibility, confirmation mode, and no-availability rules.
5. Backend API writes confirmed booking state, pending request state, reminder preference, submission timestamp, and audit events to Application Database.
6. Patient Web Application returns confirmation, pending request status, no-availability request status, or validation errors to the patient.
7. Reminder Worker reads upcoming appointment and reminder preference data, renders approved appointment-logistics templates, sends through SMS and/or email providers, and records delivery attempts.
8. Staff Web Application authenticates staff through Staff Identity and Access Module and requests the daily schedule view from Backend API.
9. Backend API composes daily appointment, pending booking request, and pending reschedule request data from Application Database and Scheduling Integration Adapter.
10. Audit and Observability Module emits structured logs, metrics, alerts, and audit records without exposing unnecessary PHI.

## 7. Architecture Decisions

### ARD-DEC-001 — Use a Single Modular Web Application

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-003, PRD-FR-004, PRD-FR-005, PRD-FR-006, PRD-FR-007, PRD-FR-008, PRD-FR-009

**Decision**

Build the first release as a single modular web application with separated frontend applications, backend API, domain modules, integration adapters, background workers, and PostgreSQL persistence.

**Rationale**

The clinic is a small business and needs a maintainable first release without microservice operational overhead. Modular boundaries keep future integration and scaling options open while reducing deployment complexity.

**Consequences**

- Lower operational complexity than microservices.
- Clear internal boundaries for patient, staff, domain, integration, worker, and persistence concerns.
- Scaling is initially application-level rather than independent per domain service.

**Alternatives Considered**

- Microservices split by booking, reminders, staff schedule, and configuration — not selected because operational overhead is too high for the first release.
- Static website plus third-party scheduling widgets — not selected because HIPAA, staff rules, pending requests, and integration needs require controlled workflows.
- No-code scheduling platform — not selected because the PRD requires specific security, reminder, schedule view, and integration behavior.

### ARD-DEC-002 — Host on HIPAA-Eligible Managed Cloud Services with BAAs

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001, PRD-NFR-002, PRD-NFR-004

**Decision**

Use HIPAA-eligible managed cloud services with signed BAAs for hosting, database, email, SMS, logging, and monitoring.

**Rationale**

Managed services reduce infrastructure burden for a small business while supporting HIPAA-aligned controls, backups, observability, access management, and operational resilience.

**Consequences**

- Cloud provider and communications vendors must support signed BAAs before production use.
- Vendor selection becomes a security and compliance gating activity.
- The clinic avoids self-hosted infrastructure operations for the pilot.

**Alternatives Considered**

- Self-hosted clinic infrastructure — not selected because it increases operational and security burden.
- Existing EHR or practice-management scheduling module only — not selected because the current scheduling system is unspecified and may not satisfy PRD requirements.
- Deployment approach TBD — not selected because HIPAA hosting posture materially affects the architecture.

### ARD-DEC-003 — Use TypeScript, React, Node.js, PostgreSQL, and Background Workers

- **Status:** Accepted
- **Related Requirements:** PRD-FR-003, PRD-FR-005, PRD-FR-007, PRD-FR-008, PRD-NFR-003

**Decision**

Standardize on TypeScript with React frontend applications, Node.js backend API, PostgreSQL database, and Node.js/TypeScript background workers.

**Rationale**

A TypeScript stack supports shared typing across frontend, backend, and workers. PostgreSQL provides relational consistency for appointment rules, pending requests, audit records, and reminder preferences.

**Consequences**

- Engineering must define exact runtime and framework versions before implementation.
- Shared TypeScript contracts can reduce integration errors between frontend and backend.
- PostgreSQL schema governance and migration discipline are required.

**Alternatives Considered**

- Python/Django with Celery — viable but not selected after architecture interview.
- Ruby on Rails — viable but not selected after architecture interview.
- Technology stack TBD — not selected because architecture and testing decisions require a stack baseline.

### ARD-DEC-004 — Isolate Scheduling Source Behind an Adapter

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-003, PRD-FR-004, PRD-FR-005, PRD-FR-008, PRD-FR-009

**Decision**

Create a scheduling integration adapter layer and initially support manual staff-managed schedule data during the pilot until the current scheduling system is identified.

**Rationale**

The PRD states that the scheduling data source is TBD. An adapter prevents the domain model and API from depending on an unknown vendor while allowing pilot progress.

**Consequences**

- Manual pilot schedule management may require operational discipline and staff training.
- Future scheduling-system integration can be added behind the adapter.
- The adapter must surface integration limitations and data freshness in staff workflows.

**Alternatives Considered**

- Assume direct real-time API integration — not selected because the current scheduling system is unspecified.
- Nightly file import/export only — not selected because it may not support real-time booking and rescheduling needs.
- Do not build until scheduling system is identified — not selected because the pilot can proceed with manual schedule data.

### ARD-DEC-005 — Keep Existing Clinic Scheduling System as Appointment System of Record Once Identified

- **Status:** Accepted
- **Related Requirements:** PRD-FR-003, PRD-FR-005, PRD-FR-008

**Decision**

Once the current scheduling system is identified, it remains the system of record for appointments. The application stores requests, configuration, reminder preferences, audit records, and synced or cached scheduling data.

**Rationale**

The clinic likely already operates a scheduling workflow. Keeping the existing scheduling system as appointment system of record reduces migration risk and aligns online scheduling with staff operations.

**Consequences**

- Integration quality and data freshness become important reliability concerns.
- The application database is not the long-term authoritative source for appointments.
- During the pilot, manual staff-managed schedule data is an interim source until system discovery is complete.

**Alternatives Considered**

- Application database as permanent appointment system of record — not selected because it may conflict with existing clinic operations.
- Staff double-entry as the normal operating model — not selected because it increases error and workload risk.
- Leave system of record TBD — not selected because the system-of-record principle affects integration and data architecture.

### ARD-DEC-006 — Use REST APIs, Background Jobs, and Provider Webhooks Where Available

- **Status:** Accepted
- **Related Requirements:** PRD-FR-003, PRD-FR-005, PRD-FR-007, PRD-FR-008

**Decision**

Use REST APIs for web and scheduling adapter calls, background jobs for reminders and asynchronous integration work, and provider webhooks where available.

**Rationale**

REST is straightforward for web application workflows and vendor adapters. Background jobs isolate reminder delivery and retries from synchronous patient and staff interactions.

**Consequences**

- API versioning and contract tests are required.
- Idempotency is required for booking, rescheduling, reminder, and webhook operations.
- Worker observability and retry policies must be implemented.

**Alternatives Considered**

- GraphQL and event streaming for all integrations — not selected because it adds unnecessary first-release complexity.
- File-only integration — not selected because it may not support interactive booking and rescheduling.
- Integration style TBD — not selected because integration style affects component boundaries.

### ARD-DEC-007 — Require Clinic-Managed Staff Accounts with MFA and Application Roles

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-008, PRD-FR-009, PRD-NFR-002

**Decision**

Use clinic-managed staff accounts with MFA and application-managed roles for clinic administrator and scheduling staff.

**Rationale**

The PRD requires role-based configuration access but does not specify an existing identity provider. Clinic-managed accounts with MFA provide a concrete, secure first-release model.

**Consequences**

- Staff account lifecycle management must be implemented.
- MFA enrollment and recovery workflows are required.
- Future SSO integration should remain possible through the Staff Identity and Access Module.

**Alternatives Considered**

- Existing identity provider SSO — not selected because no identity provider is specified.
- Shared clinic login — not selected because it undermines accountability and audit logging.
- Staff authentication TBD — not selected because access control is a core security requirement.

### ARD-DEC-008 — Verify Rescheduling with One-Time Code and Demographic Match

- **Status:** Accepted
- **Related Requirements:** PRD-FR-005, PRD-NFR-004

**Decision**

Before rescheduling, require a one-time code sent to the appointment’s existing email or mobile number plus demographic matching. Failed verification must not expose appointment details.

**Rationale**

This approach supports patients who do not have portal accounts while adding verification beyond possession of a link or contact details alone.

**Consequences**

- Rescheduling depends on accurate existing contact information.
- Verification attempts, failures, and lockouts must be logged and monitored.
- Usability must be validated during the pilot.

**Alternatives Considered**

- Patient portal sign-in only — not selected because portal availability is not specified.
- Staff-issued rescheduling link only — not selected because it adds staff workload and may not support self-service.
- Verification TBD — not selected because secure rescheduling is a must-have PRD requirement.

### ARD-DEC-009 — Avoid Bulk Data Migration for the Pilot

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-003, PRD-FR-008

**Decision**

Do not perform bulk migration for the pilot. Configure eligible appointment types and import or manually enter only required availability data for the pilot provider or appointment type.

**Rationale**

The PRD requires a pilot and the scheduling source is TBD. Avoiding bulk migration reduces risk and preserves flexibility while discovery continues.

**Consequences**

- Historical appointment data is not available in the application unless sourced through integration later.
- Pilot scope must remain narrow enough for manual or limited availability data management.
- Migration strategy must be revisited after the scheduling system is identified.

**Alternatives Considered**

- Migrate all historical scheduling records — not selected because it is unnecessary for the pilot and high risk without a known source system.
- Migrate future appointments only — possible later, but not selected as a default before source discovery.
- Migration TBD only — not selected because pilot architecture needs a migration posture.

## 8. API and Integration Architecture

### ARD-API-001 — Patient Booking API

- **Type:** REST
- **Operation/Topic:** `POST /api/v1/bookings`
- **Provider:** COMP-002
- **Consumers:** COMP-001
- **Authentication:** Public submission with server-side validation, anti-abuse controls, and PHI-minimizing payloads
- **Versioning:** URI versioning with backward-compatible additive changes within `v1`

**Request/Input Contract**

Includes selected appointment type ID, selected slot ID when available, patient contact details, required demographic details, reminder channel preference, and submission metadata. The contract must exclude payment, insurance verification, clinical intake, diagnosis, treatment details, and visit-reason details.

**Response/Output Contract**

Returns booking status of `confirmed`, `pending_staff_approval`, `no_availability_request_created`, or validation failure; includes patient-safe confirmation details limited to appointment logistics.

**Error Handling**

Validation errors return patient-safe messages. Duplicate submissions use idempotency keys. Integration timeouts create recoverable pending states where safe. Errors must not expose internal system details or unnecessary PHI.

### ARD-API-002 — Patient Rescheduling Verification API

- **Type:** REST
- **Operation/Topic:** `POST /api/v1/rescheduling/verification/start` and `POST /api/v1/rescheduling/verification/confirm`
- **Provider:** COMP-002
- **Consumers:** COMP-001
- **Authentication:** One-time code to existing appointment email or mobile number plus demographic match
- **Versioning:** URI versioning with backward-compatible additive changes within `v1`

**Request/Input Contract**

Start request includes appointment lookup inputs that do not reveal appointment details. Confirm request includes one-time code and demographic match fields.

**Response/Output Contract**

Before successful verification, responses reveal only whether the verification process can proceed, without exposing appointment details. After successful verification, the API returns patient-safe rescheduling eligibility and available logistics data.

**Error Handling**

Failed verification returns generic messages and does not expose appointment details. Rate limits, lockouts, and audit events are required for repeated failures.

### ARD-API-003 — Patient Rescheduling API

- **Type:** REST
- **Operation/Topic:** `POST /api/v1/reschedules`
- **Provider:** COMP-002
- **Consumers:** COMP-001
- **Authentication:** Verified rescheduling session
- **Versioning:** URI versioning with backward-compatible additive changes within `v1`

**Request/Input Contract**

Includes verified appointment reference, selected new slot, reminder preference updates if applicable, and idempotency key.

**Response/Output Contract**

Returns reschedule status of `confirmed`, `pending_staff_approval`, `blocked_by_cutoff`, `not_eligible`, or validation failure.

**Error Handling**

Cutoff and eligibility failures return patient-safe explanations. Integration failures create pending staff review only where clinically and operationally safe. Duplicate submissions are idempotent.

### ARD-API-004 — Staff Schedule API

- **Type:** REST
- **Operation/Topic:** `GET /api/v1/staff/schedule?date={date}`
- **Provider:** COMP-002
- **Consumers:** COMP-003
- **Authentication:** Authenticated staff session with clinic administrator or scheduling staff role
- **Versioning:** URI versioning with backward-compatible additive changes within `v1`

**Request/Input Contract**

Includes date filter and optional provider or appointment type filters. Staff session determines permitted access.

**Response/Output Contract**

Returns daily appointments, pending booking requests, pending reschedule requests, patient name, appointment time, provider, appointment type, and status when available from the scheduling source.

**Error Handling**

Unauthorized users receive access-denied responses. Partial integration failures return available cached or local pending data with freshness indicators and operational alerts.

### ARD-API-005 — Booking Rules Configuration API

- **Type:** REST
- **Operation/Topic:** `PUT /api/v1/staff/booking-rules/{appointmentTypeId}`
- **Provider:** COMP-002
- **Consumers:** COMP-003
- **Authentication:** Authenticated staff session with clinic administrator or scheduling staff role
- **Versioning:** URI versioning with backward-compatible additive changes within `v1`

**Request/Input Contract**

Includes appointment type eligibility, confirmation mode, online rescheduling eligibility, and staff-defined rescheduling cutoff.

**Response/Output Contract**

Returns updated booking rule state and audit reference.

**Error Handling**

Unauthorized changes are rejected and audited. Invalid configurations are rejected with staff-safe validation messages.

### ARD-API-006 — Scheduling Adapter Interface

- **Type:** REST | File | Event
- **Operation/Topic:** `SchedulingAdapter.getAvailability`, `SchedulingAdapter.createAppointment`, `SchedulingAdapter.updateAppointment`, `SchedulingAdapter.syncSchedule`
- **Provider:** COMP-005
- **Consumers:** COMP-004, COMP-002, COMP-007
- **Authentication:** Service-to-service credentials stored in managed secrets service; vendor-specific authentication once EXT-001 is identified
- **Versioning:** Internal interface versioning with adapter implementations per source system

**Request/Input Contract**

Includes appointment type, provider, date range, slot, appointment identifier, patient scheduling identifiers, and operation idempotency key as required by each adapter method.

**Response/Output Contract**

Returns normalized availability, appointment, provider, status, and integration error models for the domain layer.

**Error Handling**

Adapter normalizes vendor errors, timeouts, conflicts, and stale data indicators. Write operations require idempotency and audit events.

### ARD-API-007 — Reminder Delivery Interface

- **Type:** REST | Event
- **Operation/Topic:** `ReminderWorker.sendSmsReminder`, `ReminderWorker.sendEmailReminder`, provider delivery webhooks where available
- **Provider:** COMP-007
- **Consumers:** EXT-002, EXT-003, COMP-009
- **Authentication:** Provider API credentials stored in managed secrets service under signed BAA
- **Versioning:** Provider adapter versioning with internal normalized delivery contract

**Request/Input Contract**

Includes approved appointment-logistics reminder content, destination channel, appointment logistics, delivery schedule, and idempotency key. Content must exclude diagnosis, treatment details, visit reason, and sensitive PHI beyond appointment logistics.

**Response/Output Contract**

Returns normalized delivery status, provider message identifier, retry eligibility, and failure reason category.

**Error Handling**

Transient failures use bounded retries. Permanent failures are recorded and surfaced in operational logs. Delivery webhook events are idempotently processed.

## 9. Data Architecture

### DATA-001 — Patient Scheduling Profile

- **Purpose:** Stores contact and demographic details required for online booking and verification.
- **System of Record:** Application database for online-submitted scheduling profile data; existing clinic scheduling system for authoritative patient/appointment linkage once identified.
- **Classification:** Restricted
- **Retention:** Clinic-defined HIPAA-aligned period; exact duration TBD under ARC-OQ-005.
- **Key Relationships:** Booking requests, appointments, reminder preferences, verification attempts, audit events.

### DATA-002 — Appointment

- **Purpose:** Represents scheduled appointment logistics needed for booking, rescheduling, reminders, and staff daily schedule display.
- **System of Record:** Existing clinic scheduling system once identified; manual pilot schedule data is interim until EXT-001 is resolved.
- **Classification:** Restricted
- **Retention:** Existing scheduling system retention policy for authoritative records; cached/synced application data retained for clinic-defined HIPAA-aligned period, exact duration TBD under ARC-OQ-005.
- **Key Relationships:** Patient scheduling profile, appointment type, provider, reminder preference, pending request, audit event.

### DATA-003 — Appointment Type Rule

- **Purpose:** Stores online eligibility, confirmation mode, online rescheduling eligibility, and rescheduling cutoff by appointment type.
- **System of Record:** Application database.
- **Classification:** Internal
- **Retention:** Retain active rules and change history for clinic-defined HIPAA-aligned audit period; exact duration TBD under ARC-OQ-005.
- **Key Relationships:** Appointment type, staff user, audit event, booking request, reschedule request.

### DATA-004 — Booking Request

- **Purpose:** Captures patient booking submissions, no-availability requests, and pending staff approval workflows.
- **System of Record:** Application database until converted into authoritative appointment in scheduling system once identified.
- **Classification:** Restricted
- **Retention:** Clinic-defined HIPAA-aligned period; exact duration TBD under ARC-OQ-005.
- **Key Relationships:** Patient scheduling profile, appointment type rule, appointment, staff review action, audit event.

### DATA-005 — Reschedule Request

- **Purpose:** Captures verified patient rescheduling submissions and pending staff approval workflows.
- **System of Record:** Application database until applied to authoritative appointment in scheduling system once identified.
- **Classification:** Restricted
- **Retention:** Clinic-defined HIPAA-aligned period; exact duration TBD under ARC-OQ-005.
- **Key Relationships:** Appointment, patient scheduling profile, verification attempt, appointment type rule, staff review action, audit event.

### DATA-006 — Reminder Preference

- **Purpose:** Stores patient-selected SMS, email, or both reminder channel choices.
- **System of Record:** Application database for online reminder preferences.
- **Classification:** Restricted
- **Retention:** Clinic-defined HIPAA-aligned period; exact duration TBD under ARC-OQ-005.
- **Key Relationships:** Appointment, patient scheduling profile, reminder delivery attempt.

### DATA-007 — Reminder Delivery Attempt

- **Purpose:** Records SMS and email reminder delivery attempts for operational review and troubleshooting.
- **System of Record:** Application database and normalized provider event records.
- **Classification:** Restricted
- **Retention:** Clinic-defined HIPAA-aligned period; exact duration TBD under ARC-OQ-005.
- **Key Relationships:** Reminder preference, appointment, provider message ID, audit event.

### DATA-008 — Staff User and Role

- **Purpose:** Stores staff account, MFA, and authorization role data for clinic administrators and scheduling staff.
- **System of Record:** Application database for first release.
- **Classification:** Confidential
- **Retention:** Retain active accounts and deactivation/audit history for clinic-defined HIPAA-aligned period; exact duration TBD under ARC-OQ-005.
- **Key Relationships:** Role assignments, booking rule changes, staff request reviews, audit events.

### DATA-009 — Audit Event

- **Purpose:** Records security-relevant and compliance-relevant activity, including access, configuration changes, verification attempts, booking/rescheduling actions, and reminder delivery events.
- **System of Record:** Application database and centralized logging/audit service.
- **Classification:** Restricted
- **Retention:** Clinic-defined HIPAA-aligned period; exact duration TBD under ARC-OQ-005.
- **Key Relationships:** Staff user, patient scheduling profile, appointment, booking request, reschedule request, reminder delivery attempt.

### Data Migration

- **Required:** No bulk migration for the pilot.
- **Approach:** Configure eligible appointment types and import or manually enter only required availability data for the pilot provider or appointment type. Reassess migration after the current scheduling system is identified and EXT-001 integration requirements are known.

## 10. Security Architecture

### ARD-SEC-001 — HIPAA-Aligned PHI Handling

- **Related Requirements:** PRD-NFR-001, PRD-NFR-004

The implementation must:

- Treat patient scheduling, appointment, reminder, verification, and audit data as PHI where applicable.
- Use HIPAA-eligible managed services with signed BAAs for hosting, database, SMS, email, logging, and monitoring.
- Minimize PHI in API responses, logs, reminders, and operational dashboards.
- Require HIPAA/privacy review before production use.

The implementation must not:

- Include diagnosis, treatment details, visit reason, or sensitive PHI beyond appointment logistics in reminder content.
- Store secrets in source code, plain text configuration, logs, or client-side code.
- Expose appointment details before secure rescheduling verification succeeds.

### ARD-SEC-002 — Staff Role-Based Access Control

- **Related Requirements:** PRD-NFR-002

The implementation must:

- Restrict booking rule configuration to clinic administrators and scheduling staff.
- Require staff authentication with MFA.
- Audit role changes, configuration changes, staff access, and request review actions.

The implementation must not:

- Allow shared staff accounts for production use.
- Allow unauthorized staff users to modify appointment type eligibility or confirmation mode.

### ARD-SEC-003 — Secure Patient Rescheduling Verification

- **Related Requirements:** PRD-FR-005, PRD-NFR-004

The implementation must:

- Require a one-time code sent to the appointment’s existing email or mobile number plus demographic match before rescheduling.
- Rate-limit and audit verification attempts.
- Use generic failure messages that do not confirm appointment existence or expose appointment details.

The implementation must not:

- Display appointment details before verification succeeds.
- Allow rescheduling after the staff-defined cutoff.

### ARD-SEC-004 — Supply Chain Security

- **Related Requirements:** PRD-NFR-001, PRD-NFR-002, PRD-NFR-004

The implementation must:

- Run dependency and CVE scanning in CI/CD.
- Block production deployment on high and critical dependency/CVE findings unless formally approved by Security Reviewer.
- Generate an SBOM for production builds.

The implementation must not:

- Deploy production artifacts without required build, lint, test, scan, and SBOM gates.

### Security Controls

- **Authentication:** Staff use clinic-managed accounts with MFA; patients use public booking flow for new bookings and one-time code plus demographic match before rescheduling.
- **Authorization:** Application-managed RBAC for clinic administrator and scheduling staff roles; least-privilege service accounts for integrations and workers.
- **Secrets Management:** Managed cloud secrets service; secrets never stored in source code, logs, or client-side configuration.
- **Encryption:** TLS for data in transit; managed database and storage encryption at rest; encrypted backups.
- **Audit Logging:** Audit staff access, role changes, booking rule changes, booking submissions, reschedule submissions, verification attempts, reminder delivery attempts, integration writes, and security events.
- **Dependency/CVE Scanning:** CI/CD scan blocks high and critical findings unless Security Reviewer approves an exception.
- **SBOM:** Required for production builds.

## 11. Reliability, Performance, and Scalability

### ARD-OPS-001 — Availability

- **Target:** 99.5% availability during clinic-defined business hours for production patient booking, rescheduling, and staff schedule workflows.
- **Validation Method:** Managed monitoring uptime checks, incident review, and monthly availability report after launch.

### ARD-OPS-002 — Backup and Restore

- **Target:** Daily backups for scheduling-related application data, with restore validation performed before production launch and on a recurring cadence defined by Engineering Lead.
- **Validation Method:** Documented restore test in stage or test environment using backup artifacts and a restore runbook.

### ARD-OPS-003 — Performance

- **Target:** 95% of booking, rescheduling, and staff schedule page requests complete within 2 seconds under pilot load.
- **Validation Method:** Performance tests using pilot-load scenarios and production monitoring of p95 latency.

### ARD-OPS-004 — Reminder Delivery Reliability

- **Target:** Reminder delivery attempts are recorded for 100% of reminders sent through configured SMS or email providers.
- **Validation Method:** Worker integration tests, provider webhook reconciliation where available, and operational dashboard review.

### ARD-OPS-005 — Accessibility

- **Target:** Patient-facing booking and rescheduling flows conform to WCAG 2.1 AA before launch.
- **Validation Method:** Automated accessibility checks, keyboard navigation validation, screen reader label review, and documented pre-launch accessibility evaluation.

### ARD-OPS-006 — Rescheduling Verification Safety

- **Target:** 100% of rescheduling flows require successful verification before appointment details are displayed.
- **Validation Method:** Automated integration tests, security tests for failed verification paths, and audit log review.

## 12. Observability

- **Logging:** Centralized structured application, worker, security, integration, and deployment logs in HIPAA-eligible managed services with PHI redaction and correlation IDs.
- **Metrics:** API latency, API error rate, booking submission count, outside-hours booking count, pending request counts, reminder delivery attempts, reminder failures, verification failures, staff daily schedule usage, worker queue depth, integration errors, and uptime.
- **Tracing:** Request correlation across frontend, backend API, domain modules, scheduling adapter, database, and worker jobs; full distributed tracing depth depends on selected cloud services.
- **Alerting:** Alerts for production downtime, elevated API errors, reminder worker failures, high integration error rate, failed backups, failed CI/CD security gates, and suspicious verification failure patterns.
- **Sensitive Data Restrictions:** Logs, metrics, traces, alerts, and dashboards must not include diagnosis, treatment details, visit reason, full reminder content where PHI risk exists, full verification codes, secrets, or unnecessary patient identifiers.

## 13. Deployment Architecture

- **Environments:** Separate dev, test, stage, and prod environments.
- **Deployment Model:** Managed cloud deployment for frontend, backend API, worker processes, PostgreSQL database, secrets, logs, metrics, alerts, and backups.
- **CI/CD:** Pipeline runs build, lint, unit tests, integration tests, accessibility checks, dependency/CVE scan blocking high and critical findings, and SBOM generation before production deployment.
- **Configuration Management:** Environment-specific configuration through managed secrets and configuration services; no secrets in source code.
- **Rollback:** Blue/green or rolling deployment with database migrations designed to be backward compatible and rollback runbooks tested before production.
- **Infrastructure as Code:** Required; tool TBD by Engineering Lead under ARC-OQ-010.

## 14. Testing Architecture

Required test levels:

- Unit tests
- Integration tests
- Contract tests
- End-to-end tests
- Security tests
- Performance tests, where applicable

Minimum quality expectations:

- **Coverage Threshold:** 80% coverage for backend domain/API code; critical patient and staff workflows must be covered by integration or end-to-end tests.
- **Blocking Security Severity:** High and critical dependency/CVE findings block production deployment unless Security Reviewer approves an exception.
- **Required Build/Lint Checks:** TypeScript type checks, frontend build, backend build, worker build, lint checks, unit tests, integration tests, accessibility checks for patient-facing flows, contract tests for REST APIs and provider adapters, dependency/CVE scan, and SBOM generation.

## 15. Architecture Constraints

- ARC-CON-001 — The system must comply with HIPAA requirements because the product operates in a healthcare clinic.
- ARC-CON-002 — HIPAA-eligible managed services and signed BAAs are required for hosting, database, SMS, email, logging, and monitoring.
- ARC-CON-003 — Reminder content must avoid diagnosis, treatment details, visit reason, and sensitive PHI beyond appointment logistics.
- ARC-CON-004 — Emergency, urgent care, and clinically triaged appointment scenarios are excluded from launch.
- ARC-CON-005 — Payments, insurance verification, and clinical intake forms are excluded from the first release.
- ARC-CON-006 — Scheduling system integration must remain adapter-based because the current scheduling system is not specified in the PRD.
- ARC-CON-007 — Launch must begin with a pilot using one provider or appointment type before full rollout.
- ARC-CON-008 — Patient-facing booking and rescheduling flows must conform to WCAG 2.1 AA.
- ARC-CON-009 — Staff booking rule configuration must be limited to clinic administrators and scheduling staff.
- ARC-CON-010 — Production deployment must pass required build, lint, test, accessibility, CVE scan, and SBOM gates.

## 16. Risks and Trade-offs

| ID | Risk/Trade-off | Impact | Mitigation |
|---|---|---|---|
| ARC-RISK-001 | Current scheduling system is unknown, creating integration uncertainty | High | Use scheduling adapter and manual pilot schedule data until EXT-001 is identified. |
| ARC-RISK-002 | Manual pilot schedule data may diverge from clinic operations | Medium | Limit pilot to one provider or appointment type and require staff training and reconciliation procedures. |
| ARC-RISK-003 | Reminder content could expose PHI if templates are too detailed | High | Restrict reminders to appointment logistics and require HIPAA/privacy review before production. |
| ARC-RISK-004 | Verification may block legitimate patients if contact data is stale | Medium | Monitor failed verification attempts during pilot and provide staff-assisted fallback outside the online flow. |
| ARC-RISK-005 | HIPAA-eligible SMS, email, logging, or monitoring vendors may not be selected in time | High | Keep vendors behind adapters and track vendor approval as open architecture questions. |
| ARC-RISK-006 | KPI baselines and targets are TBD, limiting success measurement at launch | Medium | Instrument booking timestamps, phone workload proxies, reminder adoption, and staff view usage from pilot start. |
| ARC-RISK-007 | Small-business budget may constrain availability, observability, or environment separation | Medium | Use managed services and prioritize required HIPAA, backup, CI/CD, and observability controls. |
| ARC-RISK-008 | Backward-compatible migration discipline adds engineering overhead | Medium | Require migration review and rollback runbooks before production deployment. |

## 17. Open Architecture Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| ARC-OQ-001 | Who is the named Architecture Owner for ARD-OPSR? | Product Owner | Open | TBD |
| ARC-OQ-002 | What current scheduling system or data source will provide appointment availability, appointments, providers, and status? | Product Owner | Open | TBD |
| ARC-OQ-003 | What office hours definition should be used to measure outside-hours bookings? | Product Owner | Open | TBD |
| ARC-OQ-004 | What baseline and target should be set for completed online bookings outside office hours, staff phone workload, reminder adoption, and staff schedule visibility usage? | Product Owner | Open | TBD |
| ARC-OQ-005 | What exact retention duration applies to appointment, request, audit, reminder delivery, staff access, and verification records? | Product Owner / Security Reviewer | Open | TBD |
| ARC-OQ-006 | Which provider or appointment type will be used for the pilot rollout? | Product Owner | Open | TBD |
| ARC-OQ-007 | Which HIPAA-eligible SMS provider with signed BAA is approved for reminder and verification delivery? | Product Owner / Security Reviewer | Open | TBD |
| ARC-OQ-008 | Which HIPAA-eligible email provider with signed BAA is approved for reminder and verification delivery? | Product Owner / Security Reviewer | Open | TBD |
| ARC-OQ-009 | Which HIPAA-eligible managed cloud platform and observability services with signed BAAs are approved? | Product Owner / Security Reviewer | Open | TBD |
| ARC-OQ-010 | Which Infrastructure as Code tool and exact CI/CD platform will Engineering Lead use? | Engineering Lead | Open | TBD |
| ARC-OQ-011 | What exact active LTS Node.js version, React version, TypeScript version, and PostgreSQL version will be selected before implementation? | Engineering Lead | Open | TBD |
| ARC-OQ-012 | What recurring cadence will be used for restore validation after launch? | Engineering Lead | Open | TBD |
| ARC-OQ-013 | What staff-assisted fallback process will be used when legitimate patients fail secure rescheduling verification? | Scheduling Staff / Product Owner | Open | TBD |

## 18. Feature Mapping

| Feature | Relevant Components | Decision IDs | Interface IDs | Security IDs |
|---|---|---|---|---|
| FEAT-001 | COMP-003, COMP-002, COMP-004, COMP-006, COMP-008 | ARD-DEC-001, ARD-DEC-003, ARD-DEC-007 | ARD-API-005 | ARD-SEC-002, ARD-SEC-004 |
| FEAT-002 | COMP-001, COMP-002, COMP-004, COMP-005, COMP-006 | ARD-DEC-001, ARD-DEC-003, ARD-DEC-004, ARD-DEC-005, ARD-DEC-006, ARD-DEC-009 | ARD-API-001, ARD-API-006 | ARD-SEC-001, ARD-SEC-004 |
| FEAT-003 | COMP-001, COMP-002, COMP-004, COMP-005, COMP-006 | ARD-DEC-001, ARD-DEC-004, ARD-DEC-005, ARD-DEC-006, ARD-DEC-008 | ARD-API-002, ARD-API-003, ARD-API-006 | ARD-SEC-001, ARD-SEC-003, ARD-SEC-004 |
| FEAT-004 | COMP-001, COMP-002, COMP-006, COMP-007, COMP-009 | ARD-DEC-002, ARD-DEC-003, ARD-DEC-006 | ARD-API-007 | ARD-SEC-001, ARD-SEC-004 |
| FEAT-005 | COMP-003, COMP-002, COMP-004, COMP-005, COMP-006, COMP-008 | ARD-DEC-001, ARD-DEC-004, ARD-DEC-005, ARD-DEC-006, ARD-DEC-007 | ARD-API-004, ARD-API-006 | ARD-SEC-001, ARD-SEC-002, ARD-SEC-004 |

## 19. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Solution Architect | TBD | Pending | TBD |
| Security Reviewer | TBD | Pending | TBD |
| Engineering Lead | TBD | Pending | TBD |