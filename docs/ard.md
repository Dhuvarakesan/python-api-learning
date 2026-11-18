# Architecture Requirements Document (ARD)

> Replace all `<PLACEHOLDER>` values. Every material architecture decision or constraint should have a stable ID that feature specifications can reference.

## Document Metadata

- **Product/Application:** Clinic Online Scheduling and Reminders
- **Document ID:** ARD-COSR
- **Version:** 0.1
- **Status:** Draft
- **Architecture Owner:** TBD — Solution Architect assignment pending
- **Contributors:** Product Owner TBD, Compliance Owner TBD, Technical Lead TBD, Security Reviewer TBD, Clinic Operations Owner TBD
- **Last Updated:** 2026-11-18
- **Reference PRD:** `PRD.md`

---

## 1. Architecture Summary

Clinic Online Scheduling and Reminders will use a hybrid architecture: a custom React/TypeScript patient and staff experience, a Node.js/TypeScript backend, managed HIPAA-eligible services for messaging, authentication, data storage, queueing, and hosting, and an integration layer to the clinic’s existing scheduling system. The existing scheduling system remains the system of record for confirmed appointments, while the new application stores online booking requests, new-patient intake requests, reschedule requests, preset slot configuration, reminder scheduling state, audit records, and baseline reporting data.

The architecture prioritizes HIPAA-compliant PHI handling, staff-controlled scheduling, 24/7 patient request submission, and a unified staff daily schedule view. Production launch depends on HIPAA-eligible vendor selection, signed BAAs, validated scheduling-system integration capabilities, and clinic-approved PHI rules for reminders.

## 2. Architecture Drivers

- PRD-FR-001 — Patients must verify with a one-time code by email or SMS before submitting booking or reschedule requests.
- PRD-FR-002 — Existing patients must submit booking requests from staff-managed preset appointment slots.
- PRD-FR-003 — New patients must submit intake requests that remain pending until staff approval.
- PRD-FR-004 — Patients must request rescheduling only up to 24 hours before an appointment and staff must approve changes.
- PRD-FR-005 — Front desk schedulers must approve or reject pending booking, intake, and reschedule requests.
- PRD-FR-006 — Confirmed appointments must receive email and SMS reminders 72 hours and 24 hours before the appointment.
- PRD-FR-007 — Front desk schedulers and providers must see confirmed appointments plus pending requests in a daily schedule view.
- PRD-FR-008 — The product must integrate with the clinic’s existing scheduling system.
- PRD-FR-009 — The system must track after-hours booking submissions for baseline reporting.
- PRD-NFR-001 — PHI must be handled with HIPAA-eligible services, BAAs, and encryption in transit and at rest.
- PRD-NFR-002 — Patient booking and reschedule submission must be available 24/7 except planned maintenance.
- PRD-NFR-003 — The system must maintain audit records for request submission, staff decisions, reminder failures, and scheduling-system update failures.
- PRD-NFR-004 — Patient-facing flows should support keyboard and screen reader interactions.
- CON-001 — The clinic is a small healthcare business and must comply with HIPAA.
- CON-002 — Production launch depends on HIPAA-compliant vendor agreements and BAAs being in place.
- CON-003 — Online booking must use preset appointment slots managed by staff.
- CON-004 — New-patient intake and reschedule requests require staff approval.
- CON-005 — Reschedule requests are allowed only up to 24 hours before the appointment.
- CON-006 — The product must integrate with the clinic’s existing scheduling system.
- CON-007 — Clinical triage and medical advice routing are out of scope for the first release.

## 3. Architecture Principles

1. Keep the existing scheduling system as the system of record for confirmed appointments.
2. Minimize PHI exposure in application logs, reminders, metrics, traces, and third-party services.
3. Reuse established repository and platform patterns where appropriate.
4. Keep presentation, orchestration, domain, integration, and persistence concerns separated.
5. Prefer managed HIPAA-eligible services with BAAs over self-managed infrastructure for a small-business operating model.
6. Make staff approval and auditability explicit for all schedule-changing actions.

## 4. System Context

### 4.1 Actors

- Existing Patient — Verifies by email or SMS, views preset slots, submits booking requests, and requests rescheduling.
- New Patient — Verifies by email or SMS and submits intake requests for staff approval.
- Front Desk Scheduler — Reviews daily schedule, manages pending requests, approves or rejects requests, and follows up on integration or reminder failures.
- Provider — Views confirmed appointments and relevant pending schedule activity for the day.
- Compliance Owner — Defines HIPAA retention, PHI handling, vendor BAA readiness, and reminder content rules.
- Clinic Operations Owner — Defines office hours, appointment types, preset slot rules, and launch slot configuration.
- Existing Scheduling System — Supplies confirmed appointments and receives approved booking or reschedule updates when integration supports it.
- Managed Messaging Services — Send OTP, email reminders, and SMS reminders under HIPAA-compliant terms or clinic-approved content restrictions.
- Managed Identity Provider — Authenticates staff with MFA and role-based access.

### 4.2 External Systems

| ID | System | Purpose | Interface | Owner |
|---|---|---|---|---|
| EXT-001 | Existing Scheduling System | System of record for confirmed appointments; source for daily schedule; target for approved booking and reschedule changes | REST/Event/File/TBD | Clinic / Scheduling Vendor TBD |
| EXT-002 | HIPAA-Eligible Email Service | Sends OTP emails and appointment reminders when email contact is available | API | Vendor TBD / Compliance Owner |
| EXT-003 | HIPAA-Eligible SMS Service | Sends OTP SMS and appointment reminders when SMS contact is available | API | Vendor TBD / Compliance Owner |
| EXT-004 | Managed Identity Provider | Provides staff authentication, MFA, and role claims | OIDC/SAML/TBD | Vendor TBD / Technical Lead |
| EXT-005 | HIPAA-Eligible Cloud Platform | Hosts application, database, queue, worker services, secrets, logs, metrics, traces, and release artifacts | Managed services | Vendor TBD / Technical Lead |

## 5. Technology Stack

| Layer | Technology | Version/Constraint | Rationale |
|---|---|---|---|
| Frontend | React with TypeScript | Exact version TBD by implementation team; must support accessible patient flows | Enables custom patient and staff workflows with typed UI implementation |
| Backend | Node.js with TypeScript | Exact runtime/framework TBD by implementation team | Aligns frontend and backend language, supports API orchestration, integrations, and worker services |
| Data | Managed relational database | HIPAA-eligible cloud environment with encryption at rest and BAA | Fits structured requests, audit records, slot rules, reporting, and relational integrity needs |
| Queue/Workers | Managed queue plus worker service | HIPAA-eligible hosting environment with BAA | Supports reminders, retries, integration failure handling, and asynchronous processing |
| Authentication | Managed identity provider for staff; OTP by email/SMS for patients | Staff MFA required; patient OTP required before request submission | Supports staff role-based access and lightweight patient verification |
| Messaging | HIPAA-eligible email and SMS providers | BAA required unless reminder content avoids PHI under clinic-approved rules | Required for OTP and 72-hour/24-hour reminders |
| Hosting | HIPAA-eligible cloud platform | AWS, Azure, Google Cloud, or another eligible provider TBD during vendor selection | Cloud preference is intentionally open until vendor selection and BAAs are confirmed |
| Observability | Centralized logs, metrics, tracing, and alerting | PHI redaction required | Supports auditability, availability validation, incident response, and operational monitoring |
| CI/CD and IaC | Automated pipeline and infrastructure as code | Tools TBD by implementation team | Enforces build, lint, test, security scan, SBOM, and repeatable environment deployment controls |

## 6. Logical Architecture

### 6.1 Components

#### COMP-001 — Patient Web Application

- **Responsibility:** Presents patient verification, existing-patient booking, new-patient intake, and reschedule request flows.
- **Inputs:** Patient contact method, OTP, patient-entered request details, selected preset slot, existing appointment reference where applicable.
- **Outputs:** Verified booking, intake, or reschedule request submissions to the backend API.
- **Dependencies:** COMP-003, COMP-004, COMP-005.
- **Repository Area:** `apps/patient-web` (assumption)

#### COMP-002 — Staff Web Application

- **Responsibility:** Presents daily schedule, pending requests, request approval/rejection, and operational failure visibility for schedulers and providers.
- **Inputs:** Staff identity, selected date, pending request actions, approval or rejection decisions.
- **Outputs:** Approval/rejection commands, schedule queries, and operational follow-up actions.
- **Dependencies:** COMP-003, COMP-006, EXT-004.
- **Repository Area:** `apps/staff-web` (assumption)

#### COMP-003 — Scheduling Backend API

- **Responsibility:** Orchestrates patient requests, staff actions, validation, authorization, audit writes, reporting, and integration calls.
- **Inputs:** Patient submissions, OTP verification results, staff commands, schedule queries, report queries.
- **Outputs:** Request records, audit records, integration commands, reminder scheduling commands, schedule and reporting responses.
- **Dependencies:** COMP-004, COMP-005, COMP-006, COMP-007, COMP-008, EXT-001, EXT-004.
- **Repository Area:** `services/api` (assumption)

#### COMP-004 — Patient Verification Service

- **Responsibility:** Issues and validates one-time codes sent by email or SMS before patient request submission.
- **Inputs:** Patient contact method, OTP request, OTP validation attempt.
- **Outputs:** OTP messages through messaging providers and verification result for request submission.
- **Dependencies:** EXT-002, EXT-003, COMP-007.
- **Repository Area:** `services/api/modules/verification` (assumption)

#### COMP-005 — Request and Slot Domain Service

- **Responsibility:** Applies booking, intake, reschedule, preset slot, 24-hour cutoff, office-hours classification, and request status rules.
- **Inputs:** Preset slot configuration, appointment references, patient request data, clinic office hours.
- **Outputs:** Pending request records, status transitions, after-hours classification, validation errors.
- **Dependencies:** COMP-007, COMP-009.
- **Repository Area:** `services/api/modules/requests` (assumption)

#### COMP-006 — Scheduling Integration Adapter

- **Responsibility:** Reads confirmed appointments and writes approved booking or reschedule changes using the integration method supported by the existing scheduling system.
- **Inputs:** Schedule read requests, approved booking commands, approved reschedule commands.
- **Outputs:** Confirmed appointment data, scheduling update results, integration failure records.
- **Dependencies:** EXT-001, COMP-007, COMP-009.
- **Repository Area:** `services/api/modules/scheduling-integration` (assumption)

#### COMP-007 — Managed Relational Database

- **Responsibility:** Persists online requests, preset slots, OTP metadata, reminder scheduling state, audit records, office hours, and baseline reporting data.
- **Inputs:** Application writes from backend and worker services.
- **Outputs:** Transactional data, query results, reports, and audit evidence.
- **Dependencies:** EXT-005.
- **Repository Area:** `infra/database` (assumption)

#### COMP-008 — Reminder Worker

- **Responsibility:** Schedules and dispatches 72-hour and 24-hour email and SMS reminders only for confirmed appointments.
- **Inputs:** Confirmed appointment events or polling results, reminder scheduling state, contact-channel availability.
- **Outputs:** Reminder send requests, delivery status, delivery failure audit records.
- **Dependencies:** COMP-006, COMP-007, COMP-009, EXT-002, EXT-003.
- **Repository Area:** `services/workers/reminders` (assumption)

#### COMP-009 — Audit and Observability Service

- **Responsibility:** Records immutable audit events and emits PHI-redacted logs, metrics, traces, and alerts.
- **Inputs:** Request submissions, staff actions, reminder failures, scheduling integration failures, system health events.
- **Outputs:** Audit records, structured logs, metrics, traces, alerts, and operational records for planned maintenance or outages.
- **Dependencies:** COMP-007, EXT-005.
- **Repository Area:** `services/api/modules/audit` (assumption)

#### COMP-010 — CI/CD and Infrastructure Automation

- **Responsibility:** Builds, tests, scans, packages, deploys, rolls back, produces SBOMs, and provisions environments through infrastructure as code.
- **Inputs:** Source code, infrastructure definitions, environment configuration, release approvals.
- **Outputs:** Deployed releases, release artifacts, SBOMs, scan results, rollback events, provisioned infrastructure.
- **Dependencies:** EXT-005.
- **Repository Area:** `.github/workflows` and `infra` (assumption)

### 6.2 Data Flow

1. Patient Web Application sends contact method and OTP request to Patient Verification Service.
2. Patient Verification Service sends OTP through the HIPAA-eligible email or SMS provider and stores verification metadata.
3. Patient Web Application sends OTP and request data to Scheduling Backend API.
4. Scheduling Backend API validates OTP, booking/reschedule rules, preset slot availability, and 24-hour cutoff through Request and Slot Domain Service.
5. Request and Slot Domain Service stores pending booking, intake, or reschedule requests in the Managed Relational Database and records audit events.
6. Staff Web Application retrieves confirmed appointments from Scheduling Backend API, which obtains them through the Scheduling Integration Adapter and combines them with pending requests from the database.
7. Staff Web Application sends approval or rejection decisions to Scheduling Backend API.
8. Scheduling Backend API records staff decision audit events and, for approvals, invokes Scheduling Integration Adapter to update the existing scheduling system where supported.
9. Scheduling Integration Adapter records success or failure; failures are audited and surfaced for staff follow-up.
10. Reminder Worker identifies confirmed appointments, schedules 72-hour and 24-hour reminder events, and sends email and SMS reminders through managed messaging providers.
11. Audit and Observability Service records reminder failures, scheduling failures, planned maintenance, unplanned outages, metrics, logs, traces, and alerts with PHI redaction.
12. Reporting queries count after-hours booking submissions using stored submission timestamps and clinic-defined office hours.

## 7. Architecture Decisions

### ARD-DEC-001 — Hybrid Custom Application With Managed HIPAA-Eligible Services

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-003, PRD-FR-004, PRD-FR-005, PRD-FR-006, PRD-FR-007, PRD-FR-008, PRD-NFR-001, PRD-NFR-002, PRD-NFR-003

**Decision**

Build a custom patient and staff experience using React/TypeScript and a Node.js/TypeScript backend, while using managed HIPAA-eligible services for hosting, data, messaging, authentication, queueing, observability, and secrets.

**Rationale**

The clinic needs workflows tailored to staff-managed preset slots, staff approval, after-hours baseline tracking, and daily schedule visibility, while a small healthcare business should minimize operational burden by using managed eligible services under BAAs.

**Consequences**

- Enables precise workflow control for patient and staff experiences.
- Requires vendor selection and BAA readiness before production PHI processing.
- Requires integration design once the existing scheduling system capabilities are confirmed.

**Alternatives Considered**

- Buy/configure a scheduling platform only — not selected because the PRD requires custom staff approval, reporting, and integration behaviors that may not fit a single vendor module.
- Fully custom self-managed infrastructure — not selected because it increases operational and HIPAA control burden for a small business.
- Extend existing scheduling system only — not selected because module availability and capabilities are not confirmed.

### ARD-DEC-002 — Existing Scheduling System Remains Confirmed Appointment System of Record

- **Status:** Accepted
- **Related Requirements:** PRD-FR-005, PRD-FR-006, PRD-FR-007, PRD-FR-008, PRD-NFR-003

**Decision**

The existing scheduling system remains the system of record for confirmed appointments. The new product stores pending requests, preset slots, audit records, reminder state, and reporting data, and reflects approved changes into the scheduling system through the available integration method.

**Rationale**

The PRD requires integration with the existing scheduling system and staff continuity, while avoiding duplicate systems of record for confirmed appointments.

**Consequences**

- Reduces duplicate confirmed appointment data ownership.
- Makes integration availability and reliability critical to staff workflows.
- Requires failure handling when approved changes cannot be reflected in the scheduling system.

**Alternatives Considered**

- New product as system of record — not selected because it conflicts with the existing scheduling dependency.
- Dual-write co-equal records — not selected because reconciliation risk is too high for scheduling and PHI workflows.

### ARD-DEC-003 — Managed Relational Database for Requests, Slots, Audit, and Reminder State

- **Status:** Accepted
- **Related Requirements:** PRD-FR-002, PRD-FR-003, PRD-FR-004, PRD-FR-005, PRD-FR-006, PRD-FR-009, PRD-NFR-001, PRD-NFR-003

**Decision**

Use a managed relational database in a HIPAA-eligible cloud environment for online requests, audit records, preset slots, office hours, OTP metadata, reminder scheduling state, and baseline reporting data.

**Rationale**

The domain has structured entities, status transitions, relational links between requests and slots, and audit/reporting requirements that fit relational persistence.

**Consequences**

- Supports transactional request and audit workflows.
- Requires encryption at rest, backups, retention policy enforcement, and access controls.
- Requires schema migration discipline through CI/CD.

**Alternatives Considered**

- Managed document database — not selected because relational integrity is more important than document flexibility for first release.
- Store all persistent data in the existing scheduling system — not selected because the scheduling system is not confirmed to support request, audit, OTP, and reminder state.

### ARD-DEC-004 — Staff Authentication Through Managed Identity Provider With MFA and RBAC

- **Status:** Accepted
- **Related Requirements:** PRD-FR-005, PRD-FR-007, PRD-NFR-001, PRD-NFR-003

**Decision**

Authenticate staff through a managed identity provider with MFA and role-based access for front desk schedulers and providers.

**Rationale**

Staff access includes PHI and schedule-changing actions; managed identity with MFA and RBAC reduces implementation risk and supports HIPAA-aligned access control.

**Consequences**

- Requires identity provider selection and staff role mapping.
- Supports stronger access enforcement and audit attribution.
- Adds dependency on identity provider availability.

**Alternatives Considered**

- Application-local email/password — not selected because it increases credential-management burden and weakens MFA governance.
- Reuse scheduling-system login — not selected because SSO capability is unknown.

### ARD-DEC-005 — Managed Queue and Worker Service for Asynchronous Processing

- **Status:** Accepted
- **Related Requirements:** PRD-FR-006, PRD-FR-008, PRD-NFR-003

**Decision**

Use a managed queue plus worker service in the HIPAA-eligible hosting environment for reminders, integration retries, and audit-safe background processing.

**Rationale**

Reminder dispatch, retryable integration operations, and failure handling are asynchronous concerns that should not block interactive patient or staff requests.

**Consequences**

- Improves resilience of reminder and integration workflows.
- Requires idempotency keys and retry policies for background jobs.
- Requires monitoring for stuck queues, failed jobs, and delayed reminders.

**Alternatives Considered**

- Database polling only — not selected because queues provide clearer retry, visibility, and scaling semantics.
- Vendor workflow engine only — not selected because vendor choice is not settled and application control is needed.

### ARD-DEC-006 — Centralized Observability With PHI Redaction

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001, PRD-NFR-002, PRD-NFR-003

**Decision**

Implement centralized structured logs, metrics, traces, and alerts with explicit PHI redaction and sensitive-data restrictions.

**Rationale**

The application must distinguish planned maintenance from outages, audit security-relevant events, and support operational response without exposing PHI in telemetry.

**Consequences**

- Improves incident response and availability validation.
- Requires careful logging standards and automated reviews to prevent PHI leakage.
- Requires alert routing and operational ownership before launch.

**Alternatives Considered**

- Basic logs and uptime monitoring only — not selected because it is insufficient for auditability and integration/reminder failure visibility.
- Vendor dashboards only — not selected because cross-component workflows need consolidated visibility.

### ARD-DEC-007 — Automated CI/CD, Security Gates, SBOM, and Infrastructure as Code

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001, PRD-NFR-003

**Decision**

Use automated CI/CD with build, lint, test, dependency/CVE scan gates, SBOM generation for every production release, and infrastructure as code for environment provisioning.

**Rationale**

HIPAA-sensitive software needs repeatable deployments, traceable release artifacts, vulnerability management, and controlled infrastructure changes.

**Consequences**

- Blocks production deployment on critical and high vulnerabilities.
- Requires maintenance of pipeline and infrastructure code.
- Produces evidence for release and security review.

**Alternatives Considered**

- Manual deployment checklist — not selected because it is less repeatable and weakens evidence collection.
- Vendor deployment tooling only — not selected because hosting vendor remains TBD and architecture requires consistent controls.

### ARD-DEC-008 — No Bulk Appointment Migration for First Release

- **Status:** Accepted
- **Related Requirements:** PRD-FR-007, PRD-FR-008

**Decision**

Do not bulk-migrate historical or upcoming confirmed appointments into the new database for first release. Confirmed appointments are read from the existing scheduling system, and preset slots are configured before launch.

**Rationale**

The PRD requires integration with the existing scheduling system rather than replacing it; avoiding migration reduces launch risk.

**Consequences**

- Simplifies first release data readiness.
- Requires reliable schedule read access from the existing system.
- Preset slot configuration remains a launch dependency.

**Alternatives Considered**

- Migrate all historical appointments — not selected because it increases PHI movement and is not required by the PRD.
- Migrate upcoming appointments only — not selected because confirmed appointment system-of-record remains external.

## 8. API and Integration Architecture

### ARD-API-001 — Patient Verification API

- **Type:** REST
- **Operation/Topic:** `POST /api/patient-verifications`, `POST /api/patient-verifications/{verificationId}/confirm`
- **Provider:** COMP-003 / COMP-004
- **Consumers:** COMP-001
- **Authentication:** Public endpoint with rate limiting, abuse detection, and OTP validation; no staff session required
- **Versioning:** URI versioning when breaking changes are introduced; initial internal route version may be unversioned until external exposure is finalized

**Request/Input Contract**

OTP request includes contact channel, contact destination, purpose (`booking`, `intake`, or `reschedule`), and request context where needed. OTP confirmation includes verification identifier and code.

**Response/Output Contract**

OTP request returns verification identifier, expiration metadata, and masked destination. OTP confirmation returns verification status, allowed purpose, and short-lived verification token or server-side verification reference.

**Error Handling**

Invalid, expired, or excessive OTP attempts return non-PHI error messages and do not submit requests. Rate-limited requests return retry guidance. OTP operations are audited without storing plaintext codes.

### ARD-API-002 — Patient Request Submission API

- **Type:** REST
- **Operation/Topic:** `POST /api/booking-requests`, `POST /api/intake-requests`, `POST /api/reschedule-requests`
- **Provider:** COMP-003 / COMP-005
- **Consumers:** COMP-001
- **Authentication:** Valid OTP verification reference required before submission
- **Versioning:** URI versioning when breaking changes are introduced; request schemas versioned in API documentation

**Request/Input Contract**

Booking request includes verified patient context, selected preset slot, contact details, appointment type, and submission timestamp. Intake request includes verified contact details and intake fields approved for launch. Reschedule request includes verified patient context, existing appointment reference, requested preset slot, and submission timestamp.

**Response/Output Contract**

Successful submission returns request identifier, status `pending`, request type, submitted timestamp, and patient-safe next-step message. Validation failures return field-level errors.

**Error Handling**

Unavailable slots, missing verification, expired verification, invalid request data, or reschedule attempts less than 24 hours before appointment return explicit patient-facing errors. Duplicate submissions should use idempotency keys where supported to prevent accidental duplicates.

### ARD-API-003 — Staff Schedule and Request API

- **Type:** REST
- **Operation/Topic:** `GET /api/staff/schedule?date=YYYY-MM-DD`, `GET /api/staff/requests`, `POST /api/staff/requests/{requestId}/approve`, `POST /api/staff/requests/{requestId}/reject`
- **Provider:** COMP-003
- **Consumers:** COMP-002
- **Authentication:** Managed identity provider session with MFA and RBAC
- **Versioning:** URI versioning when breaking changes are introduced

**Request/Input Contract**

Schedule query includes selected date and optional provider filter. Request review queries include status and request type filters. Approval/rejection commands include request identifier, staff decision, optional staff note, and idempotency key.

**Response/Output Contract**

Schedule response includes confirmed appointments from the scheduling system and pending requests from application storage, with confirmed and pending states clearly distinguished. Approval/rejection responses include updated request status and scheduling integration outcome if applicable.

**Error Handling**

Unauthorized users receive access-denied responses. Integration failures do not silently mark scheduling updates complete; failures are audited and returned to the staff UI for follow-up. Approval commands must be idempotent to avoid duplicate scheduling updates.

### ARD-API-004 — Scheduling System Integration Interface

- **Type:** REST | Event | File
- **Operation/Topic:** `TBD by scheduling system capabilities`
- **Provider:** EXT-001 through COMP-006
- **Consumers:** COMP-003, COMP-008
- **Authentication:** TBD by scheduling-system capability; must use secure service credentials or equivalent
- **Versioning:** Adapter-specific versioning and contract tests once interface is selected

**Request/Input Contract**

Read operations request confirmed appointments by date and optional provider. Write operations submit approved booking or reschedule changes with appointment type, slot, patient reference, and staff approval metadata as supported by the scheduling system.

**Response/Output Contract**

Read operations return confirmed appointment data needed for staff daily view and reminder scheduling. Write operations return success/failure status, scheduling-system appointment identifier, and error details safe for staff follow-up.

**Error Handling**

Transient failures are retried through managed queue workers where safe. Non-retryable failures are audited, surfaced to staff, and do not mark requests as fully reflected in the scheduling system. Idempotency keys or equivalent deduplication must be used for write operations where supported.

### ARD-API-005 — Messaging Provider Interface

- **Type:** REST
- **Operation/Topic:** `POST /provider/email/send`, `POST /provider/sms/send` or vendor equivalent
- **Provider:** EXT-002, EXT-003 through COMP-004 and COMP-008
- **Consumers:** COMP-004, COMP-008
- **Authentication:** Vendor API credentials stored in managed secrets service
- **Versioning:** Vendor API version pinned where supported; adapter wraps vendor changes

**Request/Input Contract**

OTP messages include recipient, code or provider template reference, expiration text, and purpose. Reminder messages include recipient, approved reminder content, appointment timing context, and channel-specific template metadata.

**Response/Output Contract**

Provider returns accepted, delivered, failed, or provider-specific status. Application stores reminder scheduling and failure state, not unnecessary provider payloads containing PHI.

**Error Handling**

Delivery failures are recorded for staff review or operational follow-up. Retry behavior depends on provider error classification. PHI in message content is limited by Compliance Owner-approved rules.

### ARD-API-006 — Reporting API

- **Type:** REST
- **Operation/Topic:** `GET /api/staff/reports/after-hours-submissions`
- **Provider:** COMP-003 / COMP-005
- **Consumers:** COMP-002
- **Authentication:** Managed identity provider session with staff role authorized for reporting
- **Versioning:** URI versioning when breaking changes are introduced

**Request/Input Contract**

Report query includes date range and optional filters such as request type or provider where available.

**Response/Output Contract**

Report response includes counts of booking submissions classified as inside or outside clinic office hours for the selected period.

**Error Handling**

Invalid date ranges return validation errors. Missing office-hours configuration returns a configuration error and audit/observability event because PRD-FR-009 depends on clinic-defined office hours.

## 9. Data Architecture

### DATA-001 — Patient Verification Record

- **Purpose:** Tracks OTP issuance, verification status, purpose, expiration, and verification method before request submission.
- **System of Record:** Managed Relational Database
- **Classification:** Confidential
- **Retention:** Retain according to clinic HIPAA record-retention policy; exact period TBD by Compliance Owner.
- **Key Relationships:** Linked to DATA-002, DATA-003, or DATA-004 when a verified request is submitted.

### DATA-002 — Booking Request

- **Purpose:** Captures existing-patient online booking requests from preset slots pending staff approval.
- **System of Record:** Managed Relational Database until approved; confirmed appointment system of record is Existing Scheduling System.
- **Classification:** Restricted
- **Retention:** Retain according to clinic HIPAA record-retention policy; exact period TBD by Compliance Owner.
- **Key Relationships:** References DATA-005 preset slot, DATA-006 audit events, DATA-008 office-hours classification, and EXT-001 appointment after approval.

### DATA-003 — New-Patient Intake Request

- **Purpose:** Captures prospective patient intake requests pending staff approval before scheduling.
- **System of Record:** Managed Relational Database until approved; confirmed appointment system of record is Existing Scheduling System.
- **Classification:** Restricted
- **Retention:** Retain according to clinic HIPAA record-retention policy; exact period TBD by Compliance Owner.
- **Key Relationships:** References DATA-006 audit events and, after approval/scheduling, EXT-001 appointment where applicable.

### DATA-004 — Reschedule Request

- **Purpose:** Captures patient requests to reschedule an existing appointment to a preset slot up to 24 hours before appointment time.
- **System of Record:** Managed Relational Database for request workflow; Existing Scheduling System for confirmed appointment state.
- **Classification:** Restricted
- **Retention:** Retain according to clinic HIPAA record-retention policy; exact period TBD by Compliance Owner.
- **Key Relationships:** References existing appointment in EXT-001, DATA-005 preset slot, and DATA-006 audit events.

### DATA-005 — Preset Appointment Slot

- **Purpose:** Stores staff-managed online availability slots and appointment-type/provider rules used for booking and reschedule requests.
- **System of Record:** Managed Relational Database unless the selected scheduling-system integration later exposes authoritative slot management.
- **Classification:** Internal
- **Retention:** Retain active slots and historical slot references according to clinic operational policy; exact period TBD by Clinic Operations Owner.
- **Key Relationships:** Referenced by DATA-002 and DATA-004.

### DATA-006 — Audit Event

- **Purpose:** Records patient request submissions, staff approvals/rejections, reminder failures, scheduling-system update failures, planned maintenance, and relevant security events.
- **System of Record:** Managed Relational Database and centralized observability/audit store where applicable.
- **Classification:** Confidential or Restricted when event metadata contains PHI.
- **Retention:** Retain according to clinic HIPAA record-retention policy; exact period TBD by Compliance Owner.
- **Key Relationships:** References staff user, patient request, reminder event, scheduling integration operation, or maintenance/outage event.

### DATA-007 — Reminder Event

- **Purpose:** Tracks scheduled, sent, delivered, and failed email/SMS reminders for confirmed appointments.
- **System of Record:** Managed Relational Database for application scheduling state; messaging provider for delivery status.
- **Classification:** Restricted when linked to appointment or patient contact details.
- **Retention:** Retain according to clinic HIPAA record-retention policy; exact period TBD by Compliance Owner.
- **Key Relationships:** References EXT-001 confirmed appointment, DATA-006 audit events, EXT-002 email messages, and EXT-003 SMS messages.

### DATA-008 — Clinic Office Hours

- **Purpose:** Defines clinic office hours used to classify booking requests as inside or outside office hours.
- **System of Record:** Managed Relational Database
- **Classification:** Internal
- **Retention:** Retain current configuration and historical effective periods required for reporting accuracy; exact period TBD by Clinic Operations Owner.
- **Key Relationships:** Used by DATA-002 and ARD-API-006 reporting.

### DATA-009 — Confirmed Appointment Reference

- **Purpose:** Represents confirmed appointment data read from the existing scheduling system for staff daily view, reminder scheduling, and reschedule validation.
- **System of Record:** Existing Scheduling System
- **Classification:** Restricted
- **Retention:** No bulk migration for first release; cached data, if used, must follow clinic HIPAA retention and minimization policy.
- **Key Relationships:** Referenced by DATA-004 and DATA-007.

### Data Migration

- **Required:** No bulk migration for first release.
- **Approach:** Read confirmed appointments from the existing scheduling system and configure preset appointment slots before launch. Any cache of confirmed appointment data must be minimized, encrypted, and refreshed according to the selected integration design.

## 10. Security Architecture

### ARD-SEC-001 — HIPAA-Compliant PHI Handling

- **Related Requirements:** PRD-NFR-001, PRD-NFR-003

The implementation must:

- Use HIPAA-eligible vendors and execute BAAs before production PHI processing.
- Encrypt PHI in transit using HTTPS/TLS.
- Encrypt PHI at rest in production storage, backups, queues, and logs where applicable.
- Limit PHI access to authorized staff roles through RBAC.
- Apply least-privilege access to cloud resources, databases, messaging providers, and secrets.
- Restrict reminder content to Compliance Owner-approved PHI rules.
- Maintain audit records for patient submissions, staff decisions, reminder failures, and scheduling integration failures.

The implementation must not:

- Process production PHI with any vendor lacking a required BAA.
- Store plaintext OTP codes.
- Log PHI in application logs, metrics, traces, CI/CD output, or error messages.
- Send clinical triage or medical advice routing content because it is out of scope.
- Mark scheduling updates complete when integration failures occur.

### ARD-SEC-002 — Staff Identity, MFA, and Authorization

- **Related Requirements:** PRD-FR-005, PRD-FR-007, PRD-NFR-001, PRD-NFR-003

The implementation must:

- Authenticate staff through a managed identity provider with MFA.
- Enforce roles for front desk schedulers and providers.
- Restrict approval/rejection capabilities to authorized scheduler roles.
- Record staff user identifiers in audit events for approval and rejection actions.

The implementation must not:

- Permit unauthenticated staff access to PHI or schedule data.
- Allow provider-only roles to approve or reject requests unless explicitly authorized by clinic policy.

### ARD-SEC-003 — Patient OTP Verification and Abuse Controls

- **Related Requirements:** PRD-FR-001, PRD-NFR-001, PRD-NFR-003

The implementation must:

- Require valid OTP verification before booking, intake, or reschedule submission.
- Rate-limit OTP requests and validation attempts.
- Expire OTPs and verification references.
- Record verification method used for submitted requests.

The implementation must not:

- Allow booking or reschedule submission after invalid or expired OTP verification.
- Reveal whether a patient exists through overly specific verification errors.

### Security Controls

- **Authentication:** Staff authenticate with managed identity provider, MFA, and secure sessions; patients verify with OTP by email or SMS before request submission.
- **Authorization:** RBAC separates front desk scheduler approval rights from provider schedule-view access; least-privilege service roles are required for backend, workers, database, queue, and messaging providers.
- **Secrets Management:** Store database credentials, API keys, signing secrets, and provider credentials in a managed secrets service; never store secrets in source control or CI/CD logs.
- **Encryption:** HTTPS/TLS for data in transit; managed encryption at rest for relational database, backups, queues, object storage, logs, and release artifacts where PHI or sensitive data may exist.
- **Audit Logging:** Record patient submissions, staff approvals/rejections, reminder failures, scheduling integration failures, planned maintenance, unplanned outages, authentication/security events, and privileged configuration changes.
- **Dependency/CVE Scanning:** Critical and high vulnerabilities block production deployment; medium vulnerabilities require tracked remediation.
- **SBOM:** Generate an SBOM for every production release and retain it with release artifacts.

## 11. Reliability, Performance, and Scalability

### ARD-OPS-001 — Patient Submission Availability

- **Target:** Patient-facing booking and reschedule submission achieve 99.5% monthly uptime excluding planned maintenance.
- **Validation Method:** Synthetic uptime checks, production availability metrics, incident records, and planned-maintenance records that distinguish planned from unplanned downtime.

### ARD-OPS-002 — Core Interaction Performance

- **Target:** 95% of booking, reschedule, and daily schedule page/API interactions complete within 2 seconds under expected clinic load.
- **Validation Method:** Load testing before production release, application performance monitoring, frontend timing metrics, and backend API latency metrics.

### ARD-OPS-003 — Reminder Scheduling Coverage

- **Target:** Confirmed appointments have reminder events scheduled for 72 hours and 24 hours before appointment time when contact channels are available.
- **Validation Method:** Automated integration tests and production reporting that compare confirmed appointments against reminder events.

### ARD-OPS-004 — Integration Failure Visibility

- **Target:** 100% of scheduling-system update failures are recorded as audit events and surfaced for staff or operational follow-up.
- **Validation Method:** Contract/integration failure tests, audit-log verification, and alert/dashboard review.

### ARD-OPS-005 — Asynchronous Processing Reliability

- **Target:** Reminder jobs and retryable integration jobs are processed through managed queue workers with monitored failure and dead-letter paths.
- **Validation Method:** Queue health checks, dead-letter queue alerts, retry behavior tests, and operational dashboard review.

## 12. Observability

- **Logging:** Use centralized structured logs for backend, frontend errors, workers, integration adapters, deployment events, and security-relevant events; logs must redact PHI and secrets.
- **Metrics:** Track availability, API latency, error rate, OTP send/validation outcomes, request submissions, after-hours classifications, pending request aging, reminder scheduling/sending/failure counts, queue depth, worker failures, and scheduling integration failures.
- **Tracing:** Trace patient submission, staff approval, scheduling integration, and reminder dispatch workflows across API, database, queue, worker, and external provider boundaries with PHI-safe identifiers.
- **Alerting:** Alert on patient submission outage, elevated API errors, queue backlog, reminder failures, scheduling integration failures, authentication anomalies, database health, and planned-maintenance/outage state changes.
- **Sensitive Data Restrictions:** Do not emit PHI, plaintext OTPs, secrets, full message bodies, raw patient notes, or unnecessary appointment details into logs, metrics, traces, alerts, CI/CD output, or third-party observability tools.

## 13. Deployment Architecture

- **Environments:** Separate dev, test, staging, and production environments.
- **Deployment Model:** Automated deployment of frontend, backend API, worker services, database migrations, and infrastructure changes through controlled pipelines.
- **CI/CD:** Automated CI/CD with build, lint, unit tests, integration tests, contract tests, security scans, SBOM generation, environment promotion, and release artifact retention.
- **Configuration Management:** Environment-specific configuration managed through infrastructure as code and managed secrets; PHI-related configuration changes must be auditable.
- **Rollback:** Automated rollback to the last known good release for application components; database migration rollback or forward-fix strategy must be defined per migration before deployment.
- **Infrastructure as Code:** Required for cloud resources, network boundaries, database, queues, secrets, observability, identity integration configuration where supported, and deployment permissions.

## 14. Testing Architecture

Required test levels:

- Unit tests
- Integration tests
- Contract tests
- End-to-end tests
- Security tests
- Performance tests, where applicable

Minimum quality expectations:

- **Coverage Threshold:** 80% line coverage for backend and frontend application code.
- **Blocking Security Severity:** Critical and high vulnerabilities block production deployment; medium vulnerabilities require tracked remediation.
- **Required Build/Lint Checks:** TypeScript type checks, frontend build, backend build, linting, unit tests, integration tests for request workflows and persistence, contract tests for scheduling and messaging adapters once interfaces are selected, accessibility checks for patient flows, security scans, SBOM generation, and performance validation for core interactions.

## 15. Architecture Constraints

- ARC-CON-001 — The clinic is a small healthcare business and must comply with HIPAA.
- ARC-CON-002 — Production PHI processing requires HIPAA-eligible vendors and BAAs before launch.
- ARC-CON-003 — Online booking must use preset appointment slots managed by staff.
- ARC-CON-004 — New-patient intake and reschedule requests must require staff approval.
- ARC-CON-005 — Reschedule requests must be allowed only up to 24 hours before the appointment.
- ARC-CON-006 — The architecture must integrate with the clinic’s existing scheduling system.
- ARC-CON-007 — Clinical triage and medical advice routing are out of scope for the first release.
- ARC-CON-008 — Patient booking and reschedule submission must be available 24/7 except planned maintenance.
- ARC-CON-009 — Confirmed appointments remain in the existing scheduling system as the system of record.
- ARC-CON-010 — Cloud, messaging, identity, and integration vendor selections are TBD and must preserve HIPAA eligibility, BAA readiness, encryption, auditability, and security scanning requirements.
- ARC-CON-011 — Reminder content must follow Compliance Owner-approved PHI rules before launch.
- ARC-CON-012 — The first release must not bulk-migrate confirmed appointment history into the new database.

## 16. Risks and Trade-offs

| ID | Risk/Trade-off | Impact | Mitigation |
|---|---|---|---|
| ARC-RISK-001 | HIPAA-eligible vendor agreements or BAAs are delayed | Production launch cannot proceed with PHI | Make BAA readiness a release gate and select vendors early |
| ARC-RISK-002 | Existing scheduling system has limited or no real-time API capability | Staff workflows may require manual or file-based fallback | Keep integration method TBD until validated; isolate integration behind COMP-006 adapter |
| ARC-RISK-003 | Reminder content may expose PHI through email or SMS | Compliance and patient privacy risk | Require Compliance Owner-approved reminder content rules and HIPAA-eligible messaging vendors or non-PHI content |
| ARC-RISK-004 | Staff approval queues may delay patient benefit | Online requests may not convert to confirmed appointments quickly | Monitor pending request aging and surface pending requests in daily schedule view |
| ARC-RISK-005 | Preset slots may become stale or misconfigured | Patients may request unavailable or inappropriate appointment slots | Assign clinic operations ownership and validate slot configuration before launch |
| ARC-RISK-006 | Cloud/vendor choice remains open | Architecture implementation cannot be fully specified yet | Define vendor-neutral controls and capture vendor selection as an open architecture question |
| ARC-RISK-007 | PHI leakage through logs, traces, alerts, or CI output | Compliance and security incident risk | Enforce PHI redaction, sensitive-data restrictions, code review, and security tests |
| ARC-RISK-008 | Strict security gates may slow release cadence | Delayed deployments when vulnerabilities are found | Track medium findings, block only critical/high, and remediate dependencies continuously |

## 17. Open Architecture Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| ARC-OQ-001 | Which named person owns the architecture for ARD approval and ongoing decisions? | Business Sponsor | Open | TBD |
| ARC-OQ-002 | Which HIPAA-eligible cloud platform will host the application, data, queue, observability, and secrets? | Product Owner / Technical Lead / Compliance Owner | Open | TBD |
| ARC-OQ-003 | Which existing scheduling system will be integrated, and which interface methods are available for reads and writes? | Product Owner / Technical Lead | Open | TBD |
| ARC-OQ-004 | Which HIPAA-eligible email, SMS, identity, hosting, storage, queue, and observability vendors will be selected, and which require BAAs? | Product Owner / Compliance Owner | Open | TBD |
| ARC-OQ-005 | What exact PHI may appear in email and SMS OTP and reminder content? | Compliance Owner | Open | TBD |
| ARC-OQ-006 | What exact retention periods apply to online requests, audit events, OTP metadata, reminder events, slot history, cached appointment data, and observability data? | Compliance Owner | Open | TBD |
| ARC-OQ-007 | Which appointment types, providers, preset slot rules, and office hours are included at launch? | Clinic Operations Owner | Open | TBD |
| ARC-OQ-008 | Which staff roles beyond front desk schedulers and providers, if any, require access? | Product Owner | Open | TBD |
| ARC-OQ-009 | Which repository paths and framework conventions will the implementation team use? | Technical Lead | Open | TBD |

## 18. Feature Mapping

| Feature | Relevant Components | Decision IDs | Interface IDs | Security IDs |
|---|---|---|---|---|
| FEAT-001 | COMP-001, COMP-003, COMP-004, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-003 | ARD-API-001, ARD-API-005 | ARD-SEC-001, ARD-SEC-003 |
| FEAT-002 | COMP-001, COMP-003, COMP-005, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-003 | ARD-API-002 | ARD-SEC-001, ARD-SEC-003 |
| FEAT-003 | COMP-001, COMP-003, COMP-005, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-003 | ARD-API-002 | ARD-SEC-001, ARD-SEC-003 |
| FEAT-004 | COMP-001, COMP-003, COMP-005, COMP-006, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003 | ARD-API-002, ARD-API-004 | ARD-SEC-001, ARD-SEC-003 |
| FEAT-005 | COMP-002, COMP-003, COMP-005, COMP-006, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-004 | ARD-API-003, ARD-API-004 | ARD-SEC-001, ARD-SEC-002 |
| FEAT-006 | COMP-003, COMP-006, COMP-007, COMP-008, COMP-009 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-005 | ARD-API-004, ARD-API-005 | ARD-SEC-001 |
| FEAT-007 | COMP-002, COMP-003, COMP-006, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-004 | ARD-API-003, ARD-API-004 | ARD-SEC-001, ARD-SEC-002 |
| FEAT-008 | COMP-003, COMP-006, COMP-007, COMP-009 | ARD-DEC-002, ARD-DEC-005, ARD-DEC-008 | ARD-API-004 | ARD-SEC-001, ARD-SEC-002 |
| FEAT-009 | COMP-003, COMP-005, COMP-007, COMP-009 | ARD-DEC-003, ARD-DEC-006 | ARD-API-006 | ARD-SEC-001, ARD-SEC-002 |

## 19. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Solution Architect | TBD | Pending | TBD |
| Security Reviewer | TBD | Pending | TBD |
| Engineering Lead | TBD | Pending | TBD |