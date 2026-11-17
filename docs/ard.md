# Architecture Requirements Document (ARD)

> Replace all `<PLACEHOLDER>` values. Every material architecture decision or constraint should have a stable ID that feature specifications can reference.

## Document Metadata

- **Product/Application:** Online Appointment Scheduling System
- **Document ID:** ARD-APPT-SCHED
- **Version:** 0.1
- **Status:** Draft
- **Architecture Owner:** TBD
- **Contributors:** TBD
- **Last Updated:** 2026-11-17
- **Reference PRD:** `PRD.md`

---

## 1. Architecture Summary

The Online Appointment Scheduling System will use a hybrid architecture: HIPAA-eligible scheduling SaaS provides the patient, staff, and provider responsive web workflows, while lightweight custom integration services coordinate EHR/practice-management scheduling integration, conflict-safe slot holds, reminder orchestration, audit export, and observability.

The EHR/practice-management system is the system of record for appointments, provider schedules, patient identifiers, appointment status, and appointment history. The scheduling SaaS and integration services may store only minimum necessary operational data for booking/rescheduling, configuration, delivery tracking, and audit evidence. Operational PHI outside the EHR is retained for 90 days unless required longer by a vendor workflow; audit logs are retained for 6 years.

The architecture prioritizes HIPAA compliance, BAAs for vendors handling PHI, encryption in transit and at rest, role-based access controls, auditability, real-time scheduling consistency, patient self-service usability, WCAG 2.1 AA accessibility, and small-business operational simplicity.

## 2. Architecture Drivers

- PRD-FR-001 — Patients must be able to book configured standard non-urgent appointments online through a mobile-responsive web portal.
- PRD-FR-002 — Patients must be able to reschedule eligible appointments until a staff-configured cutoff.
- PRD-FR-003 — Booking changes require one-time-code verification by SMS or email.
- PRD-FR-004 — Availability must use real-time two-way EHR/practice-management sync and slot holds or equivalent conflict prevention.
- PRD-FR-005 — Booking and rescheduling failures must release holds where applicable and provide clear recovery guidance.
- PRD-FR-006 — Appointment reminders must be sent by email and SMS using configurable timing with 72-hour and 24-hour defaults.
- PRD-FR-007 — Staff need a daily schedule view with appointment, patient contact, reason, status, and provider grouping.
- PRD-FR-008 — Providers should be able to view their day’s appointments.
- PRD-FR-009 — Authorized staff must configure appointment eligibility, availability rules, cutoffs, and reminders.
- PRD-NFR-001 — HIPAA-aligned PHI handling requires HIPAA-eligible vendors, BAAs where required, encryption, access controls, and audit logs.
- PRD-NFR-002 — Patient-facing and staff-facing web flows must meet WCAG 2.1 AA.
- PRD-NFR-003 — The first release must be mobile-responsive web only and require no native app.
- PRD-NFR-004 — The system must prevent confirmed double bookings for the same provider, appointment time, and resource.
- PRD-NFR-005 — Audit evidence must cover booking, rescheduling, reminders, configuration changes, and schedule access.
- CON-001 — The solution must avoid unnecessary operational complexity for a small business.
- CON-002 — The clinic operates in healthcare and must satisfy HIPAA requirements for PHI handling.
- CON-003 — The first release must be mobile-responsive web only and must not include native mobile apps.
- CON-004 — The first release must support only configured standard non-urgent appointment types.
- CON-005 — The first release must not include urgent/emergency scheduling, billing/payments, clinical messaging, or telehealth.
- CON-006 — Scheduling availability must use real-time two-way EHR/practice-management integration with slot holds or equivalent conflict prevention.
- CON-007 — Reminder channels in the first release are email and SMS only.
- CON-008 — Rollout must start as a pilot with one provider before broader clinic launch.

## 3. Architecture Principles

1. Treat the EHR/practice-management system as the appointment system of record.
2. Minimize PHI replication outside the EHR and retain only minimum necessary operational data.
3. Reuse established repository and platform patterns where appropriate.
4. Keep presentation, orchestration, domain, integration, and persistence concerns separated.
5. Prefer HIPAA-eligible managed SaaS and cloud capabilities over custom infrastructure where they reduce risk and operational burden.
6. Make every scheduling write conflict-safe, auditable, and recoverable.
7. Make security, accessibility, integration contracts, and auditability release gates.

## 4. System Context

### 4.1 Actors

- Patient — Books and reschedules configured standard non-urgent appointments through the mobile-responsive scheduling portal.
- Front-Desk Scheduling Staff — Views the day’s schedule and configures eligible appointment types, provider availability rules, rescheduling cutoffs, and reminder timing.
- Provider — Views their day’s appointments.
- Clinic Administrator — Manages user access, vendor approvals, and operational rollout decisions.
- EHR/Practice-Management System — Owns appointment records, patient scheduling data, provider schedules, and scheduling status.
- Scheduling SaaS — Hosts patient, staff, and provider scheduling web workflows and stores minimum necessary operational copies.
- Integration Services — Orchestrate real-time availability, slot holds, booking writes, rescheduling writes, reminders, audit export, and observability.
- Messaging Vendors — Send SMS and email reminders and one-time codes under HIPAA-eligible arrangements where PHI is involved.
- Clinic SSO/Identity Provider — Authenticates staff and providers with MFA and provides role/group claims.
- CI/CD Platform — Builds, tests, scans, produces SBOM, and deploys custom integration services.

### 4.2 External Systems

| ID | System | Purpose | Interface | Owner |
|---|---|---|---|---|
| EXT-001 | EHR/Practice-Management System | System of record for appointments, patient scheduling data, provider schedules, and appointment status | REST/FHIR/vendor API | Clinic / EHR vendor |
| EXT-002 | HIPAA-Eligible Scheduling SaaS | Patient, staff, and provider scheduling workflows; operational scheduling configuration | REST/Event/Web | Scheduling SaaS vendor |
| EXT-003 | Clinic SSO/Identity Provider | Staff/provider authentication, MFA, role/group claims | OIDC/SAML | Clinic IT / IdP vendor |
| EXT-004 | SMS Vendor | One-time codes and appointment reminder SMS delivery | REST/Event | Messaging vendor |
| EXT-005 | Email Vendor | One-time codes and appointment reminder email delivery | REST/Event/SMTP API | Messaging vendor |
| EXT-006 | HIPAA-Eligible Cloud Runtime | Runs custom integration services and observability components | Container/Serverless | Cloud provider / Engineering |
| EXT-007 | Central Observability Platform | HIPAA-safe logs, metrics, traces, alerts, and audit exports | Event/REST | Engineering / Observability vendor |
| EXT-008 | Clinic Website | Entry point or embedded placement for patient scheduling portal | Web | Clinic / Website vendor |

## 5. Technology Stack

| Layer | Technology | Version/Constraint | Rationale |
|---|---|---|---|
| Frontend | Scheduling SaaS-hosted responsive web screens configured or embedded on clinic website | Must meet WCAG 2.1 AA and mobile-responsive requirements; no native app | Reduces custom UI operational burden while supporting patient/staff/provider workflows |
| Backend | Lightweight custom integration services | Platform-agnostic container or serverless services on a HIPAA-eligible cloud runtime; exact cloud TBD | Provides controlled EHR, messaging, audit, and observability orchestration without overbuilding a full custom product |
| Data | EHR/practice-management system plus minimum necessary SaaS/cloud operational stores | EHR is system of record; operational PHI outside EHR retained 90 days; audit logs retained 6 years | Preserves scheduling authority and minimizes PHI replication |
| Hosting | HIPAA-eligible scheduling SaaS and HIPAA-eligible cloud runtime | BAAs required where vendors handle PHI; encryption in transit and at rest required | Satisfies HIPAA posture while keeping operations manageable for a small clinic |
| Identity | Clinic SSO/OIDC with MFA for staff/providers; patient OTP by SMS/email | Staff/provider RBAC required; patient full accounts out of scope | Balances security with patient self-service simplicity |
| Messaging | HIPAA-eligible SMS and email vendors with BAAs where required | Email and SMS only in first release | Meets reminder and OTP requirements while excluding voice calls |
| CI/CD | Git-based CI/CD pipeline | Runs tests, lint/build, scans, SBOM, IaC validation, and deployment | Provides repeatable release evidence and rollback support |
| Infrastructure | Infrastructure as Code | Tool TBD by selected cloud/runtime | Ensures reproducible environments and controlled PHI access boundaries |

## 6. Logical Architecture

### 6.1 Components

#### COMP-001 — Scheduling SaaS Web Experience

- **Responsibility:** Host/configure mobile-responsive patient booking, patient rescheduling, staff daily schedule view, provider schedule view, and staff scheduling configuration screens.
- **Inputs:** Patient selections, OTP verification status, staff/provider authenticated sessions, scheduling configuration, availability data, appointment updates.
- **Outputs:** Booking/rescheduling requests, schedule views, configuration changes, reminder settings, audit events.
- **Dependencies:** EXT-002, COMP-002, COMP-003, COMP-004, COMP-005, EXT-003, EXT-008.
- **Repository Area:** `TBD: SaaS configuration repository or vendor admin export`.

#### COMP-002 — API Gateway and Integration Edge

- **Responsibility:** Expose controlled APIs/webhooks between the scheduling SaaS, EHR/practice-management system, messaging vendors, and internal integration services.
- **Inputs:** SaaS webhooks/API calls, EHR callbacks, messaging delivery callbacks, authenticated service requests.
- **Outputs:** Routed service requests, normalized events, authentication/authorization decisions, rate-limited integration traffic.
- **Dependencies:** COMP-003, COMP-004, COMP-005, COMP-006, EXT-001, EXT-002, EXT-004, EXT-005.
- **Repository Area:** `services/integration-edge/`.

#### COMP-003 — Scheduling Orchestration Service

- **Responsibility:** Coordinate availability reads, slot holds, booking confirmation, rescheduling, hold release, failure recovery, and idempotent scheduling writes.
- **Inputs:** Appointment type, provider/resource, desired time slot, patient verification context, rescheduling cutoff rules, EHR availability responses.
- **Outputs:** Held slot references, confirmed appointment references, rescheduled appointment references, patient-facing errors, audit events.
- **Dependencies:** COMP-002, COMP-006, COMP-008, EXT-001, EXT-002.
- **Repository Area:** `services/scheduling-orchestration/`.

#### COMP-004 — Reminder and OTP Orchestration Service

- **Responsibility:** Coordinate one-time-code delivery and appointment reminder scheduling through HIPAA-eligible SMS/email vendors or scheduling SaaS capabilities.
- **Inputs:** Patient contact method, verification request, appointment reminder schedule, delivery callbacks, reminder configuration.
- **Outputs:** OTP send requests, reminder send requests, delivery status events, failed-delivery audit evidence.
- **Dependencies:** COMP-002, COMP-006, COMP-008, EXT-004, EXT-005.
- **Repository Area:** `services/messaging-orchestration/`.

#### COMP-005 — Schedule View and Configuration Adapter

- **Responsibility:** Normalize staff/provider schedule view data and staff configuration data between the SaaS workflows and EHR/practice-management system.
- **Inputs:** Staff/provider role claims, schedule date/provider filters, appointment type configuration, availability rules, cutoff rules, reminder timing.
- **Outputs:** Provider-grouped schedule data, configuration updates, access audit events.
- **Dependencies:** COMP-002, COMP-006, COMP-008, EXT-001, EXT-002, EXT-003.
- **Repository Area:** `services/schedule-config-adapter/`.

#### COMP-006 — EHR Integration Adapter

- **Responsibility:** Encapsulate EHR/practice-management integration using FHIR Scheduling APIs where supported and vendor-specific APIs where required for holds and booking writes.
- **Inputs:** Availability queries, hold requests, booking requests, rescheduling requests, appointment read requests, provider/resource mappings.
- **Outputs:** Normalized availability, hold confirmations, appointment IDs, conflict responses, EHR error mappings.
- **Dependencies:** EXT-001, COMP-008.
- **Repository Area:** `services/ehr-adapter/`.

#### COMP-007 — Operational Data Store

- **Responsibility:** Store minimum necessary operational data, idempotency keys, hold state, reminder scheduling state, delivery status, integration state, and non-authoritative appointment references.
- **Inputs:** Orchestration state, vendor callbacks, delivery status events, audit references.
- **Outputs:** State lookups, retry records, retention-limited operational records.
- **Dependencies:** COMP-003, COMP-004, COMP-005, COMP-008.
- **Repository Area:** `infra/operational-data-store/`.

#### COMP-008 — Audit and Compliance Event Service

- **Responsibility:** Capture immutable audit events for booking, rescheduling, reminder activity, configuration changes, schedule access, integration failures, and security-relevant events.
- **Inputs:** Audit events from SaaS, integration services, EHR integration, staff access, provider access, messaging delivery callbacks.
- **Outputs:** Centralized audit log records, audit exports, compliance review evidence.
- **Dependencies:** EXT-007, COMP-007.
- **Repository Area:** `services/audit-compliance/`.

#### COMP-009 — Observability and Alerting

- **Responsibility:** Provide HIPAA-safe logs, metrics, traces, dashboards, and alerts for custom integration services and imported vendor audit/status events.
- **Inputs:** Application logs, metrics, traces, SaaS/EHR/messaging status events, CI/CD deployment events.
- **Outputs:** Alerts, dashboards, incident diagnostics, uptime/performance evidence.
- **Dependencies:** EXT-007, all custom services.
- **Repository Area:** `infra/observability/`.

#### COMP-010 — CI/CD and Infrastructure Automation

- **Responsibility:** Build, test, scan, generate SBOM, validate IaC, deploy custom integration services, and support rollback across environments.
- **Inputs:** Source code, IaC definitions, SaaS configuration artifacts where exportable, test suites, security policies.
- **Outputs:** Deployable artifacts, SBOM, scan reports, test evidence, deployed infrastructure, rollback artifacts.
- **Dependencies:** EXT-006, EXT-007.
- **Repository Area:** `.github/workflows/`, `infra/`, `services/`.

### 6.2 Data Flow

1. Patient opens the scheduling portal from the clinic website through EXT-008 and COMP-001.
2. COMP-001 requests available configured standard non-urgent slots through COMP-002 and COMP-003.
3. COMP-003 asks COMP-006 to query EXT-001 for real-time availability using FHIR Scheduling APIs where supported or vendor APIs where required.
4. COMP-006 normalizes availability and returns current slots to COMP-003, which returns them to COMP-001.
5. Patient selects a slot and completes OTP verification through COMP-004 and EXT-004 or EXT-005.
6. COMP-003 places a temporary hold or equivalent conflict-prevention reservation through COMP-006 and EXT-001.
7. Patient confirms booking or rescheduling; COMP-003 performs an idempotent write through COMP-006 to EXT-001.
8. EXT-001 confirms the appointment or returns a conflict/error; COMP-003 releases holds on failure and returns clear recovery guidance through COMP-001.
9. COMP-004 schedules default or configured reminders through HIPAA-eligible email/SMS vendors or scheduling SaaS delivery functions.
10. Staff and providers authenticate through EXT-003 and access schedule/configuration screens through COMP-001.
11. COMP-005 retrieves provider-grouped daily schedule data from EXT-001 through COMP-006 and applies RBAC from EXT-003 claims.
12. COMP-008 records audit events for booking, rescheduling, reminders, configuration changes, schedule access, and integration failures.
13. COMP-009 receives HIPAA-safe logs, metrics, traces, and alerts from custom services and status/audit exports from vendors where available.

## 7. Architecture Decisions

### ARD-DEC-001 — Use Hybrid SaaS Plus Lightweight Integration Services

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-007, PRD-FR-008, PRD-FR-009, PRD-NFR-001, PRD-NFR-003

**Decision**

Use HIPAA-eligible scheduling SaaS for patient, staff, and provider scheduling workflows, supplemented by lightweight custom integration services for EHR integration, reminders, audit export, and observability.

**Rationale**

A small healthcare clinic needs HIPAA-ready scheduling capabilities without unnecessary custom UI and platform complexity. Custom services are still needed to normalize EHR integration, handle conflict-safe orchestration, and produce operational/audit evidence.

**Consequences**

- Reduces custom application surface area and time-to-pilot risk.
- Requires vendor due diligence, BAAs, SaaS configuration management, and contract clarity.
- Creates dependency on SaaS capabilities for accessibility, workflow configuration, and audit export.

**Alternatives Considered**

- Fully custom web application — Rejected because it increases operational burden and HIPAA implementation scope for a small clinic.
- EHR portal only — Not selected because the exact EHR capability is unsettled and may not satisfy all workflow and reminder needs.
- Scheduling SaaS only — Not selected because EHR integration, conflict handling, audit export, and observability need explicit architecture control.

### ARD-DEC-002 — Treat EHR/Practice-Management as Scheduling System of Record

- **Status:** Accepted
- **Related Requirements:** PRD-FR-004, PRD-FR-005, PRD-FR-007, PRD-FR-008, PRD-NFR-004

**Decision**

The EHR/practice-management system is the authoritative source for appointments, provider/resource schedules, patient scheduling identifiers, and appointment status.

**Rationale**

The PRD requires real-time two-way scheduling sync and double-booking prevention. Keeping appointment authority in the EHR avoids conflicting appointment histories and aligns online scheduling with existing clinic operations.

**Consequences**

- Online scheduling depends on EHR API availability and semantics.
- Custom services must handle EHR conflicts, retries, and error mapping.
- No migration of existing appointments is required for the first release.

**Alternatives Considered**

- Scheduling SaaS as system of record — Rejected because EHR remains the clinic operational scheduling authority.
- Dual system of record — Rejected because it increases reconciliation and double-booking risk.
- Custom database as system of record — Rejected because it duplicates clinical scheduling authority unnecessarily.

### ARD-DEC-003 — Use FHIR Scheduling APIs Where Supported With Vendor API Fallback

- **Status:** Accepted
- **Related Requirements:** PRD-FR-004, PRD-FR-005, PRD-NFR-004

**Decision**

Use FHIR Scheduling APIs where supported by the EHR/practice-management system, with vendor-specific API fallback for slot holds, booking writes, rescheduling writes, and conflict semantics not fully covered by FHIR.

**Rationale**

The exact EHR/practice-management system is not yet identified. FHIR provides a standards-aligned baseline, while vendor APIs may be required for real-time holds and confirmed writes.

**Consequences**

- Integration adapter must isolate EHR-specific behavior behind normalized contracts.
- Contract tests are required against sandbox or mocked EHR interfaces.
- Exact endpoint and resource mapping remain open until EHR selection/integration discovery.

**Alternatives Considered**

- Vendor-specific APIs only — Rejected as the default because it reduces portability before EHR identification.
- HL7 v2 only — Not selected because patient-facing real-time availability/holds typically need request/response APIs.
- Manual import/export — Rejected because it cannot satisfy real-time sync or double-booking prevention.

### ARD-DEC-004 — Use Clinic SSO/OIDC With MFA for Staff and Providers

- **Status:** Accepted
- **Related Requirements:** PRD-FR-007, PRD-FR-008, PRD-FR-009, PRD-NFR-001, PRD-NFR-005

**Decision**

Staff and provider access must use clinic SSO/OIDC or equivalent federated identity with MFA and role-based authorization.

**Rationale**

Staff and providers access appointment-related PHI and configuration functions. Federated identity centralizes account lifecycle, MFA enforcement, and role claims.

**Consequences**

- Scheduling SaaS and custom services must support SSO integration and role/group mapping.
- Break-glass or local fallback accounts, if any, must be documented and audited.
- Patient authentication remains OTP-based and separate from staff/provider SSO.

**Alternatives Considered**

- SaaS-native login only — Not selected because it fragments identity governance unless clinic SSO is unavailable.
- Username/password without SSO — Rejected due to weaker lifecycle and MFA governance.
- EHR embedded auth only — Not selected because scheduling SaaS workflows may not be embedded in the EHR.

### ARD-DEC-005 — Minimize PHI Outside the EHR and Retain Operational Data for 90 Days

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001, PRD-NFR-005

**Decision**

Store only minimum necessary operational PHI outside the EHR. Retain operational data for 90 days and retain audit logs for 6 years.

**Rationale**

The system must meet HIPAA expectations while supporting reminders, troubleshooting, and auditability. Limiting replicated PHI reduces breach impact and operational risk.

**Consequences**

- Data models must distinguish operational data from audit logs.
- Retention automation is required for operational stores.
- Audit logs must avoid unnecessary PHI while retaining enough event evidence.

**Alternatives Considered**

- Retain full appointment history in scheduling SaaS indefinitely — Rejected because it expands PHI footprint unnecessarily.
- Delete all operational data immediately after appointment completion — Not selected because it may weaken troubleshooting and delivery evidence.
- Store no PHI outside EHR — Not selected because OTP, reminders, and SaaS workflows require minimum necessary operational data.

### ARD-DEC-006 — Platform-Agnostic HIPAA-Eligible Container/Serverless Runtime

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001, PRD-NFR-005

**Decision**

Implement custom integration services as platform-agnostic containerized or serverless workloads deployable to a HIPAA-eligible cloud runtime. The exact cloud provider remains TBD.

**Rationale**

Cloud choice is unsettled. Defining runtime constraints preserves portability while keeping HIPAA controls, deployment automation, and observability explicit.

**Consequences**

- IaC must abstract provider-specific services where practical.
- Cloud-native choices must be revisited after provider selection.
- Open architecture question remains for final cloud/runtime selection.

**Alternatives Considered**

- Select AWS/Azure/GCP now — Deferred because the user selected platform-agnostic requirements.
- Vendor-managed integration only — Rejected because custom orchestration and observability controls are required.

### ARD-DEC-007 — Centralize HIPAA-Safe Observability and Audit Exports

- **Status:** Accepted
- **Related Requirements:** PRD-FR-005, PRD-FR-006, PRD-NFR-001, PRD-NFR-005

**Decision**

Use centralized HIPAA-safe logs, metrics, alerts, traces for integration services, and immutable audit event export from SaaS/EHR where available.

**Rationale**

The clinic needs evidence for incidents, delivery failures, access review, booking changes, and HIPAA-aligned auditability without exposing unnecessary PHI in logs.

**Consequences**

- Logging and tracing must be PHI-minimized by design.
- Vendors must be evaluated for audit export capabilities.
- Observability platform must be HIPAA-eligible where it receives PHI or PHI-adjacent events.

**Alternatives Considered**

- Vendor dashboards only — Rejected because they fragment evidence and slow incident response.
- Audit-only logging — Rejected because performance/reliability targets require operational telemetry.

## 8. API and Integration Architecture

### ARD-API-001 — Patient Scheduling Web Interface

- **Type:** REST
- **Operation/Topic:** `GET /schedule`, `POST /booking`, `POST /reschedule` or SaaS-equivalent operations
- **Provider:** COMP-001
- **Consumers:** Patient browsers through EXT-008
- **Authentication:** Patient OTP verification before booking changes
- **Versioning:** SaaS-managed web workflow versioning with configuration version records

**Request/Input Contract**

Patient-facing workflows collect appointment type, selected slot, patient identity/contact fields, OTP verification channel, OTP code, and booking/rescheduling confirmation inputs. Required fields are defined by SaaS configuration and clinic policy.

**Response/Output Contract**

The interface returns available slots, verification status, booking/rescheduling confirmation with appointment reference, or patient-facing recovery guidance when a slot is unavailable or a scheduling action fails.

**Error Handling**

Errors must be clear to patients, avoid exposing internal integration details, and direct patients to choose another time or call the clinic. Failed booking/rescheduling must not create duplicate or partially confirmed appointments.

### ARD-API-002 — Scheduling SaaS to Integration Services Interface

- **Type:** REST/Event
- **Operation/Topic:** `POST /availability/search`, `POST /slots/hold`, `POST /appointments/confirm`, `POST /appointments/reschedule`, `POST /holds/release`
- **Provider:** COMP-002
- **Consumers:** COMP-001 / EXT-002
- **Authentication:** OAuth 2.0/OIDC or vendor-equivalent scoped API credentials; mTLS where supported
- **Versioning:** URI or header-based API versioning; initial version `v1`

**Request/Input Contract**

Requests include correlation ID, idempotency key where mutating, appointment type, provider/resource preferences, requested date/time range, patient verification context, appointment reference for rescheduling, and minimum necessary patient contact/identifier data.

**Response/Output Contract**

Responses include normalized availability slots, hold token and expiration where supported, confirmed appointment reference, rescheduled appointment reference, conflict status, retryability flag, and user-safe error code.

**Error Handling**

Mutating operations must be idempotent by key. Conflicts return a non-retryable slot-unavailable response. Transient EHR/vendor failures may be retried by integration services using bounded retries. Holds are released on failure or expiration where supported.

### ARD-API-003 — EHR Scheduling Integration Interface

- **Type:** REST
- **Operation/Topic:** `FHIR Schedule/Slot/Appointment` operations where supported; vendor-specific hold and booking endpoints where required
- **Provider:** EXT-001
- **Consumers:** COMP-006
- **Authentication:** OAuth 2.0/OIDC or vendor-equivalent scoped API credentials; mTLS where supported
- **Versioning:** EHR/FHIR API version plus adapter contract version `v1`

**Request/Input Contract**

Requests include provider/resource identifiers, appointment type mappings, date/time ranges, patient scheduling identifiers, slot/appointment references, hold tokens where supported, and idempotency/correlation metadata where supported.

**Response/Output Contract**

Responses include normalized provider/resource availability, slot status, hold status, appointment ID, appointment status, conflict/error codes, and last-updated timestamps where available.

**Error Handling**

The adapter maps EHR-specific errors into normalized retryable, non-retryable, conflict, validation, authorization, and outage categories. Confirmed double booking must be prevented by hold or equivalent conflict prevention; if conflict occurs, patient recovery guidance is returned.

### ARD-API-004 — OTP and Reminder Messaging Interface

- **Type:** REST/Event
- **Operation/Topic:** `POST /messages/sms`, `POST /messages/email`, vendor delivery callback topics
- **Provider:** EXT-004 and EXT-005 through COMP-004
- **Consumers:** COMP-004, COMP-001
- **Authentication:** OAuth 2.0/OIDC or vendor-equivalent scoped API credentials; secrets stored in managed vault
- **Versioning:** Vendor API version plus internal message contract version `v1`

**Request/Input Contract**

Requests include message type, recipient contact, template ID, appointment reference, reminder scheduled time, OTP metadata, locale/time-zone where applicable, and correlation ID. Message content must follow minimum necessary PHI rules.

**Response/Output Contract**

Responses include vendor message ID, accepted/rejected status, error reason when rejected, and delivery callback events for sent, delivered, failed, or expired states where supported.

**Error Handling**

Transient messaging failures use bounded retries. Failed reminder or OTP delivery is captured for audit/operational follow-up. OTP failures block verification; reminder failures do not change appointment state.

### ARD-API-005 — Staff and Provider Identity Interface

- **Type:** REST
- **Operation/Topic:** `OIDC authorization_code flow` or SAML equivalent if required by SaaS
- **Provider:** EXT-003
- **Consumers:** COMP-001, COMP-002, custom services where needed
- **Authentication:** MFA-backed clinic SSO
- **Versioning:** IdP protocol version and role-claim mapping version `v1`

**Request/Input Contract**

Authentication requests include client ID, redirect URI, scopes, nonce/state, and requested claims. Authorization checks consume user identifier, MFA assurance where available, role/group claims, and session metadata.

**Response/Output Contract**

Identity responses include ID/access tokens or assertions, user identifier, role/group claims, token expiration, and MFA/session assurance metadata where available.

**Error Handling**

Authentication failures deny access without exposing sensitive details. Missing or insufficient roles deny schedule/configuration actions and generate audit events.

### ARD-API-006 — Audit and Observability Event Interface

- **Type:** Event
- **Operation/Topic:** `audit.booking`, `audit.reschedule`, `audit.reminder`, `audit.configuration`, `audit.schedule_access`, `ops.integration_failure`
- **Provider:** COMP-008 and COMP-009
- **Consumers:** EXT-007, compliance reviewers, engineering operations
- **Authentication:** Scoped service credentials; mTLS where supported
- **Versioning:** Event schema version `v1`

**Request/Input Contract**

Events include event type, timestamp, actor or verified patient context, appointment reference, correlation ID, source system, outcome, and minimum necessary metadata. PHI must be minimized and excluded unless required for audit purpose.

**Response/Output Contract**

The observability/audit platform acknowledges receipt, stores immutable records where supported, and exposes searchable audit evidence according to access controls.

**Error Handling**

Audit event delivery uses durable buffering and retry where available. Audit delivery failures create operational alerts and must not silently discard events.

## 9. Data Architecture

### DATA-001 — Appointment

- **Purpose:** Represent scheduled patient visit time, provider/resource, appointment type, status, and appointment reference.
- **System of Record:** EHR/practice-management system.
- **Classification:** Restricted.
- **Retention:** Retained according to EHR/practice-management retention policy; operational copies outside EHR retained for 90 days.
- **Key Relationships:** Linked to DATA-002 Patient Scheduling Identity, DATA-003 Provider/Resource, DATA-004 Appointment Type, DATA-006 Reminder, and DATA-007 Audit Event.

### DATA-002 — Patient Scheduling Identity and Contact

- **Purpose:** Verify patient booking/rescheduling context and deliver OTP/reminders.
- **System of Record:** EHR/practice-management system for patient identity; messaging/scheduling systems may store minimum necessary operational contact data.
- **Classification:** Restricted.
- **Retention:** EHR retention in system of record; operational copies outside EHR retained for 90 days; OTP secrets retained only until expiration plus operational troubleshooting window where applicable.
- **Key Relationships:** Linked to DATA-001 Appointment, DATA-006 Reminder, and DATA-007 Audit Event.

### DATA-003 — Provider and Resource Schedule

- **Purpose:** Represent provider availability, resources, and scheduling calendars.
- **System of Record:** EHR/practice-management system.
- **Classification:** Confidential.
- **Retention:** EHR/practice-management retention policy; operational cache outside EHR retained for no longer than needed for real-time scheduling and not more than 90 days.
- **Key Relationships:** Determines availability for DATA-001 Appointment and DATA-004 Appointment Type.

### DATA-004 — Online Scheduling Configuration

- **Purpose:** Define eligible appointment types, provider availability rules, rescheduling cutoffs, reminder timing, and online scheduling controls.
- **System of Record:** Scheduling SaaS or EHR/practice-management system depending on selected vendor capability; final authority mapping TBD.
- **Classification:** Confidential.
- **Retention:** Current and historical configuration changes retained according to audit policy; audit events retained 6 years.
- **Key Relationships:** Controls DATA-001 Appointment availability, DATA-005 Slot Hold, and DATA-006 Reminder generation.

### DATA-005 — Slot Hold / Conflict Prevention Record

- **Purpose:** Temporarily reserve or protect a selected appointment slot during checkout.
- **System of Record:** EHR/practice-management system where native holds exist; otherwise integration service operational store for equivalent conflict-prevention state.
- **Classification:** Restricted.
- **Retention:** Retained until hold expiration/release plus operational troubleshooting period, not more than 90 days outside EHR.
- **Key Relationships:** Associated with DATA-001 Appointment candidate, DATA-003 Provider/Resource, and patient verification context.

### DATA-006 — Reminder and OTP Delivery Record

- **Purpose:** Track scheduled reminders, OTP requests, vendor message IDs, and delivery/failure outcomes.
- **System of Record:** Reminder/OTP orchestration service and messaging vendor delivery logs; appointment authority remains EHR.
- **Classification:** Restricted.
- **Retention:** Operational delivery records retained 90 days; audit summary retained 6 years.
- **Key Relationships:** Linked to DATA-001 Appointment, DATA-002 Patient Scheduling Identity, and DATA-007 Audit Event.

### DATA-007 — Audit Event

- **Purpose:** Provide evidence for booking, rescheduling, reminder activity, configuration changes, schedule access, and security-relevant events.
- **System of Record:** Audit and compliance event store / centralized observability platform.
- **Classification:** Restricted when containing PHI or patient-related references; otherwise Confidential.
- **Retention:** 6 years.
- **Key Relationships:** References DATA-001 Appointment, actor/user context, source component, event outcome, and correlation ID.

### Data Migration

- **Required:** No.
- **Approach:** Existing appointments remain in the EHR/practice-management system. The scheduling SaaS and integration services read/sync existing appointments from the EHR as the system of record; no one-time migration of existing appointments is planned for the first release.

## 10. Security Architecture

### ARD-SEC-001 — HIPAA Vendor and PHI Handling Constraint

- **Related Requirements:** PRD-NFR-001, PRD-NFR-005

The implementation must:

- Use HIPAA-eligible vendors with BAAs where vendors handle PHI.
- Encrypt PHI in transit and at rest across SaaS, cloud services, EHR integrations, messaging vendors, logs, backups, and operational stores.
- Store only minimum necessary PHI outside the EHR/practice-management system.
- Retain operational PHI outside the EHR for 90 days and audit logs for 6 years.
- Capture audit logs for appointment booking, rescheduling, reminder activity, staff configuration changes, and schedule access.

The implementation must not:

- Use vendors handling PHI without required BAA coverage.
- Store full appointment history indefinitely outside the EHR unless explicitly approved through compliance review.
- Include PHI in logs, traces, analytics, or error messages beyond minimum necessary audit/operations needs.

### ARD-SEC-002 — Staff and Provider Access Control

- **Related Requirements:** PRD-FR-007, PRD-FR-008, PRD-FR-009, PRD-NFR-001, PRD-NFR-005

The implementation must:

- Authenticate staff and providers through clinic SSO/OIDC or equivalent federated identity with MFA.
- Enforce role-based access controls for staff schedule view, provider schedule view, and scheduling configuration changes.
- Record schedule access and configuration changes in audit logs.
- Deny configuration actions to unauthorized users.

The implementation must not:

- Permit staff/provider access without MFA-backed authentication.
- Allow provider users to view other providers’ schedules as their own schedule outside configured permissions.
- Allow unaudited configuration changes to online scheduling rules.

### ARD-SEC-003 — Service-to-Service Security

- **Related Requirements:** PRD-NFR-001, PRD-NFR-004, PRD-NFR-005

The implementation must:

- Use OAuth 2.0/OIDC or vendor-equivalent scoped API credentials for service-to-service calls.
- Use mTLS where supported by vendors and runtime platforms.
- Store secrets in a managed vault or equivalent cloud secret manager.
- Rotate credentials according to vendor/cloud capability and clinic security policy.
- Use least-privilege scopes for EHR, SaaS, messaging, observability, and deployment credentials.

The implementation must not:

- Store static API keys directly in source code, container images, frontend artifacts, or unencrypted configuration.
- Use broad administrative credentials for routine integration calls.
- Reuse production credentials in lower environments.

### ARD-SEC-004 — Secure SDLC, Scanning, and SBOM

- **Related Requirements:** PRD-NFR-001, PRD-NFR-005

The implementation must:

- Run dependency, container, IaC, and secret scanning in CI/CD for custom integration services and infrastructure.
- Block release on high and critical vulnerabilities unless formally risk-accepted outside this ARD.
- Generate and retain an SBOM for custom services and deployable artifacts.
- Run security tests for authentication, authorization, audit logging, and integration boundary controls.

The implementation must not:

- Deploy custom service changes that bypass required CI/CD security gates.
- Suppress high or critical findings without documented remediation or risk acceptance.
- Include secrets in SBOM, build logs, or deployment logs.

### Security Controls

- **Authentication:** Patients use OTP by SMS or email before booking changes; staff/providers use clinic SSO/OIDC with MFA.
- **Authorization:** RBAC controls staff schedule access, provider schedule access, and scheduling configuration; service accounts use least-privilege scopes.
- **Secrets Management:** Managed vault or equivalent cloud secret manager; no secrets in source code, frontend artifacts, or plaintext deployment configuration.
- **Encryption:** TLS for all network traffic; encryption at rest for SaaS stores, cloud databases, audit stores, logs, backups, and vendor systems handling PHI.
- **Audit Logging:** Required for booking creation, rescheduling, reminder activity, staff configuration changes, staff/provider schedule access, authentication/authorization failures, and integration failures.
- **Dependency/CVE Scanning:** CI/CD blocks high and critical vulnerabilities for custom services, containers, and IaC unless risk-accepted.
- **SBOM:** Required for custom integration services and deployable artifacts.

## 11. Reliability, Performance, and Scalability

### ARD-OPS-001 — Booking Confirmation Performance

- **Target:** 95% of booking and rescheduling confirmation attempts complete within 5 seconds, excluding patient time spent entering OTP or form data.
- **Validation Method:** E2E performance tests against staging with EHR sandbox/mocks and production monitoring of confirmation latency percentiles.

### ARD-OPS-002 — Staff Schedule View Performance

- **Target:** 95% of daily schedule view requests load within 3 seconds for pilot provider scope and expected clinic daily schedule volume.
- **Validation Method:** E2E tests using seeded schedule data and production monitoring of schedule view latency percentiles.

### ARD-OPS-003 — Monthly Availability

- **Target:** 99.5% monthly availability for patient booking/rescheduling entry points and custom integration services, subject to upstream vendor availability.
- **Validation Method:** Synthetic uptime checks, cloud service metrics, SaaS status monitoring, and monthly availability reports.

### ARD-OPS-004 — Double-Booking Prevention

- **Target:** 0 confirmed double bookings for the same provider, appointment time, and resource in acceptance conflict tests.
- **Validation Method:** Concurrent booking integration tests, EHR sandbox conflict tests, idempotency tests, and pilot monitoring of conflict events.

### ARD-OPS-005 — Reminder Delivery Evidence

- **Target:** 100% of reminder send attempts produce a delivery status event or vendor-accepted/error event in audit/operational logs.
- **Validation Method:** Messaging integration tests, vendor callback tests, and audit log sampling.

### ARD-OPS-006 — Audit Event Capture

- **Target:** 100% of booking, rescheduling, reminder activity, configuration change, and schedule access events emit an audit event in integration and acceptance tests.
- **Validation Method:** Automated audit assertion tests and compliance review of sampled audit records.

## 12. Observability

- **Logging:** Custom services must emit structured, correlation-ID-based logs with PHI minimized or redacted; logs must not include OTP codes, full message bodies, or unnecessary patient details.
- **Metrics:** Track booking confirmation latency, availability lookup latency, EHR API latency/error rate, hold success/release/failure counts, booking conflict counts, schedule view latency, OTP/reminder send success/failure counts, and audit delivery failures.
- **Tracing:** Trace custom integration service calls across SaaS, EHR adapter, messaging orchestration, and audit event emission using correlation IDs while avoiding PHI in span attributes.
- **Alerting:** Alert on integration outage, EHR API error spikes, booking failure spikes, hold release failures, audit delivery failures, message delivery callback failures, availability SLA breaches, and CI/CD security gate failures.
- **Sensitive Data Restrictions:** PHI, OTP codes, message bodies, secrets, access tokens, and full patient contact details must not be intentionally logged or traced unless explicitly required for audit and protected by access controls.

## 13. Deployment Architecture

- **Environments:** Local development, dev, test, staging, and production, with separated PHI access controls and no production credentials in lower environments.
- **Deployment Model:** Scheduling SaaS workflows are configured in vendor-managed environments; custom integration services deploy as platform-agnostic container/serverless workloads on a HIPAA-eligible cloud runtime.
- **CI/CD:** Git-based pipeline runs dependency install, build, lint, unit tests, integration tests, contract tests, E2E tests, accessibility tests, performance tests where applicable, security scans, SBOM generation, IaC validation, deployment, and smoke tests.
- **Configuration Management:** SaaS configuration is versioned where exportable; service configuration uses environment-specific managed configuration and secrets from a managed vault.
- **Rollback:** Redeploy previous known-good custom service artifact and IaC state; revert SaaS configuration using vendor-supported version/export/import process or documented manual rollback runbook.
- **Infrastructure as Code:** Required for custom cloud infrastructure; exact tool TBD by selected cloud/runtime.

## 14. Testing Architecture

Required test levels:

- Unit tests
- Integration tests
- Contract tests
- End-to-end tests
- Security tests
- Performance tests, where applicable

Minimum quality expectations:

- **Coverage Threshold:** 80% line coverage for custom domain, integration orchestration, and security/audit utility code.
- **Blocking Security Severity:** High and Critical vulnerabilities block release unless formally risk-accepted outside this ARD.
- **Required Build/Lint Checks:** Build, lint, type/static checks appropriate to implementation language, unit tests, integration tests, EHR/messaging/SaaS contract tests, E2E booking/rescheduling/schedule-view flows, WCAG 2.1 AA automated accessibility scans with no critical violations, performance tests for booking confirmation and schedule view targets, dependency/container/IaC/secret scans, SBOM generation, IaC validation, deployment smoke tests, and audit event assertion tests.

## 15. Architecture Constraints

- ARC-CON-001 — The solution must avoid unnecessary operational complexity for a small business, per CON-001.
- ARC-CON-002 — The solution must satisfy HIPAA requirements for PHI handling, per CON-002 and PRD-NFR-001.
- ARC-CON-003 — Vendors handling PHI must be HIPAA-eligible and have BAAs where required, per PRD-NFR-001.
- ARC-CON-004 — PHI must be encrypted in transit and at rest, per PRD-NFR-001.
- ARC-CON-005 — Staff and provider access must use role-based access control and audit logs, per PRD-NFR-001 and PRD-NFR-005.
- ARC-CON-006 — Patient experience must be mobile-responsive web only with no native app, per CON-003 and PRD-NFR-003.
- ARC-CON-007 — The first release supports only staff-configured standard non-urgent appointment types, per CON-004.
- ARC-CON-008 — Urgent/emergency scheduling, billing/payments, clinical messaging, telehealth, and native apps are out of scope, per CON-005.
- ARC-CON-009 — Scheduling availability must use real-time two-way EHR/practice-management integration with slot holds or equivalent conflict prevention, per CON-006 and PRD-NFR-004.
- ARC-CON-010 — Reminder channels are email and SMS only in the first release, per CON-007.
- ARC-CON-011 — Rollout must start with a one-provider pilot before broader launch, per CON-008.
- ARC-CON-012 — Patient-facing and staff-facing web flows must meet WCAG 2.1 AA, per PRD-NFR-002.
- ARC-CON-013 — Existing appointments must not be migrated into a new system of record for the first release.

## 16. Risks and Trade-offs

| ID | Risk/Trade-off | Impact | Mitigation |
|---|---|---|---|
| ARC-RISK-001 | EHR/practice-management system may not support real-time two-way sync or slot holds through usable APIs | High impact to double-booking prevention and patient trust | Validate EHR capabilities early; isolate EHR behavior in COMP-006; define approved equivalent conflict-prevention mechanism |
| ARC-RISK-002 | Scheduling SaaS may not expose sufficient audit export, SSO, accessibility, or configuration versioning | High impact to HIPAA evidence and release readiness | Include audit/SSO/WCAG/export requirements in vendor selection and acceptance criteria |
| ARC-RISK-003 | Hybrid architecture introduces vendor dependency and integration complexity | Medium operational complexity for small clinic | Keep custom services narrow, use managed cloud capabilities, and automate deployment/monitoring |
| ARC-RISK-004 | SMS/email vendors may handle PHI or message content inconsistently | High compliance and privacy impact | Use HIPAA-eligible vendors with BAAs; minimize PHI in templates; audit delivery events |
| ARC-RISK-005 | Centralized observability can accidentally collect PHI | High compliance impact | Redaction, PHI-safe logging standards, access controls, and log review tests |
| ARC-RISK-006 | Platform-agnostic cloud decision defers concrete IaC and managed service choices | Medium schedule/design risk | Keep provider-specific decisions in Open Architecture Questions and finalize before implementation |
| ARC-RISK-007 | Staff workflow adoption may be limited if SaaS schedule view does not match operations | Medium adoption impact | Pilot with one provider, gather feedback, and tune configuration before broader rollout |

## 17. Open Architecture Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| ARC-OQ-001 | Which exact EHR/practice-management system and integration interface will be used for real-time two-way scheduling sync? | Technical Owner | Open | TBD |
| ARC-OQ-002 | Which HIPAA-eligible cloud provider/runtime will host custom integration services? | Architecture Owner | Open | TBD |
| ARC-OQ-003 | Which scheduling SaaS vendor will host patient, staff, and provider web workflows? | Product Owner / Architecture Owner | Open | TBD |
| ARC-OQ-004 | Which SMS and email vendors will be used for OTP and reminders, and what BAA coverage applies? | Security Reviewer | Open | TBD |
| ARC-OQ-005 | Which exact appointment types, provider availability rules, and rescheduling cutoff values apply to the one-provider pilot? | Clinic Operations | Open | TBD |
| ARC-OQ-006 | Which system is authoritative for online scheduling configuration if the EHR and SaaS both support rule storage? | Architecture Owner | Open | TBD |
| ARC-OQ-007 | Who is the named Architecture Owner and who are named ARD contributors? | Product Owner | Open | TBD |
| ARC-OQ-008 | What is the target date for the one-provider pilot and broader rollout decision? | Product Owner | Open | TBD |
| ARC-OQ-009 | What is the exact numeric baseline and target for missed-call/voicemail scheduling burden reduction? | Product Owner | Open | TBD |

## 18. Feature Mapping

| Feature | Relevant Components | Decision IDs | Interface IDs | Security IDs |
|---|---|---|---|---|
| FEAT-001 | COMP-001, COMP-002, COMP-003, COMP-006, COMP-007, COMP-008 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003 | ARD-API-001, ARD-API-002, ARD-API-003, ARD-API-006 | ARD-SEC-001, ARD-SEC-003 |
| FEAT-002 | COMP-001, COMP-002, COMP-004, COMP-008 | ARD-DEC-001, ARD-DEC-005 | ARD-API-001, ARD-API-004, ARD-API-006 | ARD-SEC-001, ARD-SEC-003 |
| FEAT-003 | COMP-002, COMP-003, COMP-006, COMP-007, COMP-008, COMP-009 | ARD-DEC-002, ARD-DEC-003, ARD-DEC-007 | ARD-API-002, ARD-API-003, ARD-API-006 | ARD-SEC-001, ARD-SEC-003, ARD-SEC-004 |
| FEAT-004 | COMP-001, COMP-002, COMP-004, COMP-008, COMP-009 | ARD-DEC-001, ARD-DEC-005, ARD-DEC-007 | ARD-API-004, ARD-API-006 | ARD-SEC-001, ARD-SEC-003 |
| FEAT-005 | COMP-001, COMP-005, COMP-006, COMP-008 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-004 | ARD-API-003, ARD-API-005, ARD-API-006 | ARD-SEC-001, ARD-SEC-002 |
| FEAT-006 | COMP-001, COMP-005, COMP-008, COMP-010 | ARD-DEC-001, ARD-DEC-004, ARD-DEC-005 | ARD-API-005, ARD-API-006 | ARD-SEC-001, ARD-SEC-002, ARD-SEC-004 |

## 19. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Solution Architect | TBD | Pending | TBD |
| Security Reviewer | TBD | Pending | TBD |
| Engineering Lead | TBD | Pending | TBD |