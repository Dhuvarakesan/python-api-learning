# Architecture Requirements Document (ARD)

> Every material architecture decision or constraint has a stable ID that feature specifications can reference.

## Document Metadata

- **Product/Application:** Online Patient Scheduling and Reminders
- **Document ID:** ARD-OPSR
- **Version:** 0.1
- **Status:** Draft
- **Architecture Owner:** TBD; see ARC-OQ-001
- **Contributors:** TBD; see ARC-OQ-002
- **Last Updated:** 2026-11-19
- **Reference PRD:** `PRD.md`

---

## 1. Architecture Summary

Online Patient Scheduling and Reminders will use a hybrid architecture: a custom React patient/staff portal backed by a Node.js/TypeScript API, integrated with the existing practice management/EHR scheduling system, HIPAA-eligible SMS and email vendors, and managed cloud services covered by Business Associate Agreements where PHI is processed.

The EHR/practice management scheduling system remains the source of truth for availability, confirmed bookings, rescheduling, appointment status, and staff daily schedule data. The application stores only operational metadata needed to support the product: OTP verification records, audit logs, reminder state, reminder delivery/action status, pending new-patient requests, integration error records, configuration, and staff account/access data. Confirmed schedule data is not migrated into the application database.

Key architecture drivers are HIPAA-conscious PHI handling, source-of-truth integrity, safe handling of integration unavailability, small-business operational simplicity, patient accessibility, and measurable outside-hours booking capture.

## 2. Architecture Drivers

- PRD-FR-001 — Existing patients must verify identity with a one-time code before booking or rescheduling.
- PRD-FR-002 — Online booking must be limited to clinic-selected standard, low-risk appointment types.
- PRD-FR-003 — The system must handle no matching appointment slots without creating appointments.
- PRD-FR-004 — Existing-patient rescheduling must be blocked inside the clinic-defined cutoff window.
- PRD-FR-005 — New patients can create pending requests only and cannot directly book confirmed slots.
- PRD-FR-006 — The system must send SMS and email appointment reminders through suitable vendors.
- PRD-FR-007 — Reminder recipients must be able to confirm, cancel, or request reschedule.
- PRD-FR-008 — Staff need one daily view of appointments, appointment status, and reminder confirmations.
- PRD-FR-009 — The system must fail safely when the scheduling integration is unavailable.
- PRD-NFR-001 — PHI handling must support HIPAA compliance review and minimum-necessary reminder content.
- PRD-NFR-002 — Patient-facing flows must meet WCAG 2.1 AA.
- PRD-NFR-003 — The existing practice management/EHR system must remain the scheduling source of truth.
- CON-001 — The clinic is a small business, so the architecture must avoid unnecessary operational complexity.
- CON-002 — The product operates in healthcare and must support HIPAA-compliant handling of PHI.
- CON-003 — The existing practice management/EHR scheduling system is the source of truth.
- CON-004 — Direct online booking is limited to selected standard, low-risk appointment types.
- CON-005 — Existing-patient rescheduling is limited to outside a clinic-defined cutoff window.
- CON-006 — New patients can create pending requests only and cannot directly book confirmed slots.

## 3. Architecture Principles

1. Treat the EHR/practice management scheduling system as authoritative for confirmed schedule state.
2. Minimize PHI collection, retention, display, and transmission across all application flows.
3. Reuse established repository and platform patterns where appropriate.
4. Keep presentation, orchestration, domain, integration, and persistence concerns separated.
5. Prefer managed HIPAA-eligible services with BAAs to reduce operational burden for a small business.
6. Fail closed for booking/rescheduling when source-system confirmation is unavailable.
7. Make integration boundaries explicit through adapters so the EHR vendor can be confirmed later without redesigning the application core.

## 4. System Context

### 4.1 Actors

- Existing Patient — Uses OTP verification to book eligible appointments, reschedule eligible appointments, and act on reminders.
- New Patient — Submits a pending appointment request for staff review without booking a confirmed slot.
- Clinic Staff — Uses MFA-protected staff access to review the daily schedule, statuses, reminder confirmations, pending requests, and exceptions.
- Clinic Operations — Defines eligible appointment types, rescheduling cutoff rules, reminder timing, and operational exception handling.
- HIPAA/Security Reviewer — Reviews PHI handling, vendors, templates, access controls, audit logging, and production readiness.
- Practice Management/EHR Scheduling System — Owns availability, confirmed appointments, reschedules, appointment status, and daily schedule data.
- SMS Vendor — Sends OTPs or reminders by SMS and returns delivery/action status where supported.
- Email Vendor — Sends OTPs or reminders by email and returns delivery/action status where supported.
- Managed Cloud Platform — Hosts PHI-bearing application, database, secrets, logs, metrics, and deployment infrastructure under HIPAA-eligible services and BAAs.

### 4.2 External Systems

| ID | System | Purpose | Interface | Owner |
|---|---|---|---|---|
| EXT-001 | Existing practice management/EHR scheduling system | Source of truth for availability, confirmed bookings, reschedules, appointment status, and daily schedule data | Adapter; final API/file/interface TBD | EHR Vendor / Clinic |
| EXT-002 | SMS delivery vendor | SMS OTP and appointment reminder delivery with status callbacks | REST/Webhook | Vendor TBD; Clinic accountable |
| EXT-003 | Email delivery vendor | Email OTP and appointment reminder delivery with status callbacks | REST/Webhook | Vendor TBD; Clinic accountable |
| EXT-004 | Managed cloud platform | Hosting, database, secrets, observability, deployment, and encrypted storage | Managed services | Cloud Vendor TBD; Clinic accountable |
| EXT-005 | Staff MFA mechanism | MFA for application-managed staff accounts | Auth service/API | Vendor or platform TBD |

## 5. Technology Stack

| Layer | Technology | Version/Constraint | Rationale |
|---|---|---|---|
| Frontend | React | Exact version TBD; use supported production release | Supports accessible patient and staff web flows with componentized UI. |
| Backend | Node.js with TypeScript | Exact Node.js LTS version TBD | Matches selected stack and enables typed API, orchestration, and integration adapters. |
| Data | Encrypted managed relational database | HIPAA-eligible managed service with BAA; exact product TBD | Stores operational metadata, pending requests, audit logs, reminder state, and configuration without becoming schedule source of truth. |
| Hosting | HIPAA-eligible managed cloud services | Signed BAA required for PHI-bearing services | Reduces operational complexity while supporting HIPAA control needs. |
| Identity | Application-managed staff accounts with MFA and RBAC | MFA required for staff; patient OTP for patient flows | Meets staff access-control needs without assuming clinic SSO availability. |
| Messaging | HIPAA-eligible SMS and email vendors | BAAs, delivery webhooks, and minimum-necessary template controls required | Supports reminders and OTP delivery with compliance review. |
| Integration | EHR scheduling adapter layer | Final adapter implementation TBD after EHR capabilities are confirmed | Isolates vendor-specific scheduling integration from core workflows. |
| Infrastructure | Infrastructure as Code | Tool TBD | Enables repeatable environments, reviewable changes, and rollback-capable deployment. |
| Observability | Structured logs, metrics, traces, alerts | PHI redaction required | Supports HIPAA-conscious operations and integration troubleshooting. |

## 6. Logical Architecture

### 6.1 Components

#### COMP-001 — Patient Web Portal

- **Responsibility:** Presents patient-facing booking, rescheduling, reminder-action, OTP verification, no-slot, and new-patient request flows meeting WCAG 2.1 AA.
- **Inputs:** Patient-entered identity data, OTP code, appointment type criteria, slot selection, reschedule request, reminder action, new-patient request details.
- **Outputs:** API requests to booking, rescheduling, OTP, reminder-action, and request endpoints; accessible confirmations and safe error states.
- **Dependencies:** COMP-002, COMP-003, COMP-004, COMP-008.
- **Repository Area:** `frontend/` or TBD by Engineering Lead; see ARC-OQ-003.

#### COMP-002 — Staff Web Portal

- **Responsibility:** Provides staff daily schedule view, appointment statuses, reminder confirmations, pending request visibility, and integration exception visibility.
- **Inputs:** Staff credentials/MFA, schedule date filters, pending request review context.
- **Outputs:** Daily schedule display, reminder status display, pending request lists, exception indicators.
- **Dependencies:** COMP-003, COMP-005, COMP-006, COMP-008, COMP-009.
- **Repository Area:** `frontend/` or TBD by Engineering Lead; see ARC-OQ-003.

#### COMP-003 — Backend API and Workflow Orchestrator

- **Responsibility:** Owns server-side workflow orchestration for OTP verification, eligibility filtering, booking, rescheduling cutoff enforcement, reminder actions, pending requests, staff schedule aggregation, validation, and safe failure handling.
- **Inputs:** REST API calls from patient and staff portals, vendor callbacks, scheduled reminder triggers.
- **Outputs:** Responses to portals, EHR adapter calls, SMS/email vendor calls, database writes, audit events, observability telemetry.
- **Dependencies:** COMP-004, COMP-005, COMP-006, COMP-007, COMP-008, COMP-009.
- **Repository Area:** `backend/` or TBD by Engineering Lead; see ARC-OQ-003.

#### COMP-004 — Patient Verification Service

- **Responsibility:** Matches existing patients using approved matching rules, issues OTPs through phone or email, validates OTPs, enforces expiry/rate limits, and records audit events.
- **Inputs:** Patient identity match fields, selected delivery channel, OTP submissions.
- **Outputs:** OTP delivery requests, verification status, audit records, safe failure responses.
- **Dependencies:** COMP-003, COMP-006, COMP-007, EXT-002, EXT-003.
- **Repository Area:** `backend/auth/` or TBD by Engineering Lead; see ARC-OQ-003.

#### COMP-005 — Scheduling Integration Adapter

- **Responsibility:** Encapsulates all EHR/practice management scheduling operations for availability lookup, appointment creation, rescheduling, cancellation/status updates where supported, and daily schedule retrieval.
- **Inputs:** Availability criteria, appointment creation requests, reschedule requests, schedule date queries, reminder-action updates where supported.
- **Outputs:** Source-system confirmed booking/reschedule results, current schedule data, appointment status data, integration errors.
- **Dependencies:** EXT-001, COMP-003, COMP-009.
- **Repository Area:** `backend/integrations/ehr/` or TBD by Engineering Lead; see ARC-OQ-003.

#### COMP-006 — Reminder Service

- **Responsibility:** Schedules and sends SMS/email reminders, handles delivery callbacks, processes confirm/cancel/reschedule-request actions, and exposes reminder status to staff workflows.
- **Inputs:** Appointment data from EHR adapter, reminder timing configuration, patient contact data, vendor callbacks, reminder action submissions.
- **Outputs:** SMS/email sends, reminder state records, action records, audit events, staff-visible status.
- **Dependencies:** COMP-003, COMP-005, COMP-007, EXT-002, EXT-003.
- **Repository Area:** `backend/reminders/` or TBD by Engineering Lead; see ARC-OQ-003.

#### COMP-007 — Operational Data Store

- **Responsibility:** Persists operational metadata, OTP records, audit logs, reminder state, pending new-patient requests, integration errors, configuration, and staff account/access data.
- **Inputs:** Writes from backend workflows, callbacks, staff actions, system jobs.
- **Outputs:** Operational records for workflow continuity, staff display, audit review, metrics, and troubleshooting.
- **Dependencies:** Managed database in EXT-004.
- **Repository Area:** Database migrations in `backend/db/` or TBD by Engineering Lead; see ARC-OQ-003.

#### COMP-008 — Staff Identity and Access Control

- **Responsibility:** Manages staff accounts, MFA, roles, authorization checks, and session controls for staff-only functions.
- **Inputs:** Staff login credentials, MFA challenges, role assignments.
- **Outputs:** Staff sessions, authorization decisions, audit events.
- **Dependencies:** COMP-003, COMP-007, EXT-005.
- **Repository Area:** `backend/auth/` or TBD by Engineering Lead; see ARC-OQ-003.

#### COMP-009 — Observability and Audit Layer

- **Responsibility:** Produces structured logs, metrics, traces, alerts, security/audit events, and PHI-redacted operational telemetry across application and integration flows.
- **Inputs:** Events and telemetry from application components and integrations.
- **Outputs:** Dashboards, alerts, audit records, troubleshooting views, security-review evidence.
- **Dependencies:** EXT-004, COMP-003, COMP-007.
- **Repository Area:** `infra/observability/` or TBD by Engineering Lead; see ARC-OQ-003.

### 6.2 Data Flow

1. Existing Patient opens the Patient Web Portal and submits identity information for booking or rescheduling.
2. Patient Web Portal sends identity information to the Backend API and Workflow Orchestrator.
3. Patient Verification Service matches the patient against the approved source-system or adapter-supported lookup path, sends an OTP by SMS or email, and records an audit event.
4. Existing Patient submits the OTP; Patient Verification Service validates it and returns verified status to the workflow.
5. Backend API requests eligible appointment types and availability through the Scheduling Integration Adapter.
6. Scheduling Integration Adapter reads availability from the EHR/practice management scheduling system and returns only source-system-confirmed availability.
7. Existing Patient selects a slot; Backend API validates eligibility and, for rescheduling, checks the clinic-defined cutoff window.
8. Scheduling Integration Adapter creates or updates the appointment in the EHR; Backend API does not show completion unless the EHR confirms success.
9. Operational Data Store records audit events, verification metadata, reminder state, integration outcomes, and non-authoritative operational metadata.
10. Reminder Service sends HIPAA-reviewed SMS/email reminders and records delivery status and reminder actions.
11. Staff Web Portal requests the daily schedule; Backend API aggregates EHR schedule data with local reminder status, pending request state, and integration exception indicators.
12. If the EHR integration is unavailable, Backend API prevents confirmed appointment creation/modification, presents safe patient messaging, records the exception, and exposes it to staff.

## 7. Architecture Decisions

### ARD-DEC-001 — Hybrid Custom Portal Over Vendor Scheduling and Messaging

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-004, PRD-FR-006, PRD-FR-008, PRD-NFR-001, PRD-NFR-003

**Decision**

Use a custom React patient/staff portal and Node.js/TypeScript backend over vendor systems for EHR scheduling, SMS, email, managed cloud, and MFA.

**Rationale**

The PRD requires clinic-specific booking restrictions, OTP verification, reminder actions, daily schedule aggregation, and safe EHR source-of-truth behavior. A hybrid model allows workflow customization while avoiding the operational burden of replacing EHR scheduling or self-hosting commodity infrastructure.

**Consequences**

- Enables controlled patient and staff experience tailored to PRD flows.
- Keeps the EHR as the confirmed scheduling authority.
- Requires strong integration and vendor-management practices.
- Introduces dependency on third-party EHR, SMS, email, cloud, and MFA vendors.

**Alternatives Considered**

- Configure only a scheduling/reminder SaaS: not selected because staff daily view, OTP flow, EHR-source-of-truth behavior, and custom pending-request behavior may require clinic-specific orchestration.
- Fully custom scheduling system: not selected because it conflicts with the EHR source-of-truth constraint and increases operational complexity.
- Defer implementation until EHR capabilities are known: not selected as the architecture can proceed with an adapter boundary while vendor discovery remains open.

### ARD-DEC-002 — EHR Adapter Pattern for Scheduling Integration

- **Status:** Accepted
- **Related Requirements:** PRD-FR-002, PRD-FR-004, PRD-FR-008, PRD-FR-009, PRD-NFR-003

**Decision**

All scheduling operations must pass through a Scheduling Integration Adapter. The final implementation can use standards-based API, vendor REST API, file interface, or another vendor-supported method after EHR capabilities are confirmed.

**Rationale**

The approved PRD leaves the exact EHR system and capabilities open. An adapter boundary protects core workflows from vendor-specific implementation details and allows safe handling of integration unavailability.

**Consequences**

- Reduces rework when the EHR product is confirmed.
- Makes integration failure behavior explicit and testable.
- Requires contract tests or mocks for adapter behavior before the final vendor contract is available.
- Exact latency, consistency, and supported operations remain open until EHR discovery is complete.

**Alternatives Considered**

- Directly code against a presumed EHR API: not selected because the EHR product and capabilities are not settled.
- Flat-file-only integration: not selected because it may not support real-time booking and rescheduling needs.
- Manual staff reconciliation only: not selected because the PRD requires confirmed bookings and reschedules to reflect in the source scheduling system.

### ARD-DEC-003 — Minimal Local Persistence With EHR as Schedule Source of Truth

- **Status:** Accepted
- **Related Requirements:** PRD-FR-005, PRD-FR-006, PRD-FR-007, PRD-FR-008, PRD-NFR-001, PRD-NFR-003

**Decision**

The application will store only operational metadata, audit logs, OTP records, reminder state, pending new-patient requests, integration errors, configuration, and staff account/access records. Confirmed schedule and patient record data remain authoritative in the EHR/practice management system.

**Rationale**

This minimizes PHI footprint, supports HIPAA minimum-necessary principles, and avoids conflicting schedule state while still supporting reminders, staff visibility, and operational troubleshooting.

**Consequences**

- Reduces data migration and reconciliation risk.
- Limits duplicate PHI storage.
- Requires live or recently refreshed EHR access for confirmed schedule display.
- Staff view must identify or avoid stale schedule data if the EHR cannot be refreshed.

**Alternatives Considered**

- Replicate all EHR schedule and patient data locally: not selected because it increases PHI footprint and source-of-truth risk.
- Store no application data: not selected because reminders, pending requests, audit logs, OTPs, and exception handling require durable operational records.

### ARD-DEC-004 — HIPAA-Eligible Managed Cloud With BAAs

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001

**Decision**

Host PHI-bearing application components on HIPAA-eligible managed cloud services with signed BAAs, encrypted managed database/storage, managed secrets, audited key access, and environment isolation.

**Rationale**

The clinic is a small business and should avoid unnecessary operational complexity while still meeting healthcare compliance expectations.

**Consequences**

- Reduces infrastructure maintenance burden.
- Requires vendor due diligence and BAA completion before production.
- Requires careful service selection so only HIPAA-eligible services process PHI.

**Alternatives Considered**

- Clinic-owned on-premises hosting: not selected because it increases operational complexity.
- Existing EHR-hosted portal only: not selected because the selected implementation model requires a custom portal and workflow orchestration.

### ARD-DEC-005 — Application-Managed Staff Accounts With MFA and RBAC

- **Status:** Accepted
- **Related Requirements:** PRD-FR-008, PRD-NFR-001

**Decision**

Staff will use application-managed accounts with MFA and role-based access control for staff schedule and operational views.

**Rationale**

Clinic SSO and EHR delegated authorization are not confirmed. Application-managed accounts with MFA provide a clear launch path while enforcing staff-specific access controls.

**Consequences**

- Enables staff access before clinic identity integrations are confirmed.
- Requires account lifecycle administration and access review procedures.
- Can later be superseded by clinic SSO if available.

**Alternatives Considered**

- Clinic identity provider SSO: not selected for launch because availability is not established.
- EHR-launched staff access: not selected because EHR capability is unknown.

### ARD-DEC-006 — HIPAA-Eligible Messaging Vendors With Minimum-Necessary Templates

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-006, PRD-FR-007, PRD-NFR-001

**Decision**

SMS and email must use HIPAA-eligible vendors with BAAs, delivery webhooks, and minimum-necessary template controls approved before production.

**Rationale**

The PRD requires SMS and email reminders and OTP delivery while protecting PHI under HIPAA.

**Consequences**

- Supports delivery/status tracking and reminder actions.
- Requires compliance review of vendors and message templates.
- Reminder content must avoid unnecessary PHI.

**Alternatives Considered**

- Low-cost generic messaging vendors: not selected because HIPAA suitability and BAAs are required.
- EHR-native reminders only: not selected because EHR capability is unknown and the PRD requires custom reminder actions and staff visibility.

### ARD-DEC-007 — Automated CI/CD, IaC, Security Gates, and Rollback

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001, PRD-NFR-003

**Decision**

Use automated CI/CD with build, lint, tests, security scans, SBOM generation, infrastructure as code, and rollback-capable deployments with backward-compatible migrations.

**Rationale**

The system handles PHI and operational scheduling flows, so releases must be repeatable, reviewable, testable, and reversible.

**Consequences**

- Improves release reliability and auditability.
- Requires CI/CD and IaC setup before production.
- Database changes must be designed for safe rollback or forward-compatible deployment.

**Alternatives Considered**

- Manual deployments: not selected because they increase release and compliance evidence risk.
- Vendor-console-only deployments: not selected because they do not provide sufficient repeatability and traceability.

## 8. API and Integration Architecture

### ARD-API-001 — Patient Scheduling API

- **Type:** REST
- **Operation/Topic:** `POST /api/patient/otp/request`, `POST /api/patient/otp/verify`, `GET /api/patient/availability`, `POST /api/patient/bookings`, `POST /api/patient/reschedules`
- **Provider:** COMP-003
- **Consumers:** COMP-001
- **Authentication:** OTP-verified patient session for booking/rescheduling; unauthenticated initiation with rate limiting for OTP request.
- **Versioning:** URI versioning or compatible additive changes; exact convention TBD by Engineering Lead.

**Request/Input Contract**

Requests include minimum necessary identity-match fields, requested OTP channel, OTP code, eligible appointment criteria, selected slot identifier, appointment identifier for rescheduling, and action context. Exact schema is TBD after EHR integration discovery and clinic-approved matching rules.

**Response/Output Contract**

Responses include verification status, eligible appointment types, source-system-confirmed availability, booking/reschedule confirmation only after EHR acceptance, no-slot state, cutoff-blocked state, and integration-unavailable state.

**Error Handling**

Use safe patient-facing messages that do not expose unnecessary PHI. Booking/rescheduling errors must not create or display confirmed appointments without EHR confirmation. OTP failures must be rate-limited and audited. Integration timeouts/unavailability return a staff-follow-up or temporarily-unavailable path and record ARC-RISK-001 evidence.

### ARD-API-002 — New-Patient Request API

- **Type:** REST
- **Operation/Topic:** `POST /api/patient/appointment-requests`, `GET /api/staff/appointment-requests`
- **Provider:** COMP-003
- **Consumers:** COMP-001, COMP-002
- **Authentication:** Public submission with abuse controls for patients; staff MFA/RBAC for staff retrieval.
- **Versioning:** URI versioning or compatible additive changes; exact convention TBD by Engineering Lead.

**Request/Input Contract**

Patient submission includes minimum necessary request details required for staff review. Exact fields are TBD after clinic operations defines the intake request content.

**Response/Output Contract**

Patient response confirms that the request is pending staff review and not a confirmed appointment. Staff response lists pending requests with status and audit context.

**Error Handling**

Validation errors return accessible field-level messages. Submission errors must not create confirmed EHR appointments. PHI must be redacted from logs and telemetry.

### ARD-API-003 — Staff Schedule API

- **Type:** REST
- **Operation/Topic:** `GET /api/staff/schedule?date={date}`
- **Provider:** COMP-003
- **Consumers:** COMP-002
- **Authentication:** Staff MFA session with RBAC authorization.
- **Versioning:** URI versioning or compatible additive changes; exact convention TBD by Engineering Lead.

**Request/Input Contract**

Staff requests include date and optional view filters. The API uses staff authorization context to limit access.

**Response/Output Contract**

Response aggregates EHR daily schedule data with appointment status when available, reminder confirmation/delivery state, pending reschedule/cancellation context, pending new-patient request indicators, and stale/unavailable integration indicators.

**Error Handling**

If EHR data cannot be refreshed, the API must identify or avoid stale schedule data per PRD-NFR-003 and expose an operational exception state to staff.

### ARD-API-004 — EHR Scheduling Adapter Interface

- **Type:** REST | File | Adapter
- **Operation/Topic:** `lookupAvailability`, `createAppointment`, `rescheduleAppointment`, `getDailySchedule`, `getAppointmentStatus`, `updateActionStatus` where supported
- **Provider:** COMP-005
- **Consumers:** COMP-003, COMP-006
- **Authentication:** Vendor-supported secure integration authentication; exact method TBD after EHR discovery.
- **Versioning:** Adapter contract versioned internally; vendor API version pinned when known.

**Request/Input Contract**

Adapter calls include minimum necessary patient, appointment, appointment type, slot, and date criteria required by the EHR. Exact vendor schema is TBD after the EHR product and capabilities are confirmed.

**Response/Output Contract**

Adapter responses return source-system-confirmed availability, appointment creation/reschedule outcomes, appointment status, daily schedule data, and structured integration error categories.

**Error Handling**

All write operations must be idempotency-aware where vendor support permits. On timeout, unavailable, or ambiguous outcome, the application must not present a confirmed booking/reschedule unless the EHR confirms it. Errors must be logged with PHI redaction and exposed to staff as operational exceptions.

### ARD-API-005 — Messaging Vendor APIs and Webhooks

- **Type:** REST | Event
- **Operation/Topic:** `sendSms`, `sendEmail`, delivery-status webhook, action-link callback
- **Provider:** EXT-002, EXT-003, COMP-006
- **Consumers:** COMP-004, COMP-006
- **Authentication:** Vendor API credentials in managed secrets; signed webhooks where supported.
- **Versioning:** Vendor API version pinned when known; application callback contract versioned.

**Request/Input Contract**

Outbound messages use approved templates with minimum-necessary content, recipient contact details, template variables, delivery metadata, and callback identifiers.

**Response/Output Contract**

Vendor responses and callbacks include delivery state, failure reason where available, and callback identifiers mapped to local reminder or OTP records.

**Error Handling**

Retries may be used for transient vendor failures if they do not duplicate patient actions or spam recipients. Webhooks must validate authenticity where supported and be idempotent by callback identifier.

## 9. Data Architecture

### DATA-001 — Existing Patient Verification Record

- **Purpose:** Track OTP request, delivery channel, verification outcome, expiry, rate-limit context, and audit evidence.
- **System of Record:** Application operational data store; patient identity source remains EHR/practice management system where supported.
- **Classification:** Restricted
- **Retention:** Per clinic HIPAA record-retention policy; exact duration TBD in ARC-OQ-004.
- **Key Relationships:** Related to patient match context, OTP delivery event, booking/rescheduling session, and audit log.

### DATA-002 — Eligible Appointment Type Configuration

- **Purpose:** Define clinic-selected standard, low-risk appointment types available for online booking.
- **System of Record:** Application configuration, aligned with clinic operations and EHR appointment type identifiers.
- **Classification:** Internal
- **Retention:** Retain current and audit history per clinic configuration-retention policy; exact duration TBD in ARC-OQ-004.
- **Key Relationships:** Maps to EHR appointment type identifiers and booking/rescheduling rules.

### DATA-003 — Appointment Availability and Confirmed Schedule Data

- **Purpose:** Support availability display, booking, rescheduling, and staff daily schedule view.
- **System of Record:** Existing practice management/EHR scheduling system.
- **Classification:** Restricted
- **Retention:** Governed by EHR/practice management system policy; application should not persist confirmed schedule data beyond operational metadata/cache rules approved by compliance.
- **Key Relationships:** Related to reminder state, staff daily view, booking/rescheduling workflows, and audit events.

### DATA-004 — Reminder State and Delivery Status

- **Purpose:** Track SMS/email reminder scheduling, delivery status, confirmations, cancellations, and reschedule requests.
- **System of Record:** Application operational data store for reminder state; appointment truth remains EHR.
- **Classification:** Restricted
- **Retention:** Per clinic HIPAA record-retention policy; exact duration TBD in ARC-OQ-004.
- **Key Relationships:** Related to EHR appointment identifier, messaging vendor callback identifier, patient action, and staff schedule display.

### DATA-005 — Pending New-Patient Request

- **Purpose:** Capture new-patient appointment requests for staff review without booking a confirmed slot.
- **System of Record:** Application operational data store until staff processes the request; subsequent confirmed appointment remains EHR source of truth.
- **Classification:** Restricted
- **Retention:** Per clinic HIPAA record-retention policy; exact duration TBD in ARC-OQ-004.
- **Key Relationships:** Related to staff review status, audit log, and possible later EHR patient/appointment record.

### DATA-006 — Staff Account and Access Record

- **Purpose:** Manage staff authentication, MFA status, roles, authorization, and access review evidence.
- **System of Record:** Application identity/access store for launch.
- **Classification:** Confidential
- **Retention:** Per clinic access-management and audit policy; exact duration TBD in ARC-OQ-004.
- **Key Relationships:** Related to staff sessions, role assignments, audit events, and administrative actions.

### DATA-007 — Audit Log and Integration Error Record

- **Purpose:** Record security-relevant events, PHI-sensitive operational actions, OTP verification events, booking/rescheduling attempts, reminder actions, and integration failures.
- **System of Record:** Application operational data store and observability/audit platform.
- **Classification:** Restricted
- **Retention:** Per clinic HIPAA/security audit-retention policy; exact duration TBD in ARC-OQ-004.
- **Key Relationships:** Related to patient sessions, staff accounts, EHR adapter calls, messaging callbacks, and incident investigations.

### Data Migration

- **Required:** No
- **Approach:** No migration of confirmed schedule data. Configure eligible appointment types, cutoff rules, vendor credentials, staff accounts, environments, and operational settings before launch. Seed only required operational settings; confirmed appointment and patient schedule state remains in the EHR/practice management system.

## 10. Security Architecture

### ARD-SEC-001 — HIPAA-Conscious PHI Handling

- **Related Requirements:** PRD-NFR-001, PRD-FR-001, PRD-FR-006, PRD-FR-008

The implementation must:

- Use HIPAA-eligible vendors and signed BAAs for PHI-bearing cloud, SMS, email, and other processing services.
- Apply minimum-necessary PHI principles to booking, rescheduling, reminders, OTPs, staff views, logs, and alerts.
- Obtain clinic HIPAA compliance review approval before production launch.
- Require reminder templates to be approved for minimum-necessary PHI disclosure before production use.
- Redact PHI from logs, traces, metrics, alerts, and error messages unless explicitly approved for audit storage.
- Record verification, booking/rescheduling, reminder action, staff access, and integration exception events for audit or operational review.

The implementation must not:

- Send reminder or OTP content that exposes unnecessary PHI.
- Store confirmed schedule data as authoritative outside the EHR/practice management system.
- Display confirmed booking or rescheduling success unless the EHR/source scheduling system has accepted the write.
- Use vendors that process PHI before HIPAA suitability and BAA status are reviewed.

### ARD-SEC-002 — Patient OTP Verification

- **Related Requirements:** PRD-FR-001, PRD-NFR-001

The implementation must:

- Verify existing patients with one-time codes sent to phone or email before booking or rescheduling.
- Enforce OTP expiry, retry limits, request rate limits, and safe failure handling.
- Record verification events without exposing unnecessary PHI.
- Prevent booking or rescheduling if identity cannot be matched.

The implementation must not:

- Allow unauthenticated existing-patient booking or rescheduling.
- Reveal whether sensitive patient attributes exist beyond what is needed for safe user guidance.

### ARD-SEC-003 — Staff MFA and RBAC

- **Related Requirements:** PRD-FR-008, PRD-NFR-001

The implementation must:

- Require MFA for staff access to daily schedule, pending requests, reminder status, and exception views.
- Enforce role-based authorization on staff APIs and UI functions.
- Log staff authentication, authorization failures, and PHI-access-relevant events.
- Support account lifecycle administration and access review.

The implementation must not:

- Permit shared staff accounts for production use.
- Allow staff-only functions from patient sessions.

### ARD-SEC-004 — Secure SDLC and Supply Chain Controls

- **Related Requirements:** PRD-NFR-001

The implementation must:

- Run dependency, container, and code security scans in CI/CD.
- Block release on critical and high dependency/container/code-scan findings unless formally risk-accepted by the security owner.
- Generate an SBOM for every release.
- Protect production secrets through managed secrets storage.

The implementation must not:

- Store secrets in source control, application logs, or client-side code.
- Deploy PHI-bearing changes without required build, test, lint, and security gates.

### Security Controls

- **Authentication:** Patient OTP verification for booking/rescheduling; application-managed staff accounts with MFA for staff functions.
- **Authorization:** RBAC for staff APIs and UI; verified patient sessions limited to the specific booking/rescheduling/reminder-action context.
- **Secrets Management:** Managed secrets vault; no secrets in source control, logs, or client bundles.
- **Encryption:** TLS for data in transit; encrypted managed database/storage at rest; key access audited.
- **Audit Logging:** Audit OTP events, staff access, booking/reschedule attempts, reminder actions, configuration changes, vendor callbacks, and integration exceptions with PHI redaction controls.
- **Dependency/CVE Scanning:** Block critical and high dependency, container, and code-scan findings unless formally risk-accepted.
- **SBOM:** Required for every release.

## 11. Reliability, Performance, and Scalability

### ARD-OPS-001 — Application Availability

- **Target:** 99.5% monthly uptime for patient-facing booking and reminder-action application components, excluding third-party EHR/SMS/email outages.
- **Validation Method:** Production uptime monitoring, synthetic checks for patient entry points, incident review, and monthly availability reporting.

### ARD-OPS-002 — Source-of-Truth Integrity

- **Target:** 100% of displayed booking and rescheduling completions must have source-system confirmation from the EHR/practice management system.
- **Validation Method:** Integration contract tests, transactional workflow tests, audit sampling, and reconciliation between application events and EHR outcomes.

### ARD-OPS-003 — Safe Integration Failure Behavior

- **Target:** 100% of EHR integration unavailable, timeout, or ambiguous write outcomes must avoid creating or displaying unconfirmed completed appointments.
- **Validation Method:** Failure-injection tests, adapter timeout tests, end-to-end negative tests, and staff exception-view acceptance testing.

### ARD-OPS-004 — Patient and Staff Interaction Performance

- **Target:** 95% of non-EHR-dependent page/API interactions complete within 2 seconds under expected launch load.
- **Validation Method:** Load tests for application-owned flows, frontend performance tests, API latency dashboards, and release performance reports.

### ARD-OPS-005 — Accessibility Quality

- **Target:** Patient-facing booking, rescheduling, reminder-action, and new-patient request flows pass WCAG 2.1 AA review before launch.
- **Validation Method:** Automated accessibility checks, keyboard-only testing, screen-reader review, and acceptance testing against PRD-NFR-002.

### ARD-OPS-006 — Reminder Operational Visibility

- **Target:** 100% of reminder send attempts produce a recorded delivery status, failure status, or vendor-unavailable exception visible for operational review.
- **Validation Method:** Messaging vendor callback tests, reminder job integration tests, and staff/operations dashboard acceptance testing.

## 12. Observability

- **Logging:** Use structured application, integration, security, and audit logs with PHI redaction and correlation IDs across patient, staff, EHR, SMS, and email flows.
- **Metrics:** Track application uptime, API latency, OTP success/failure, booking attempts, source-system confirmations, reschedule attempts, reminder sends, delivery status, reminder actions, integration errors, and outside-hours online bookings.
- **Tracing:** Trace backend orchestration across portal APIs, EHR adapter calls, messaging vendor calls, database operations, and callbacks, with PHI-safe attributes only.
- **Alerting:** Alert on application downtime, elevated API errors, EHR adapter failures, SMS/email delivery failures, security scan failures, suspicious OTP rates, and reminder job failures.
- **Sensitive Data Restrictions:** Do not log full patient identifiers, OTP codes, detailed appointment reasons, message bodies, or unnecessary PHI in logs, traces, metrics, alerts, or client-side analytics.

## 13. Deployment Architecture

- **Environments:** Development, test, staging, and production with isolated data, configuration, credentials, and access controls.
- **Deployment Model:** Managed cloud deployment for React frontend, Node.js/TypeScript backend, managed database, managed secrets, observability, and integration configuration.
- **CI/CD:** Automated CI/CD with build, lint, unit tests, integration tests, contract tests, accessibility checks, security scans, SBOM generation, IaC validation, and deployment promotion.
- **Configuration Management:** Environment-specific configuration through managed configuration/secrets services; clinic rules for appointment eligibility, cutoff windows, and reminder timing must be configurable.
- **Rollback:** Automated rollback to the prior application version; database migrations must be backward-compatible or otherwise designed for safe rollback/roll-forward.
- **Infrastructure as Code:** Required; exact tool TBD by Engineering Lead in ARC-OQ-005.

## 14. Testing Architecture

Required test levels:

- Unit tests
- Integration tests
- Contract tests
- End-to-end tests
- Security tests
- Performance tests, where applicable

Minimum quality expectations:

- **Coverage Threshold:** Minimum 80% coverage for backend domain/integration orchestration and critical frontend flows.
- **Blocking Security Severity:** Critical and high dependency, container, and code-scan findings block release unless formally risk-accepted by the security owner.
- **Required Build/Lint Checks:** TypeScript type checks, frontend build, backend build, lint, unit tests, integration tests, EHR adapter contract tests or mocks, end-to-end tests for critical booking/rescheduling/reminder/staff flows, accessibility checks for patient-facing flows, security scans, SBOM generation, and IaC validation.

## 15. Architecture Constraints

- ARC-CON-001 — The clinic is a small business, so architecture must avoid unnecessary operational complexity and prefer managed services where appropriate.
- ARC-CON-002 — The product operates in healthcare and must support HIPAA-compliant PHI handling.
- ARC-CON-003 — PHI-bearing vendors and managed services require HIPAA suitability review and BAAs before production use.
- ARC-CON-004 — The existing practice management/EHR scheduling system is the source of truth for availability, confirmed bookings, rescheduling, appointment status, and daily schedule display.
- ARC-CON-005 — Direct online booking is limited to clinic-selected standard, low-risk appointment types.
- ARC-CON-006 — Existing-patient rescheduling is limited to outside a clinic-defined cutoff window.
- ARC-CON-007 — New patients can create pending requests only and cannot directly book confirmed slots.
- ARC-CON-008 — Patient-facing booking, rescheduling, reminder-action, and new-patient request flows must meet WCAG 2.1 AA.
- ARC-CON-009 — The application must not display a completed booking or reschedule without EHR/source-system confirmation.
- ARC-CON-010 — Confirmed schedule data must not be migrated into the application database for launch.
- ARC-CON-011 — Production deployments require automated CI/CD, security gates, SBOM generation, IaC, and rollback support.
- ARC-CON-012 — Application-held operational data retention must follow clinic HIPAA record-retention policy configured per data type before launch.

## 16. Risks and Trade-offs

| ID | Risk/Trade-off | Impact | Mitigation |
|---|---|---|---|
| ARC-RISK-001 | EHR integration capabilities may not support required booking, rescheduling, status, or daily schedule operations | High | Use adapter pattern, validate capabilities early, keep unsupported operations under Open Architecture Questions, and fail safely without unconfirmed appointments. |
| ARC-RISK-002 | Reminder or OTP content may expose more PHI than clinic policy permits | High | Require minimum-necessary template design, HIPAA review, BAA review, and PHI redaction in logs/telemetry. |
| ARC-RISK-003 | Patient identity matching may be unreliable due to incomplete phone/email records | High | Provide safe failure handling, block booking/rescheduling on failed match, and route to staff-contact or request-review path. |
| ARC-RISK-004 | Application-managed staff accounts add account lifecycle burden | Medium | Require MFA, RBAC, access review, audit logging, and consider future SSO if clinic identity capability becomes available. |
| ARC-RISK-005 | Minimal local persistence reduces PHI footprint but increases dependency on EHR availability | Medium | Use safe degraded states, staff exception visibility, monitoring, and no confirmed writes without EHR confirmation. |
| ARC-RISK-006 | Unknown retention policy could delay production approval | Medium | Configure retention per data type before launch and track under ARC-OQ-004. |
| ARC-RISK-007 | Small-business support capacity may be strained by after-hours exceptions | Medium | Define production support owner, escalation paths, and operational alerts before launch. |
| ARC-RISK-008 | Performance target excludes EHR-dependent latency, so end-to-end booking may vary by EHR responsiveness | Medium | Measure EHR adapter latency separately, show safe loading/error states, and add targets after integration benchmarks. |

## 17. Open Architecture Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| ARC-OQ-001 | Who is the named Architecture Owner for ARD-OPSR? | Product Owner | Open | TBD |
| ARC-OQ-002 | Who are the named ARD contributors, security reviewer, and engineering lead? | Product Owner | Open | TBD |
| ARC-OQ-003 | What repository structure and exact repository paths will be used for frontend, backend, database, integration, and infrastructure code? | Engineering Lead | Open | TBD |
| ARC-OQ-004 | What exact retention periods apply per data type for OTP records, reminder state, pending requests, staff access records, audit logs, and integration errors under clinic HIPAA policy? | Compliance Owner | Open | TBD |
| ARC-OQ-005 | Which IaC tool, managed cloud provider, managed database product, secrets service, and observability platform will be selected? | Engineering Lead | Open | TBD |
| ARC-OQ-006 | Which practice management/EHR system is in use, and which integration operations, authentication methods, latency, rate limits, and webhook/file/API capabilities are available? | Technical Owner | Open | TBD |
| ARC-OQ-007 | Which exact appointment types are approved as standard, low-risk, and eligible for online booking? | Clinic Operations | Open | TBD |
| ARC-OQ-008 | What is the clinic-defined cutoff window for online rescheduling? | Clinic Operations | Open | TBD |
| ARC-OQ-009 | What clinic-approved next step should patients see when no slots match, identity cannot be matched, or the scheduling integration is unavailable? | Product Owner | Open | TBD |
| ARC-OQ-010 | What reminder timing rules apply for SMS and email reminders? | Clinic Operations | Open | TBD |
| ARC-OQ-011 | What are the baseline counts and post-baseline numeric targets for outside-hours bookings, total booking volume, reminder confirmations, cancellations, and reschedule requests? | Product Owner | Open | TBD |
| ARC-OQ-012 | What production support owner and escalation process will handle failed bookings, reminder failures, integration outages, and pending requests? | Clinic Operations | Open | TBD |
| ARC-OQ-013 | What exact patient identity matching fields are approved for OTP initiation, and which source system supplies them? | Compliance Owner / Technical Owner | Open | TBD |
| ARC-OQ-014 | What exact API versioning convention will be used for application REST APIs? | Engineering Lead | Open | TBD |
| ARC-OQ-015 | What target release date and production readiness review date will be used? | Product Owner | Open | TBD |

## 18. Feature Mapping

| Feature | Relevant Components | Decision IDs | Interface IDs | Security IDs |
|---|---|---|---|---|
| FEAT-001 | COMP-001, COMP-003, COMP-004, COMP-005, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003, ARD-DEC-004 | ARD-API-001, ARD-API-004 | ARD-SEC-001, ARD-SEC-002, ARD-SEC-004 |
| FEAT-002 | COMP-001, COMP-003, COMP-004, COMP-005, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003, ARD-DEC-004 | ARD-API-001, ARD-API-004 | ARD-SEC-001, ARD-SEC-002, ARD-SEC-004 |
| FEAT-003 | COMP-001, COMP-002, COMP-003, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-003, ARD-DEC-004 | ARD-API-002 | ARD-SEC-001, ARD-SEC-004 |
| FEAT-004 | COMP-001, COMP-002, COMP-003, COMP-006, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-003, ARD-DEC-006 | ARD-API-005 | ARD-SEC-001, ARD-SEC-004 |
| FEAT-005 | COMP-002, COMP-003, COMP-005, COMP-006, COMP-007, COMP-008, COMP-009 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003, ARD-DEC-005 | ARD-API-003, ARD-API-004 | ARD-SEC-001, ARD-SEC-003, ARD-SEC-004 |
| FEAT-006 | COMP-001, COMP-002, COMP-003, COMP-005, COMP-007, COMP-009 | ARD-DEC-002, ARD-DEC-003, ARD-DEC-007 | ARD-API-001, ARD-API-003, ARD-API-004 | ARD-SEC-001, ARD-SEC-004 |

## 19. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Solution Architect | TBD; see ARC-OQ-001 | Pending | TBD |
| Security Reviewer | TBD; see ARC-OQ-002 | Pending | TBD |
| Engineering Lead | TBD; see ARC-OQ-002 | Pending | TBD |