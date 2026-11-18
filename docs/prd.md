# Product Requirements Document (PRD)

> Requirement source: “Our clinic loses bookings because patients can only book by phone during office hours. We want patients to book, reschedule and get reminders online. Staff should see the day's schedule in one place.”

## Document Metadata

- **Product/Application:** Online Patient Scheduling and Reminders
- **Document ID:** PRD-OPSR
- **Version:** 0.1
- **Status:** Draft
- **Owner:** TBD
- **Contributors:** TBD
- **Last Updated:** 2026-11-18

---

## 1. Executive Summary

The clinic needs an online scheduling capability because patients currently can only book by phone during office hours, leading to lost bookings. Staff-reported missed calls and patient complaints indicate the problem exists, although quantified baseline data is not yet available.

This product will let existing and new patients book selected appointment types online, reschedule eligible appointments within clinic-configured rules, and receive SMS and email reminders. Front-desk staff will be able to view the day’s schedule in one place, including appointment status and reminder/contact status.

The first release will prioritize reducing lost bookings caused by phone-only scheduling while maintaining HIPAA-aligned controls appropriate for a small healthcare business.

## 2. Problem Statement

Patients who want to schedule outside office hours cannot complete bookings without calling the clinic during business hours. This creates missed booking opportunities for patients and front-desk staff, and the clinic has staff-reported missed calls and patient complaints with no quantified baseline yet. Staff also need a consolidated view of the day’s schedule to manage appointments and follow-up efficiently.

## 3. Product Vision

Create a HIPAA-aligned, mobile-first online scheduling experience where eligible patients can book and reschedule selected appointment types without calling the clinic, receive compliant reminders, and enable staff to manage the day’s schedule from one consolidated operational view.

## 4. Goals and Success Measures

### 4.1 Goals

- Reduce lost bookings caused by phone-only office-hour scheduling.
- Enable existing and new patients to book selected appointment types online with safeguards before confirmation.
- Allow patients to reschedule eligible appointments up to a clinic-configured cutoff time.
- Send SMS and email reminders with consent, opt-out, and staff-visible delivery/failure status.
- Give front-desk staff a consolidated view of the day’s schedule.
- Maintain HIPAA-aligned handling of protected health information.

### 4.2 Success Measures

| ID | Measure | Baseline | Target | Measurement Method |
|---|---|---:|---:|---|
| KPI-001 | Online booking adoption | Not quantified; see OQ-001 | Not set; see OQ-001 | Percentage of eligible appointments booked online during the measurement period |
| KPI-002 | Reduction in missed scheduling calls | Not quantified; see OQ-001 | Not set; see OQ-001 | Comparison of missed scheduling calls before and after launch |
| KPI-003 | Reminder delivery rate | Not quantified; see OQ-001 | Not set; see OQ-001 | Percentage of scheduled reminders successfully delivered by channel |
| KPI-004 | Staff schedule usage | Not quantified; see OQ-001 | Not set; see OQ-001 | Count or percentage of front-desk staff using the day schedule view during operating days |

## 5. Scope

### 5.1 In Scope

- Online booking for appointment types manually marked by staff as eligible for online booking.
- Online booking for both existing and new patients.
- Required new-patient intake details and insurance information before confirming an available slot.
- Patient rescheduling for eligible appointments up to a clinic-configured cutoff time.
- SMS and email appointment reminders.
- Reminder consent capture, opt-out support, and delivery/failure visibility for staff.
- Staff day schedule view with patient, provider, time, status, appointment type, and contact/reminder status.
- Clinic administrator controls for online-bookable appointment types and rescheduling cutoff rules.
- Integration with the existing scheduling/EHR system if available; import/export as an interim dependency if integration is not available.

### 5.2 Out of Scope

- Payments.
- Telehealth visits.
- Waitlist management.
- Room assignments.
- Clinical triage.
- Provider capacity planning or advanced scheduling optimization.
- Voice call reminders.

## 6. Personas and Users

### PERSONA-001 — Patient

- **Description:** Existing or new clinic patient who wants to book or reschedule eligible appointments online.
- **Primary Need:** Complete scheduling tasks without calling during office hours.
- **Key Pain Point:** Phone-only scheduling creates friction and prevents booking outside office hours.

### PERSONA-002 — Front-Desk Staff

- **Description:** Clinic staff responsible for managing daily appointments, scheduling follow-up, and handling patient communication.
- **Primary Need:** See the day’s schedule and appointment communication status in one place.
- **Key Pain Point:** Missed calls, manual scheduling work, and fragmented schedule visibility create operational burden.

### PERSONA-003 — Clinic Administrator

- **Description:** Clinic operator or manager responsible for scheduling policies, eligible appointment types, compliance, and rollout configuration.
- **Primary Need:** Control which appointment types are bookable online and maintain HIPAA-aligned scheduling operations.
- **Key Pain Point:** Needs patient convenience without exposing inappropriate appointment types or compliance risk.

## 7. Assumptions and Dependencies

### Assumptions

- ASM-001 — The clinic has appointment types that can be safely marked as eligible for online booking without clinical triage (assumption).
- ASM-002 — Patients can provide required intake details and insurance information online before confirmation (assumption).
- ASM-003 — SMS and email reminder vendors can support HIPAA-aligned operations with required business associate agreements where applicable (assumption).
- ASM-004 — The clinic can define a rescheduling cutoff time before launch (assumption).

### Dependencies

- DEP-001 — Existing scheduling/EHR system capabilities determine whether launch uses integration or interim import/export.
- DEP-002 — HIPAA-aligned vendors and business associate agreements are required for services that handle protected health information.
- DEP-003 — Staff must configure which appointment types are eligible for online booking.
- DEP-004 — Clinic policy must define the rescheduling cutoff time.

## 8. Functional Requirements

### PRD-FR-001 — Configure Online-Bookable Appointment Types

- **Priority:** Must
- **Feature:** FEAT-001
- **Persona:** PERSONA-003

**Requirement**

The system shall allow clinic administrators or authorized staff to mark appointment types as eligible or ineligible for online booking.

**Business Rationale**

The first release supports selected appointment types only, and staff need control to prevent inappropriate appointment types from being booked online.

**Acceptance Criteria**

1. An authorized user can mark an appointment type as online-bookable.
2. An authorized user can remove online-bookable status from an appointment type.
3. Patients only see appointment types marked as online-bookable in the patient booking flow.
4. Appointment types not marked as online-bookable cannot be selected by patients online.

---

### PRD-FR-002 — Book Eligible Appointments Online

- **Priority:** Must
- **Feature:** FEAT-002
- **Persona:** PERSONA-001

**Requirement**

The system shall allow existing and new patients to book available slots for appointment types marked as eligible for online booking.

**Business Rationale**

Patients need a way to complete bookings outside office hours, reducing lost bookings caused by phone-only scheduling.

**Acceptance Criteria**

1. A patient can select an online-bookable appointment type and view available appointment slots.
2. A patient can submit a booking for an available slot.
3. The system prevents booking for unavailable slots.
4. The system prevents patient booking for appointment types not marked as online-bookable.
5. A successfully booked appointment appears in the staff day schedule view.

---

### PRD-FR-003 — Collect New-Patient Intake and Insurance Information

- **Priority:** Must
- **Feature:** FEAT-002
- **Persona:** PERSONA-001

**Requirement**

The system shall collect required intake details and insurance information from new patients before confirming an available appointment slot.

**Business Rationale**

New-patient bookings require safeguards before confirmation so staff have necessary intake and insurance information.

**Acceptance Criteria**

1. A new patient cannot complete confirmation until required intake fields are submitted.
2. A new patient cannot complete confirmation until required insurance information is submitted.
3. A completed new-patient booking includes submitted intake and insurance information for staff review.
4. Existing patients are not forced through the new-patient intake flow unless identified as new patients.

---

### PRD-FR-004 — Reschedule Eligible Appointments Online

- **Priority:** Must
- **Feature:** FEAT-003
- **Persona:** PERSONA-001

**Requirement**

The system shall allow patients to reschedule eligible appointments up to a clinic-configured cutoff time.

**Business Rationale**

Patients need self-service flexibility while the clinic needs operational control over late schedule changes.

**Acceptance Criteria**

1. A patient can view rescheduling options for an appointment eligible for online rescheduling.
2. A patient can move an eligible appointment to an available slot before the configured cutoff time.
3. The system prevents patient self-service rescheduling after the configured cutoff time.
4. The rescheduled appointment is reflected in the staff day schedule view.
5. The system prevents rescheduling into unavailable slots.

---

### PRD-FR-005 — Configure Rescheduling Cutoff

- **Priority:** Must
- **Feature:** FEAT-003
- **Persona:** PERSONA-003

**Requirement**

The system shall allow authorized clinic administrators or staff to configure the cutoff time for patient self-service rescheduling.

**Business Rationale**

The clinic must control how close to an appointment patients can make schedule changes online.

**Acceptance Criteria**

1. An authorized user can set a rescheduling cutoff rule.
2. The patient rescheduling flow enforces the configured cutoff rule.
3. Unauthorized users cannot modify the rescheduling cutoff rule.

---

### PRD-FR-006 — Send SMS and Email Reminders

- **Priority:** Must
- **Feature:** FEAT-004
- **Persona:** PERSONA-001

**Requirement**

The system shall send appointment reminders by SMS and email to patients who have provided the required consent for each channel.

**Business Rationale**

Reminders support attendance and reduce manual reminder work while respecting consent requirements.

**Acceptance Criteria**

1. The system records whether a patient has consented to SMS reminders.
2. The system records whether a patient has consented to email reminders.
3. The system sends SMS reminders only when SMS consent is recorded.
4. The system sends email reminders only when email consent is recorded.
5. The system does not include protected health information in reminder content beyond the minimum necessary.

---

### PRD-FR-007 — Manage Reminder Opt-Outs and Delivery Status

- **Priority:** Must
- **Feature:** FEAT-004
- **Persona:** PERSONA-002

**Requirement**

The system shall allow reminder opt-out and show reminder delivery or failure status to staff.

**Business Rationale**

Staff need visibility into failed communications and patients must be able to opt out of reminder channels.

**Acceptance Criteria**

1. A patient can opt out of SMS reminders.
2. A patient can opt out of email reminders.
3. Staff can see whether each reminder was delivered, failed, or not sent due to missing consent or opt-out.
4. The system stops sending reminders through a channel after the patient opts out of that channel.

---

### PRD-FR-008 — View the Day’s Schedule

- **Priority:** Must
- **Feature:** FEAT-005
- **Persona:** PERSONA-002

**Requirement**

The system shall provide staff with a day schedule view containing all appointments for the day with patient, provider, time, status, appointment type, and contact/reminder status.

**Business Rationale**

Front-desk staff need one place to understand the day’s schedule and manage appointment communication.

**Acceptance Criteria**

1. Staff can view all appointments scheduled for a selected day.
2. Each appointment displays patient, provider, time, status, appointment type, and contact/reminder status.
3. Online-booked appointments appear in the day schedule view after confirmation.
4. Rescheduled appointments show the updated appointment time in the day schedule view.

---

### PRD-FR-009 — Connect to Existing Scheduling/EHR Workflow

- **Priority:** Should
- **Feature:** FEAT-006
- **Persona:** PERSONA-002

**Requirement**

The system shall integrate with the existing scheduling/EHR system if available, and shall support import/export as an interim workflow if integration is not available.

**Business Rationale**

The clinic’s existing scheduling/EHR workflow is a dependency, and launch should avoid duplicate schedule management where possible.

**Acceptance Criteria**

1. If scheduling/EHR integration is available, online bookings and schedule updates synchronize with the existing scheduling/EHR workflow.
2. If scheduling/EHR integration is not available, staff can use an import/export workflow to move schedule data between systems.
3. The implementation approach is documented before launch as either integrated or import/export.

## 9. Non-Functional Requirements

### PRD-NFR-001 — HIPAA-Aligned PHI Protection

- **Category:** Compliance
- **Priority:** Must

**Requirement**

The system shall use HIPAA-aligned controls for protected health information, including business associate agreements with PHI-handling vendors, encryption in transit and at rest, role-based access, audit logs, and minimum necessary PHI handling.

**Acceptance Criteria**

1. PHI-handling vendors have business associate agreements in place before production use.
2. PHI is encrypted in transit and at rest in production environments.
3. Staff access is restricted by role-based permissions.
4. Access to PHI and changes to scheduling records are captured in audit logs.
5. Reminder content is reviewed to verify it avoids PHI beyond the minimum necessary.

### PRD-NFR-002 — Patient Accessibility and Mobile Support

- **Category:** Accessibility
- **Priority:** Must

**Requirement**

The patient booking and rescheduling experience shall be mobile-first, responsive, and meet WCAG 2.1 AA.

**Acceptance Criteria**

1. Patient booking and rescheduling pages render correctly on mobile and desktop viewport sizes.
2. Accessibility testing verifies conformance with WCAG 2.1 AA for patient-facing scheduling flows.
3. Core booking and rescheduling tasks can be completed using keyboard navigation and assistive technology.

### PRD-NFR-003 — Privacy-Conscious Reminder Content

- **Category:** Privacy
- **Priority:** Must

**Requirement**

The system shall avoid including protected health information in SMS and email reminder content beyond the minimum necessary for appointment reminders.

**Acceptance Criteria**

1. SMS reminder templates do not include diagnosis, procedure details, or clinical notes.
2. Email reminder templates do not include diagnosis, procedure details, or clinical notes.
3. Reminder templates are reviewed before production use for minimum necessary PHI handling.

### PRD-NFR-004 — Auditability

- **Category:** Observability
- **Priority:** Must

**Requirement**

The system shall maintain audit records for scheduling actions, reminder consent changes, opt-outs, and staff access to appointment data.

**Acceptance Criteria**

1. Booking, rescheduling, and schedule-view access events are recorded with actor, timestamp, and action.
2. Reminder consent and opt-out changes are recorded with actor, timestamp, channel, and action.
3. Authorized staff can retrieve audit records when needed for compliance review.

## 10. User Journeys

### JOURNEY-001 — Patient Books an Eligible Appointment Online

1. Patient opens the online scheduling experience outside office hours.
2. Patient selects an appointment type marked as online-bookable.
3. Patient selects an available slot.
4. Existing patient confirms required booking information, or new patient submits required intake and insurance information.
5. Appointment is confirmed and appears in the staff day schedule view.

**Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-003, PRD-FR-008, PRD-NFR-001, PRD-NFR-002

### JOURNEY-002 — Patient Reschedules an Eligible Appointment

1. Patient accesses their eligible appointment.
2. System verifies the appointment is before the clinic-configured rescheduling cutoff.
3. Patient selects a new available slot.
4. System updates the appointment time.
5. Staff day schedule view reflects the rescheduled appointment.

**Related Requirements:** PRD-FR-004, PRD-FR-005, PRD-FR-008, PRD-NFR-001, PRD-NFR-002

### JOURNEY-003 — Patient Receives and Manages Reminders

1. Patient provides SMS and/or email reminder consent.
2. System sends reminders through consented channels.
3. Patient may opt out of a reminder channel.
4. Staff can see delivery, failure, opt-out, or not-sent status.

**Related Requirements:** PRD-FR-006, PRD-FR-007, PRD-NFR-001, PRD-NFR-003, PRD-NFR-004

### JOURNEY-004 — Staff Reviews the Day’s Schedule

1. Front-desk staff opens the day schedule view.
2. Staff reviews all appointments for the selected day.
3. Staff sees patient, provider, time, status, appointment type, and contact/reminder status.
4. Staff identifies any failed reminders or schedule changes requiring follow-up.

**Related Requirements:** PRD-FR-008, PRD-FR-009, PRD-NFR-001, PRD-NFR-004

## 11. Feature Breakdown

### FEAT-001 — Appointment Type Eligibility Configuration

- **Summary:** Enables authorized staff to choose which appointment types patients can book online.
- **Priority:** Must
- **Related Requirements:** PRD-FR-001
- **Expected Spec File:** `specs/FEAT-001-appointment-type-eligibility-configuration.spec.md`

### FEAT-002 — Online Patient Booking

- **Summary:** Enables existing and new patients to book available slots for online-bookable appointment types, including new-patient intake and insurance collection.
- **Priority:** Must
- **Related Requirements:** PRD-FR-002, PRD-FR-003, PRD-NFR-001, PRD-NFR-002
- **Expected Spec File:** `specs/FEAT-002-online-patient-booking.spec.md`

### FEAT-003 — Patient Rescheduling

- **Summary:** Enables patients to reschedule eligible appointments before a clinic-configured cutoff time.
- **Priority:** Must
- **Related Requirements:** PRD-FR-004, PRD-FR-005, PRD-NFR-001, PRD-NFR-002
- **Expected Spec File:** `specs/FEAT-003-patient-rescheduling.spec.md`

### FEAT-004 — Appointment Reminders

- **Summary:** Sends SMS and email reminders with consent, opt-out, delivery tracking, and minimum necessary PHI handling.
- **Priority:** Must
- **Related Requirements:** PRD-FR-006, PRD-FR-007, PRD-NFR-001, PRD-NFR-003, PRD-NFR-004
- **Expected Spec File:** `specs/FEAT-004-appointment-reminders.spec.md`

### FEAT-005 — Staff Day Schedule View

- **Summary:** Gives staff one place to view the day’s appointments and contact/reminder status.
- **Priority:** Must
- **Related Requirements:** PRD-FR-008, PRD-NFR-001, PRD-NFR-004
- **Expected Spec File:** `specs/FEAT-005-staff-day-schedule-view.spec.md`

### FEAT-006 — Scheduling/EHR Workflow Connection

- **Summary:** Connects online scheduling to the clinic’s existing scheduling/EHR workflow through integration when available or import/export as an interim workflow.
- **Priority:** Should
- **Related Requirements:** PRD-FR-009, PRD-NFR-001, PRD-NFR-004
- **Expected Spec File:** `specs/FEAT-006-scheduling-ehr-workflow-connection.spec.md`

## 12. Constraints

- CON-001 — The product must support HIPAA-aligned handling of protected health information.
- CON-002 — The clinic is a small business, so implementation should account for budget sensitivity.
- CON-003 — Existing scheduling/EHR capabilities may constrain integration scope and may require an interim import/export workflow.
- CON-004 — Target release date is not set; see OQ-002.
- CON-005 — First release is limited to selected appointment types marked as eligible for online booking.
- CON-006 — Reminder vendors that handle PHI must support required HIPAA-aligned agreements and controls.

## 13. Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| RISK-001 | Existing scheduling/EHR integration is unavailable or costly | High | Medium | Support import/export as an interim workflow and document the selected launch approach |
| RISK-002 | HIPAA-aligned vendor agreements delay reminder launch | High | Medium | Identify PHI-handling vendors early and require BAAs before production use |
| RISK-003 | Online-bookable appointment types are configured too broadly | High | Medium | Restrict launch to staff-marked eligible appointment types and exclude clinical triage |
| RISK-004 | Success cannot be evaluated because baselines are not quantified | Medium | High | Establish baseline measurement for KPIs before or immediately after pilot launch |
| RISK-005 | Reminder failures are missed by staff | Medium | Medium | Show delivery/failure status in the staff schedule view |

## 14. Open Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| OQ-001 | What are the quantified baselines and launch targets for online booking adoption, missed scheduling calls, reminder delivery rate, and staff schedule usage? | Product Owner | Open | Not set |
| OQ-002 | What is the target release date or pilot launch window? | Product Owner | Open | Not set |
| OQ-003 | What exact rescheduling cutoff time should the clinic configure for launch? | Clinic Administrator | Open | Not set |
| OQ-004 | Which existing scheduling/EHR system is used, and does it support the required integration approach? | Technical Owner | Open | Not set |
| OQ-005 | Which fields are required for new-patient intake and insurance collection? | Clinic Administrator | Open | Not set |

## 15. Release and Rollout Considerations

- **Target Release:** Not set; see OQ-002
- **Rollout Approach:** Pilot with selected appointment types before broader rollout.
- **Migration Considerations:** Existing scheduling/EHR integration should be used if available; otherwise launch requires an import/export workflow. Final approach depends on OQ-004.
- **Training/Support Considerations:** Staff training should cover appointment type eligibility configuration, the day schedule view, reminder delivery/failure status, opt-out handling, and any integration or import/export workflow.

## 16. Definition of Product Success

The product is successful when:

- Patients can book selected appointment types online without calling during office hours.
- Patients can reschedule eligible appointments before the clinic-configured cutoff time.
- SMS and email reminders are sent only with consent and show delivery/failure status to staff.
- Front-desk staff can use one day schedule view to see appointments and reminder/contact status.
- KPI targets in OQ-001 are defined and met or trending positively during rollout.
- All Must-priority requirements are accepted.
- Required quality, security, and compliance outcomes are evidenced.

## 17. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Product Owner | TBD | Pending | TBD |
| Business Sponsor | TBD | Pending | TBD |