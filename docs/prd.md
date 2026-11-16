# Product Requirements Document (PRD)

> Replace all `<PLACEHOLDER>` values. Keep requirement IDs stable after approval so downstream ARD and feature specifications retain traceability.

## Document Metadata

- **Product/Application:** Online Patient Scheduling and Reminders
- **Document ID:** PRD-OPSR
- **Version:** 0.1
- **Status:** Draft
- **Owner:** TBD
- **Contributors:** TBD
- **Last Updated:** 2026-11-16

---

## 1. Executive Summary

The Online Patient Scheduling and Reminders product enables new and existing patients to book eligible appointment types online, reschedule eligible appointments, and receive appointment reminders through their chosen channel of SMS, email, or both. The product addresses the clinic’s current dependency on phone-only booking during office hours, which causes missed booking opportunities when patients try to schedule outside staff availability.

Staff will be able to manage which appointment types are eligible for online booking, decide whether each appointment type is instantly confirmed or staff-approved, and view the day’s appointments and pending booking or reschedule requests in one place. The solution must operate in a healthcare setting for a small business and comply with HIPAA requirements.

## 2. Problem Statement

The clinic loses bookings because patients can currently book only by phone during office hours. This affects new and existing patients who want to schedule or change appointments outside office hours, and it creates staff workload by concentrating booking activity into phone-based workflows. The baseline for lost bookings and outside-hours demand is currently unknown and must be measured during rollout.

## 3. Product Vision

Patients can self-serve routine scheduling needs online at any time for clinic-approved appointment types, while staff retain control over appointment eligibility, confirmation rules, and same-day schedule operations. The target end state is a HIPAA-conscious scheduling experience that captures outside-hours demand, reduces reliance on phone-only booking, and gives staff a consolidated daily operational view.

## 4. Goals and Success Measures

### 4.1 Goals

- Enable new and existing patients to book eligible appointments online outside office hours.
- Enable patients to reschedule eligible appointments online within staff-defined rules.
- Allow patients to receive appointment reminders by SMS, email, or both.
- Give staff one daily schedule view that includes all appointments and pending booking or reschedule requests.
- Preserve staff control over eligible appointment types and confirmation workflows.
- Maintain HIPAA-conscious handling of appointment and reminder information.

### 4.2 Success Measures

| ID | Measure | Baseline | Target | Measurement Method |
|---|---|---:|---:|---|
| KPI-001 | Completed online bookings outside office hours | TBD | TBD | Compare timestamped online bookings against defined office hours during rollout |
| KPI-002 | Staff phone booking workload | TBD | TBD | Track booking-related phone volume or staff-reported booking workload before and after launch |
| KPI-003 | Reminder adoption | TBD | TBD | Measure percentage of booked appointments with SMS, email, or both reminders enabled |
| KPI-004 | Staff schedule visibility usage | TBD | TBD | Track daily staff use of the schedule view and pending request queue |

## 5. Scope

### 5.1 In Scope

- Online booking for a limited staff-managed list of eligible appointment types.
- Support for both new and existing patients.
- Staff-configurable confirmation mode per appointment type: instant confirmation or manual approval.
- Online rescheduling for eligible appointments until a staff-defined cutoff.
- Secure verification before rescheduling.
- Patient reminder preference selection for SMS, email, or both.
- Reminder content that avoids diagnosis, treatment details, or sensitive PHI beyond appointment logistics.
- Staff daily schedule view showing all appointments plus pending booking and reschedule requests.
- Staff-reviewed booking request flow when no eligible appointment slots are available.
- Role-based staff ability for clinic administrators and scheduling staff to configure online booking rules.
- Patient-facing flows conforming to WCAG 2.1 AA.
- Pilot rollout with one provider or appointment type before broader rollout.

### 5.2 Out of Scope

- Payments.
- Insurance verification.
- Clinical intake forms.
- Emergency or urgent care scheduling.
- Appointments requiring clinical triage before scheduling.
- Detailed diagnosis, treatment, or visit-reason content in reminders.

## 6. Personas and Users

### PERSONA-001 — Patient

- **Description:** A new or existing patient who needs to book or reschedule an eligible clinic appointment online.
- **Primary Need:** Schedule or change an appointment without calling during office hours.
- **Key Pain Point:** Phone-only booking makes it difficult or impossible to book when the clinic office is closed.

### PERSONA-002 — Scheduling Staff

- **Description:** Clinic staff responsible for appointment scheduling, reviewing requests, and managing daily schedule operations.
- **Primary Need:** See the day’s schedule and pending scheduling work in one place.
- **Key Pain Point:** Phone-based booking and scattered scheduling information create workload and operational friction.

### PERSONA-003 — Clinic Administrator

- **Description:** Authorized clinic staff responsible for configuring scheduling rules and appointment type eligibility.
- **Primary Need:** Control which appointment types can be booked online and whether they require staff approval.
- **Key Pain Point:** Uncontrolled online booking could create operational, clinical, or compliance risk.

## 7. Assumptions and Dependencies

### Assumptions

- ASM-001 — The clinic will define the limited list of appointment types eligible for online booking before launch (assumption).
- ASM-002 — The clinic will define office hours so outside-hours booking capture can be measured (assumption).
- ASM-003 — Clinic administrators and scheduling staff are the staff roles authorized to configure booking rules.
- ASM-004 — The first rollout will be a pilot with one provider or appointment type before full clinic rollout.
- ASM-005 — Emergency, urgent care, and clinically triaged appointments require workflows outside this first release.

### Dependencies

- DEP-001 — Scheduling data source is TBD because the current scheduling system is not specified.
- DEP-002 — Staff must provide appointment type eligibility, confirmation mode, and rescheduling cutoff rules.
- DEP-003 — SMS and email reminder delivery depend on approved communication services that can support HIPAA requirements.
- DEP-004 — Secure verification for rescheduling depends on an approved identity or verification approach.
- DEP-005 — HIPAA/privacy review is required before launch.

## 8. Functional Requirements

### PRD-FR-001 — Eligible Appointment Type Configuration

- **Priority:** Must
- **Feature:** FEAT-001
- **Persona:** PERSONA-003

**Requirement**

The system shall allow clinic administrators and scheduling staff to configure a limited list of appointment types that are eligible for online booking.

**Business Rationale**

The clinic needs staff control over what patients can book online to avoid exposing appointment types that require triage, special handling, or phone-based coordination.

**Acceptance Criteria**

1. Authorized staff can mark an appointment type as eligible or not eligible for online booking.
2. Appointment types not marked eligible are not shown as bookable in the patient booking flow.
3. Unauthorized staff cannot modify online booking eligibility.

---

### PRD-FR-002 — Appointment Confirmation Mode Configuration

- **Priority:** Must
- **Feature:** FEAT-001
- **Persona:** PERSONA-003

**Requirement**

The system shall allow authorized staff to configure each eligible appointment type as either instantly confirmed or requiring manual staff approval.

**Business Rationale**

Different appointment types may carry different scheduling risk, so staff need control over whether an online action creates a confirmed appointment or a pending request.

**Acceptance Criteria**

1. Authorized staff can select instant confirmation or manual approval for each eligible appointment type.
2. When an appointment type is configured for instant confirmation, a patient receives confirmation after selecting an available slot and completing required booking details.
3. When an appointment type is configured for manual approval, the patient submission creates a pending request visible to staff.

---

### PRD-FR-003 — Online Appointment Booking

- **Priority:** Must
- **Feature:** FEAT-002
- **Persona:** PERSONA-001

**Requirement**

The system shall allow new and existing patients to book eligible appointment types online using required contact and demographic details.

**Business Rationale**

Patients need an alternative to office-hours phone booking so the clinic can capture scheduling demand at any time.

**Acceptance Criteria**

1. A patient can view eligible appointment types in the online booking flow.
2. A patient can submit required contact and demographic details to book an eligible appointment.
3. The system records whether the appointment is confirmed instantly or pending staff approval based on the appointment type configuration.
4. The system captures booking submission time so outside-hours booking activity can be measured.

---

### PRD-FR-004 — No Availability Booking Request

- **Priority:** Must
- **Feature:** FEAT-002
- **Persona:** PERSONA-001

**Requirement**

The system shall show when no eligible appointment slots are available and offer the patient a staff-reviewed booking request.

**Business Rationale**

Patients should not abandon scheduling entirely when an eligible slot is unavailable online; staff should still be able to capture and respond to demand.

**Acceptance Criteria**

1. When no eligible slots are available for the selected appointment type, the system displays that no online availability is available.
2. The patient can submit a booking request for staff review from the no-availability state.
3. Staff can see the submitted booking request in the pending request queue.

---

### PRD-FR-005 — Online Rescheduling

- **Priority:** Must
- **Feature:** FEAT-003
- **Persona:** PERSONA-001

**Requirement**

The system shall allow patients to reschedule eligible appointments online until a staff-defined cutoff after completing secure verification.

**Business Rationale**

Patients need a self-service way to change eligible appointments while the clinic maintains control over timing and patient identity safeguards.

**Acceptance Criteria**

1. A patient must complete secure verification before accessing online rescheduling for an appointment.
2. The system allows rescheduling only for appointments and appointment types marked eligible for online rescheduling.
3. The system blocks rescheduling after the staff-defined cutoff and communicates that online rescheduling is no longer available.
4. A successful reschedule updates or creates the appropriate schedule item or pending request according to staff configuration.

---

### PRD-FR-006 — Reminder Preference Selection

- **Priority:** Must
- **Feature:** FEAT-004
- **Persona:** PERSONA-001

**Requirement**

The system shall allow patients to choose appointment reminders by SMS, email, or both.

**Business Rationale**

Reminder preferences help patients receive appointment information through the channels they are most likely to use.

**Acceptance Criteria**

1. A patient can select SMS, email, or both as reminder channels during booking or appointment management.
2. The system stores the selected reminder channel preference with the appointment or patient scheduling record.
3. The system does not send a reminder to a channel the patient did not select.

---

### PRD-FR-007 — Reminder Delivery

- **Priority:** Must
- **Feature:** FEAT-004
- **Persona:** PERSONA-001

**Requirement**

The system shall send appointment reminders through the patient-selected channel or channels using content limited to appointment logistics.

**Business Rationale**

Reminders can reduce missed appointments while minimizing privacy risk in a HIPAA-regulated healthcare environment.

**Acceptance Criteria**

1. Reminder messages are generated only for appointments with an active selected reminder channel.
2. Reminder message content excludes diagnosis, treatment details, and sensitive PHI beyond appointment logistics.
3. Reminder delivery attempts are recorded for staff or operational review.

---

### PRD-FR-008 — Staff Daily Schedule View

- **Priority:** Must
- **Feature:** FEAT-005
- **Persona:** PERSONA-002

**Requirement**

The system shall provide staff with a daily schedule view showing all appointments plus pending booking and reschedule requests.

**Business Rationale**

Staff need one place to understand the day’s schedule and pending scheduling work to manage clinic operations efficiently.

**Acceptance Criteria**

1. Staff can view all appointments scheduled for the selected day.
2. Each displayed appointment includes patient name, appointment time, provider, appointment type, and appointment status when those data are available from the scheduling source.
3. Staff can view pending booking requests in the same daily schedule area.
4. Staff can view pending reschedule requests in the same daily schedule area.

---

### PRD-FR-009 — Pending Request Review

- **Priority:** Should
- **Feature:** FEAT-005
- **Persona:** PERSONA-002

**Requirement**

The system shall allow scheduling staff to review pending booking and reschedule requests from the staff schedule workflow.

**Business Rationale**

Manual approval flows are useful only if staff can find and act on pending requests without switching contexts.

**Acceptance Criteria**

1. Staff can distinguish pending booking requests from pending reschedule requests.
2. Staff can open a pending request to view submitted scheduling details.
3. Staff can identify which requests are awaiting action.

## 9. Non-Functional Requirements

### PRD-NFR-001 — HIPAA-Conscious Reminder Content

- **Category:** Compliance
- **Priority:** Must

**Requirement**

The system shall prevent reminder templates from including diagnosis, treatment details, or sensitive PHI beyond appointment logistics.

**Acceptance Criteria**

1. Approved reminder templates contain only appointment logistics and exclude diagnosis, treatment details, and visit-reason details.
2. A HIPAA/privacy review can verify all launch reminder templates before production use.

### PRD-NFR-002 — Role-Based Access Control

- **Category:** Security
- **Priority:** Must

**Requirement**

The system shall restrict staff scheduling configuration access to clinic administrators and scheduling staff.

**Acceptance Criteria**

1. A user without clinic administrator or scheduling staff permissions cannot modify appointment type eligibility or confirmation mode.
2. A user with clinic administrator or scheduling staff permissions can access booking rule configuration.

### PRD-NFR-003 — Accessibility Compliance

- **Category:** Accessibility
- **Priority:** Must

**Requirement**

The patient-facing booking and rescheduling flows shall conform to WCAG 2.1 AA.

**Acceptance Criteria**

1. Patient-facing booking and rescheduling screens pass WCAG 2.1 AA accessibility evaluation before launch.
2. Patients can complete booking and rescheduling flows using keyboard navigation and screen reader-compatible labels.

### PRD-NFR-004 — Secure Rescheduling Verification

- **Category:** Privacy
- **Priority:** Must

**Requirement**

The system shall require secure verification before a patient can reschedule an appointment online.

**Acceptance Criteria**

1. A patient cannot view or change an appointment for rescheduling until secure verification succeeds.
2. Failed verification does not expose appointment details.

## 10. User Journeys

### JOURNEY-001 — Patient Books an Eligible Appointment Online

1. Patient opens the online booking flow outside or during office hours.
2. Patient selects an eligible appointment type and available slot.
3. Patient enters required contact and demographic details.
4. System confirms the booking instantly or creates a pending request based on staff configuration.
5. Patient selects SMS, email, or both for reminders.

**Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-003, PRD-FR-006, PRD-NFR-001, PRD-NFR-003

### JOURNEY-002 — Patient Requests Booking When No Slot Is Available

1. Patient selects an eligible appointment type.
2. System finds no eligible slots available.
3. System displays no availability and offers a staff-reviewed booking request.
4. Patient submits the booking request.
5. Staff sees the request in the pending request queue.

**Related Requirements:** PRD-FR-004, PRD-FR-008, PRD-FR-009, PRD-NFR-003

### JOURNEY-003 — Patient Reschedules an Eligible Appointment

1. Patient starts the rescheduling flow.
2. System requires secure verification before showing appointment details.
3. Patient selects a new eligible slot before the staff-defined cutoff.
4. System updates the appointment or creates a pending request according to staff configuration.
5. Patient reminder preference remains available for the updated appointment.

**Related Requirements:** PRD-FR-005, PRD-FR-006, PRD-NFR-004, PRD-NFR-003

### JOURNEY-004 — Staff Reviews the Day’s Schedule

1. Scheduling staff opens the daily schedule view.
2. System displays all appointments for the day.
3. System displays pending booking and reschedule requests.
4. Staff reviews pending requests and identifies work awaiting action.

**Related Requirements:** PRD-FR-008, PRD-FR-009, PRD-NFR-002

## 11. Feature Breakdown

### FEAT-001 — Booking Rules Configuration

- **Summary:** Staff controls eligible appointment types and whether each type is instantly confirmed or manually approved.
- **Priority:** Must
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-NFR-002
- **Expected Spec File:** `specs/FEAT-001-booking-rules-configuration.spec.md`

### FEAT-002 — Online Booking

- **Summary:** New and existing patients can book eligible appointments online or submit staff-reviewed requests when no slot is available.
- **Priority:** Must
- **Related Requirements:** PRD-FR-003, PRD-FR-004, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-002-online-booking.spec.md`

### FEAT-003 — Online Rescheduling

- **Summary:** Patients can securely verify themselves and reschedule eligible appointments within staff-defined rules.
- **Priority:** Must
- **Related Requirements:** PRD-FR-005, PRD-NFR-004, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-003-online-rescheduling.spec.md`

### FEAT-004 — Appointment Reminders

- **Summary:** Patients choose SMS, email, or both for reminder delivery using HIPAA-conscious reminder content.
- **Priority:** Must
- **Related Requirements:** PRD-FR-006, PRD-FR-007, PRD-NFR-001
- **Expected Spec File:** `specs/FEAT-004-appointment-reminders.spec.md`

### FEAT-005 — Staff Daily Schedule

- **Summary:** Staff sees all daily appointments plus pending booking and reschedule requests in one operational view.
- **Priority:** Must
- **Related Requirements:** PRD-FR-008, PRD-FR-009, PRD-NFR-002
- **Expected Spec File:** `specs/FEAT-005-staff-daily-schedule.spec.md`

## 12. Constraints

- CON-001 — The product is for a healthcare clinic and must comply with HIPAA requirements.
- CON-002 — The company is a small business, so the first release should avoid unnecessary scope such as payments, insurance verification, and clinical intake forms.
- CON-003 — Reminder content must avoid diagnosis, treatment details, or sensitive PHI beyond appointment logistics.
- CON-004 — Emergency, urgent care, and clinically triaged appointment scenarios are excluded from launch.
- CON-005 — Scheduling system integration or data source is TBD until the current scheduling system is identified.
- CON-006 — Launch approach must begin with a pilot using one provider or appointment type before full rollout.

## 13. Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| RISK-001 | Current scheduling system is unknown, creating integration uncertainty | High | Medium | Resolve scheduling data source as an open question before technical design |
| RISK-002 | Reminder content could expose PHI if templates are too detailed | High | Medium | Require HIPAA/privacy review and limit reminders to appointment logistics |
| RISK-003 | Staff may configure too many appointment types for online booking before workflows are ready | Medium | Medium | Start with a pilot using one provider or appointment type |
| RISK-004 | Baseline and target metrics are unknown at launch | Medium | High | Instrument booking timestamps, phone workload, reminders, and staff view usage during rollout |
| RISK-005 | Patients may be blocked from rescheduling if secure verification is too difficult | Medium | Medium | Validate verification flow during pilot and monitor failed verification events |

## 14. Open Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| OQ-001 | What current scheduling system or data source will provide appointment availability, appointments, providers, and status? | Product Owner | Open | TBD |
| OQ-002 | What office hours definition should be used to measure outside-hours bookings? | Product Owner | Open | TBD |
| OQ-003 | What baseline and target should be set for completed online bookings outside office hours after rollout measurement begins? | Product Owner | Open | TBD |
| OQ-004 | What baseline and target should be set for staff phone booking workload reduction? | Product Owner | Open | TBD |
| OQ-005 | What baseline and target should be set for reminder adoption? | Product Owner | Open | TBD |
| OQ-006 | What baseline and target should be set for staff schedule visibility usage? | Product Owner | Open | TBD |
| OQ-007 | What exact secure verification method will be used before online rescheduling? | Product Owner | Open | TBD |
| OQ-008 | What staff-defined rescheduling cutoff should apply by appointment type? | Scheduling Staff | Open | TBD |
| OQ-009 | Which provider or appointment type will be used for the pilot rollout? | Product Owner | Open | TBD |
| OQ-010 | Which SMS and email communication services are approved for HIPAA-aligned reminder delivery? | Product Owner | Open | TBD |

## 15. Release and Rollout Considerations

- **Target Release:** TBD
- **Rollout Approach:** Pilot with one provider or appointment type before full clinic rollout.
- **Migration Considerations:** Scheduling data source and integration approach are TBD; migration needs depend on the current scheduling system.
- **Training/Support Considerations:** Clinic administrators and scheduling staff need training on appointment type eligibility, confirmation modes, pending request review, daily schedule view usage, reminder content limits, and rescheduling cutoff rules.

## 16. Definition of Product Success

The product is successful when:

- New and existing patients can complete online booking for eligible appointment types.
- Patients can securely reschedule eligible appointments within staff-defined rules.
- Patients can select SMS, email, or both for appointment reminders.
- Staff can view the day’s appointments and pending booking or reschedule requests in one place.
- Outside-hours booking capture, staff phone workload, reminder adoption, and staff schedule visibility are measured during rollout.
- All Must-priority requirements are accepted.
- Required quality, security, and compliance outcomes are evidenced.

## 17. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Product Owner | TBD | Pending | TBD |
| Business Sponsor | TBD | Pending | TBD |