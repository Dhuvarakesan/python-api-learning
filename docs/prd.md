# Product Requirements Document (PRD)

> Replace all `<PLACEHOLDER>` values. Keep requirement IDs stable after approval so downstream ARD and feature specifications retain traceability.

## Document Metadata

- **Product/Application:** Online Patient Scheduling and Reminders
- **Document ID:** PRD-OPSR
- **Version:** 0.1
- **Status:** Draft
- **Owner:** TBD
- **Contributors:** TBD
- **Last Updated:** 2026-11-19

---

## 1. Executive Summary

The clinic currently loses bookings because patients can only book appointments by phone during office hours. This product will allow patients to book selected standard, low-risk appointment types online, reschedule eligible appointments, and receive SMS and email reminders.

Existing patients will verify identity using a one-time code sent to phone or email before booking or rescheduling. New patients will be able to submit appointment requests for staff review without directly booking a slot. Staff will see the day’s schedule in one place, including appointment status and reminder confirmations, using the existing practice management/EHR scheduling system as the source of truth.

The first release prioritizes reducing lost bookings outside office hours while protecting PHI under HIPAA and keeping staff workflows centered on the existing schedule system.

## 2. Problem Statement

Patients who want to book appointments outside office hours cannot complete bookings because the clinic currently supports booking by phone only during office hours. This creates lost booking opportunities for the clinic and inconvenience for patients. Staff also need a consolidated view of the day’s schedule, appointment status, and reminder confirmations.

## 3. Product Vision

Patients can complete routine appointment scheduling tasks online at their convenience, while staff retain control over availability, appointment types, and new-patient review. The scheduling experience is HIPAA-conscious, integrated with the existing practice management/EHR schedule, and gives staff one operational view of the day.

## 4. Goals and Success Measures

### 4.1 Goals

- Reduce lost bookings outside office hours.
- Enable existing patients to book and reschedule selected standard, low-risk appointments online.
- Enable new patients to submit appointment requests for staff review.
- Send appointment reminders by SMS and email with confirmation, cancellation, and reschedule-request actions.
- Give staff one daily schedule view showing appointments, statuses, and reminder confirmations.
- Maintain HIPAA-compliant handling of PHI and reminder content.

### 4.2 Success Measures

| ID | Measure | Baseline | Target | Measurement Method |
|---|---|---:|---:|---|
| KPI-001 | Outside-hours online bookings | TBD | Any measurable outside-hours online bookings in the first month; target set after baseline | Count bookings completed online outside office hours |
| KPI-002 | Total booking volume trend | TBD | Positive trend after launch | Compare total booking volume before and after launch |
| KPI-003 | Reminder action tracking | TBD | TBD | Track reminder confirmations, cancellations, and reschedule requests |
| KPI-004 | Staff daily schedule visibility | TBD | Staff can view the day’s appointments, statuses, and reminder confirmations in one place | Staff acceptance testing |

## 5. Scope

### 5.1 In Scope

- Online booking for selected standard, low-risk appointment types.
- Existing-patient identity verification by one-time code sent to phone or email.
- Existing-patient online rescheduling outside a clinic-defined cutoff window.
- New-patient appointment requests routed to staff review without booking a slot.
- SMS and email appointment reminders.
- Reminder actions to confirm, cancel, or request reschedule.
- Staff daily schedule view showing appointments, appointment status, and reminder confirmations.
- Integration with the existing practice management/EHR scheduling system as the source of truth.
- Explicit handling of no matching slots, failed identity match, and temporary scheduling integration unavailability.
- WCAG 2.1 AA accessibility for patient-facing booking flows.
- HIPAA-compliant handling of PHI and reminder content.

### 5.2 Out of Scope

- Direct online booking for all appointment types.
- Direct booking by new patients without staff review.
- Intake form completion from reminders.
- Phone call reminders.
- Replacing the existing practice management/EHR scheduling system.
- Rescheduling inside the clinic-defined cutoff window.

## 6. Personas and Users

### PERSONA-001 — Existing Patient

- **Description:** A patient already known to the clinic who wants to book or reschedule selected standard, low-risk appointments online.
- **Primary Need:** Complete scheduling tasks without calling during office hours.
- **Key Pain Point:** Phone-only booking is unavailable when the office is closed.

### PERSONA-002 — New Patient

- **Description:** A patient not yet established with the clinic who wants to request an appointment online.
- **Primary Need:** Submit an appointment request without calling during office hours.
- **Key Pain Point:** Cannot directly access clinic scheduling outside phone hours.

### PERSONA-003 — Clinic Staff

- **Description:** Clinic team members responsible for managing schedules, reviewing requests, and monitoring appointment status.
- **Primary Need:** See the day’s schedule, statuses, and reminder confirmations in one place.
- **Key Pain Point:** Scheduling activity and reminder responses are difficult to manage without a consolidated daily view.

## 7. Assumptions and Dependencies

### Assumptions

- ASM-001 — The clinic can identify which appointment types are standard and low-risk for online booking.
- ASM-002 — Existing patient records contain phone or email data sufficient for one-time code verification.
- ASM-003 — The clinic will define the rescheduling cutoff window before launch.
- ASM-004 — Reminder content can be configured to meet HIPAA minimum-necessary expectations.
- ASM-005 — Staff will continue using the existing practice management/EHR scheduling system as the operational source of truth.

### Dependencies

- DEP-001 — Existing practice management/EHR scheduling system integration for availability, appointment creation, updates, and daily schedule data.
- DEP-002 — SMS and email delivery services suitable for clinic reminder workflows.
- DEP-003 — One-time code delivery through phone or email.
- DEP-004 — Clinic-approved list of appointment types eligible for online booking.
- DEP-005 — HIPAA compliance review of PHI handling, authentication, reminder content, and vendor obligations.

## 8. Functional Requirements

### PRD-FR-001 — Existing Patient Verification

- **Priority:** Must
- **Feature:** FEAT-001
- **Persona:** PERSONA-001

**Requirement**

The system shall verify existing patients using a one-time code sent to the patient’s phone or email before allowing online booking or rescheduling.

**Business Rationale**

Verification protects patient scheduling access while supporting convenient online self-service.

**Acceptance Criteria**

1. An existing patient cannot complete booking or rescheduling until a valid one-time code is submitted.
2. If the patient identity cannot be matched, the system prevents booking or rescheduling and displays a staff-contact or request-review path.
3. The verification event is recorded for audit or operational review without exposing unnecessary PHI in patient-facing messages.

---

### PRD-FR-002 — Online Booking for Eligible Appointment Types

- **Priority:** Must
- **Feature:** FEAT-001
- **Persona:** PERSONA-001

**Requirement**

The system shall allow verified existing patients to book only clinic-selected standard, low-risk appointment types online.

**Business Rationale**

Restricting online booking to approved appointment types reduces scheduling risk while addressing outside-hours booking loss.

**Acceptance Criteria**

1. The booking flow shows only appointment types marked eligible by staff or clinic configuration.
2. A verified existing patient can select an available eligible appointment slot and complete booking online.
3. A completed booking is reflected in the source scheduling system.
4. Ineligible appointment types are not available for direct online booking.

---

### PRD-FR-003 — No Available Slot Handling

- **Priority:** Must
- **Feature:** FEAT-001
- **Persona:** PERSONA-001

**Requirement**

The system shall handle cases where no available appointment slots match the patient’s selected criteria.

**Business Rationale**

Patients need a clear next step when online booking cannot present a matching slot.

**Acceptance Criteria**

1. When no matching slots exist, the system displays that no matching availability is available.
2. The system does not create an appointment when no matching slot is selected.
3. The patient is offered an alternate path consistent with clinic workflow, such as changing criteria or contacting/requesting staff assistance.

---

### PRD-FR-004 — Existing Patient Rescheduling

- **Priority:** Must
- **Feature:** FEAT-002
- **Persona:** PERSONA-001

**Requirement**

The system shall allow verified existing patients to reschedule eligible appointments only outside a clinic-defined cutoff window.

**Business Rationale**

Online rescheduling improves patient convenience while preserving clinic control over near-term schedule changes.

**Acceptance Criteria**

1. If an appointment is outside the cutoff window, the patient can select an eligible replacement slot and complete rescheduling.
2. If an appointment is inside the cutoff window, the system prevents direct rescheduling and displays the clinic-approved next step.
3. A completed reschedule updates the source scheduling system and removes or updates the prior slot accordingly.
4. The system uses the clinic-defined cutoff value once configured.

---

### PRD-FR-005 — New Patient Appointment Requests

- **Priority:** Must
- **Feature:** FEAT-003
- **Persona:** PERSONA-002

**Requirement**

The system shall allow new patients to create pending appointment requests for staff review without booking a confirmed slot.

**Business Rationale**

New-patient staff review protects clinic intake and scheduling controls while still capturing demand outside office hours.

**Acceptance Criteria**

1. A new patient can submit an appointment request online.
2. Submitted new-patient requests are marked pending and are not placed as confirmed appointments on the schedule.
3. Staff can identify pending new-patient requests for review.
4. The system communicates to the patient that the request requires staff review.

---

### PRD-FR-006 — SMS and Email Reminders

- **Priority:** Must
- **Feature:** FEAT-004
- **Persona:** PERSONA-001

**Requirement**

The system shall send appointment reminders through SMS and email.

**Business Rationale**

Reminders reduce missed appointments and give patients a convenient channel to act on scheduled visits.

**Acceptance Criteria**

1. The system can send an SMS reminder for a scheduled appointment when patient contact data supports SMS.
2. The system can send an email reminder for a scheduled appointment when patient contact data supports email.
3. Reminder content follows clinic-approved HIPAA-conscious content rules.
4. Reminder delivery status is available for staff or operational review.

---

### PRD-FR-007 — Reminder Actions

- **Priority:** Must
- **Feature:** FEAT-004
- **Persona:** PERSONA-001

**Requirement**

The system shall allow patients receiving reminders to confirm, cancel, or request reschedule from the reminder.

**Business Rationale**

Actionable reminders improve schedule reliability and give staff earlier visibility into patient intent.

**Acceptance Criteria**

1. A patient can confirm an appointment from a reminder-supported flow.
2. A patient can cancel an appointment from a reminder-supported flow.
3. A patient can request reschedule from a reminder-supported flow.
4. Staff can see confirmation, cancellation, and reschedule-request status associated with the appointment or request.

---

### PRD-FR-008 — Staff Daily Schedule View

- **Priority:** Must
- **Feature:** FEAT-005
- **Persona:** PERSONA-003

**Requirement**

The system shall provide staff with a daily schedule view showing appointments, appointment status, and reminder confirmations in one place.

**Business Rationale**

Staff need a consolidated daily view to manage schedule operations efficiently.

**Acceptance Criteria**

1. Staff can view the current day’s appointments in a single schedule view.
2. Each appointment displays its current status when available from the source scheduling system.
3. Reminder confirmation status is visible alongside the related appointment.
4. The schedule view uses the existing practice management/EHR scheduling system as the source of truth.

---

### PRD-FR-009 — Scheduling System Integration Unavailability

- **Priority:** Must
- **Feature:** FEAT-006
- **Persona:** PERSONA-001

**Requirement**

The system shall handle temporary unavailability of the existing practice management/EHR scheduling integration.

**Business Rationale**

Patients and staff need safe behavior when availability or schedule updates cannot be confirmed.

**Acceptance Criteria**

1. If the scheduling integration is unavailable, the system does not create or modify confirmed appointments without source-system confirmation.
2. The patient receives a clear message that online scheduling is temporarily unavailable or that staff follow-up is required.
3. Staff can identify scheduling attempts or requests affected by integration unavailability.

## 9. Non-Functional Requirements

### PRD-NFR-001 — HIPAA-Conscious PHI Handling

- **Category:** Compliance
- **Priority:** Must

**Requirement**

The system shall handle PHI in booking, rescheduling, reminders, identity verification, and staff views in a manner approved for HIPAA compliance by the clinic.

**Acceptance Criteria**

1. HIPAA compliance review approves the PHI handling design before production launch.
2. SMS and email reminder templates are approved for minimum-necessary PHI disclosure before production use.
3. Vendors that process PHI are reviewed for HIPAA suitability before production use.

### PRD-NFR-002 — Patient-Facing Accessibility

- **Category:** Accessibility
- **Priority:** Must

**Requirement**

The system shall meet WCAG 2.1 AA for patient-facing booking, rescheduling, reminder action, and new-patient request flows.

**Acceptance Criteria**

1. Patient-facing flows pass WCAG 2.1 AA accessibility review before launch.
2. Keyboard-only users can complete supported patient-facing flows.
3. Screen-reader users can understand form labels, errors, and confirmation states in supported patient-facing flows.

### PRD-NFR-003 — Source-of-Truth Data Integrity

- **Category:** Reliability
- **Priority:** Must

**Requirement**

The system shall treat the existing practice management/EHR scheduling system as the source of truth for availability, bookings, rescheduling, and daily schedule display.

**Acceptance Criteria**

1. Confirmed online bookings are not displayed as complete unless accepted by the source scheduling system.
2. Confirmed reschedules are not displayed as complete unless accepted by the source scheduling system.
3. The staff daily schedule view identifies or avoids stale schedule data when integration data cannot be refreshed.

## 10. User Journeys

### JOURNEY-001 — Existing Patient Books Online Outside Office Hours

1. Existing patient opens the online booking flow outside office hours.
2. Patient verifies identity using a one-time code sent to phone or email.
3. Patient selects an eligible standard, low-risk appointment type and available slot.
4. System books the appointment through the source scheduling system.
5. Patient receives booking confirmation and the appointment appears in the staff daily schedule view.

**Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-003, PRD-FR-009, PRD-NFR-001, PRD-NFR-003

### JOURNEY-002 — Existing Patient Reschedules Appointment

1. Existing patient enters the rescheduling flow and verifies identity with a one-time code.
2. System checks whether the appointment is outside the clinic-defined cutoff window.
3. Patient selects an eligible replacement slot.
4. System updates the appointment through the source scheduling system.
5. Staff can see the updated appointment in the daily schedule view.

**Related Requirements:** PRD-FR-001, PRD-FR-004, PRD-FR-009, PRD-NFR-003

### JOURNEY-003 — Patient Responds to Reminder

1. Patient receives an SMS or email appointment reminder.
2. Patient chooses to confirm, cancel, or request reschedule.
3. System records the selected action.
4. Staff can see the resulting status or request in the daily schedule context.

**Related Requirements:** PRD-FR-006, PRD-FR-007, PRD-FR-008, PRD-NFR-001

### JOURNEY-004 — New Patient Requests Appointment

1. New patient opens the online appointment request flow.
2. New patient submits an appointment request.
3. System creates a pending request without booking a confirmed slot.
4. Staff reviews the pending request.

**Related Requirements:** PRD-FR-005, PRD-FR-008, PRD-NFR-001, PRD-NFR-002

## 11. Feature Breakdown

### FEAT-001 — Existing Patient Online Booking

- **Summary:** Verified existing patients can book clinic-selected standard, low-risk appointment types online.
- **Priority:** Must
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-003, PRD-FR-009, PRD-NFR-001, PRD-NFR-002, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-001-existing-patient-online-booking.spec.md`

### FEAT-002 — Existing Patient Rescheduling

- **Summary:** Verified existing patients can reschedule eligible appointments outside the clinic-defined cutoff window.
- **Priority:** Must
- **Related Requirements:** PRD-FR-001, PRD-FR-004, PRD-FR-009, PRD-NFR-001, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-002-existing-patient-rescheduling.spec.md`

### FEAT-003 — New Patient Appointment Requests

- **Summary:** New patients can submit pending appointment requests for staff review without booking a slot.
- **Priority:** Must
- **Related Requirements:** PRD-FR-005, PRD-NFR-001, PRD-NFR-002
- **Expected Spec File:** `specs/FEAT-003-new-patient-appointment-requests.spec.md`

### FEAT-004 — Appointment Reminders and Actions

- **Summary:** Patients receive SMS and email reminders and can confirm, cancel, or request reschedule.
- **Priority:** Must
- **Related Requirements:** PRD-FR-006, PRD-FR-007, PRD-NFR-001
- **Expected Spec File:** `specs/FEAT-004-appointment-reminders-and-actions.spec.md`

### FEAT-005 — Staff Daily Schedule View

- **Summary:** Staff can view the day’s appointments, appointment statuses, and reminder confirmations in one place.
- **Priority:** Must
- **Related Requirements:** PRD-FR-008, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-005-staff-daily-schedule-view.spec.md`

### FEAT-006 — Scheduling Integration Resilience

- **Summary:** The system safely handles unavailable scheduling-system integration without creating unconfirmed appointments.
- **Priority:** Must
- **Related Requirements:** PRD-FR-009, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-006-scheduling-integration-resilience.spec.md`

## 12. Constraints

- CON-001 — The clinic is a small business, so the solution should avoid unnecessary operational complexity.
- CON-002 — The product operates in healthcare and must support HIPAA-compliant handling of PHI.
- CON-003 — The existing practice management/EHR scheduling system is the source of truth.
- CON-004 — Direct online booking is limited to selected standard, low-risk appointment types.
- CON-005 — Existing-patient rescheduling is limited to outside a clinic-defined cutoff window.
- CON-006 — New patients can create pending requests only and cannot directly book confirmed slots.

## 13. Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| RISK-001 | Existing practice management/EHR integration cannot support required booking, rescheduling, or status data | High | TBD | Validate integration capabilities before finalizing implementation scope |
| RISK-002 | Reminder content exposes more PHI than clinic policy allows | High | TBD | Require HIPAA-focused review and approval of reminder templates before launch |
| RISK-003 | Patient identity cannot be reliably matched for one-time code verification | High | TBD | Provide safe failure handling and staff-assisted path when identity match fails |
| RISK-004 | Staff workflow disruption reduces adoption | Medium | TBD | Keep the source scheduling system as source of truth and provide a focused daily view |
| RISK-005 | Undefined cutoff window or eligible appointment list delays launch | Medium | TBD | Require clinic decisions on appointment eligibility and cutoff rules before release |

## 14. Open Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| OQ-001 | What is the baseline monthly count of outside-hours bookings, total bookings, reminder confirmations, cancellations, and reschedule requests? | Product Owner | Open | TBD |
| OQ-002 | What numeric targets should replace “any measurable outside-hours online bookings” after baseline is known? | Product Owner | Open | TBD |
| OQ-003 | Which exact appointment types are approved as standard, low-risk, and eligible for online booking? | Clinic Operations | Open | TBD |
| OQ-004 | What is the clinic-defined cutoff window for online rescheduling? | Clinic Operations | Open | TBD |
| OQ-005 | Which practice management/EHR system and integration capabilities are available? | Technical Owner | Open | TBD |
| OQ-006 | What clinic-approved next step should patients see when no slots match, identity cannot be matched, or the scheduling integration is unavailable? | Product Owner | Open | TBD |
| OQ-007 | What is the target release date? | Product Owner | Open | TBD |
| OQ-008 | What reminder timing rules should apply for SMS and email reminders? | Clinic Operations | Open | TBD |
| OQ-009 | What production support owner and escalation process will handle failed bookings, reminder failures, and pending requests? | Clinic Operations | Open | TBD |

## 15. Release and Rollout Considerations

- **Target Release:** TBD
- **Rollout Approach:** Launch with selected standard, low-risk appointment types and staff-reviewed new-patient requests.
- **Migration Considerations:** No replacement of the existing practice management/EHR scheduling system is in scope; integration with the existing source of truth is required.
- **Training/Support Considerations:** Staff need training on the daily schedule view, pending new-patient requests, reminder confirmations, cancellation indicators, reschedule requests, and exception handling.

## 16. Definition of Product Success

The product is successful when:

- Existing patients can book eligible appointments online outside office hours.
- Existing patients can reschedule eligible appointments outside the clinic-defined cutoff window.
- New patients can submit pending appointment requests for staff review.
- SMS and email reminders support confirmation, cancellation, and reschedule-request actions.
- Staff can view the day’s appointments, statuses, and reminder confirmations in one place.
- Outside-hours online bookings are measurable in the first month and total booking volume trend can be assessed.
- All Must-priority requirements are accepted.
- Required quality, security, and compliance outcomes are evidenced.

## 17. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Product Owner | TBD | Pending | TBD |
| Business Sponsor | TBD | Pending | TBD |