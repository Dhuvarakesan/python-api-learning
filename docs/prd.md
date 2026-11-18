# Product Requirements Document (PRD)

> Requirement IDs should remain stable after approval so downstream ARD and feature specifications retain traceability.

## Document Metadata

- **Product/Application:** Clinic Online Scheduling and Reminders
- **Document ID:** PRD-COSR
- **Version:** 0.1
- **Status:** Draft
- **Owner:** TBD
- **Contributors:** TBD
- **Last Updated:** 2026-11-17

---

## 1. Executive Summary

The clinic currently loses bookings because patients can only book appointments by phone during office hours. Staff report frequent after-hours voicemails and callbacks, creating missed booking opportunities and operational friction for a small healthcare business operating under HIPAA requirements.

This product will allow existing patients to book online from preset appointment slots, allow new patients to submit intake requests for staff approval, allow patients to request rescheduling online, and send appointment reminders by email and SMS. Front desk schedulers and providers will use a daily schedule view that shows confirmed appointments and pending booking or reschedule requests in one place.

The first release prioritizes reducing missed booking opportunities outside office hours while maintaining staff control over scheduling decisions, protecting PHI, and integrating with the clinic’s existing scheduling system.

## 2. Problem Statement

Patients who need to book or change appointments outside office hours must leave voicemail or wait until the clinic reopens. This creates lost booking opportunities, delayed callbacks, and administrative burden for front desk staff. Staff currently lack a single online view that combines the day’s confirmed schedule with pending online booking and reschedule requests.

## 3. Product Vision

Patients can securely submit booking and reschedule requests online at any time, receive timely reminders, and reduce reliance on phone-only scheduling. Staff can manage preset availability, approve new-patient intake and reschedule requests, and view the day’s schedule from a single HIPAA-compliant workflow connected to the existing scheduling system.

## 4. Goals and Success Measures

### 4.1 Goals

- Reduce missed booking opportunities outside office hours.
- Let existing patients book from preset appointment slots online.
- Let new patients submit intake requests that staff approve before scheduling.
- Let patients request rescheduling online, subject to clinic rules and staff approval.
- Send email and SMS reminders 72 hours and 24 hours before confirmed appointments.
- Give front desk schedulers and providers a unified daily view of confirmed appointments and pending requests.
- Maintain HIPAA-compliant handling of patient data using eligible vendors, BAAs, and encryption.

### 4.2 Success Measures

| ID | Measure | Baseline | Target | Measurement Method |
|---|---|---:|---:|---|
| KPI-001 | After-hours online booking submissions | Not yet established | Establish baseline during first release; numeric target not yet set | Count booking submissions created outside office hours |
| KPI-002 | Online reschedule requests submitted | Not yet established | Establish baseline during first release; numeric target not yet set | Count reschedule requests submitted online |
| KPI-003 | Reminder delivery coverage for confirmed appointments | Not yet established | Establish baseline during first release; numeric target not yet set | Compare confirmed appointments against scheduled reminder events |
| KPI-004 | Staff visibility into daily schedule and pending requests | Not yet established | Front desk schedulers and providers can access the daily view in production | Staff access verification and user acceptance testing |

## 5. Scope

### 5.1 In Scope

- Online booking for existing patients using preset appointment slots managed by staff.
- Online new-patient intake requests that require staff approval before scheduling.
- Online reschedule requests up to 24 hours before the appointment, requiring staff approval.
- Email and SMS appointment reminders at 72 hours and 24 hours before confirmed appointments.
- Staff daily schedule view for front desk schedulers and providers.
- Display of confirmed appointments plus pending booking and reschedule requests.
- One-time code verification by email or SMS before request submission.
- Integration with the clinic’s existing scheduling system.
- HIPAA-compliant vendor stack, business associate agreements, and encrypted PHI handling.

### 5.2 Out of Scope

- Clinical triage or medical advice routing.
- Self-service rescheduling without staff approval.
- Phone-based scheduling workflow redesign beyond supporting online requests.
- Numeric KPI targets beyond baseline establishment for the first release.
- Selection of exact vendors, scheduling-system integration details, and production support model until dependencies are confirmed.

## 6. Personas and Users

### PERSONA-001 — Existing Patient

- **Description:** A patient already known to the clinic who wants to book or request changes without calling during office hours.
- **Primary Need:** Securely book from available preset slots and request rescheduling online.
- **Key Pain Point:** Phone-only scheduling is unavailable after hours and creates delays.

### PERSONA-002 — New Patient

- **Description:** A prospective patient who wants to initiate care with the clinic.
- **Primary Need:** Submit an intake request online for staff review and scheduling approval.
- **Key Pain Point:** Cannot start the appointment request process outside office hours without calling or leaving voicemail.

### PERSONA-003 — Front Desk Scheduler

- **Description:** Clinic staff member responsible for managing appointment requests, approving schedule changes, and maintaining the day’s schedule.
- **Primary Need:** See confirmed appointments and pending booking or reschedule requests in one place.
- **Key Pain Point:** After-hours voicemails and callbacks create manual work and missed booking opportunities.

### PERSONA-004 — Provider

- **Description:** Clinician or provider who needs visibility into the day’s schedule.
- **Primary Need:** View confirmed appointments and relevant pending schedule activity for the day.
- **Key Pain Point:** Schedule visibility is fragmented when requests are handled outside a unified view.

## 7. Assumptions and Dependencies

### Assumptions

- ASM-001 — Staff will manage preset appointment slots for online booking.
- ASM-002 — New-patient intake requests require staff approval before becoming confirmed appointments.
- ASM-003 — Reschedule requests require staff approval and are allowed only up to 24 hours before the existing appointment.
- ASM-004 — The first release will establish baselines for success metrics instead of committing to numeric targets.
- ASM-005 — Online booking must be available 24/7 except planned maintenance.

### Dependencies

- DEP-001 — A HIPAA-eligible vendor stack must be selected, with BAAs in place before launch.
- DEP-002 — The product must integrate with the clinic’s existing scheduling system.
- DEP-003 — Email and SMS delivery services must support HIPAA-compliant handling of PHI or use clinic-approved reminder content rules.
- DEP-004 — Staff must define preset appointment slots and appointment types before launch.
- DEP-005 — Clinic staff must define office hours so after-hours submissions can be measured.

## 8. Functional Requirements

### PRD-FR-001 — Patient Verification

- **Priority:** Must
- **Feature:** FEAT-001
- **Persona:** PERSONA-001

**Requirement**

The system shall require a one-time code sent by email or SMS before a patient can submit an online booking or reschedule request.

**Business Rationale**

Verification reduces unauthorized request submission while keeping access lighter than full portal login.

**Acceptance Criteria**

1. A patient cannot submit a booking request until a valid one-time code is entered.
2. A patient cannot submit a reschedule request until a valid one-time code is entered.
3. An invalid or expired code produces a visible error and does not submit the request.
4. The system records the verification method used for the submitted request.

---

### PRD-FR-002 — Existing Patient Online Booking

- **Priority:** Must
- **Feature:** FEAT-002
- **Persona:** PERSONA-001

**Requirement**

The system shall allow existing patients to submit booking requests by selecting from preset appointment slots managed by staff.

**Business Rationale**

Online booking from controlled slots reduces missed after-hours booking opportunities while preserving clinic control over availability.

**Acceptance Criteria**

1. Existing patients can view available preset appointment slots after completing required verification.
2. Selecting an available slot and submitting required information creates a booking request.
3. A submitted booking request is visible to staff in the pending requests area.
4. If no preset slots are available, the patient sees a clear no-availability message and no booking request is created.

---

### PRD-FR-003 — New-Patient Intake Request

- **Priority:** Must
- **Feature:** FEAT-003
- **Persona:** PERSONA-002

**Requirement**

The system shall allow new patients to submit online intake requests that staff must approve before scheduling.

**Business Rationale**

New patients can initiate care outside office hours while staff retain control over acceptance and scheduling.

**Acceptance Criteria**

1. A new patient can submit an intake request after completing required verification.
2. A submitted new-patient intake request is marked as pending staff approval.
3. Pending new-patient intake requests appear in the staff daily schedule or pending requests view.
4. The system does not create a confirmed appointment from a new-patient intake request until staff approve it.

---

### PRD-FR-004 — Patient Reschedule Request

- **Priority:** Must
- **Feature:** FEAT-004
- **Persona:** PERSONA-001

**Requirement**

The system shall allow patients to request rescheduling to a preset slot up to 24 hours before the existing appointment, with staff approval required.

**Business Rationale**

Online reschedule requests reduce phone dependence while preserving staff oversight and protecting near-term schedules.

**Acceptance Criteria**

1. A patient can submit a reschedule request only after completing required verification.
2. The system blocks reschedule requests for appointments less than 24 hours away and shows a clear message.
3. A valid reschedule request remains pending until staff approve or reject it.
4. The existing appointment remains unchanged until staff approve the reschedule request.

---

### PRD-FR-005 — Staff Request Approval

- **Priority:** Must
- **Feature:** FEAT-005
- **Persona:** PERSONA-003

**Requirement**

The system shall allow front desk schedulers to approve or reject pending booking, new-patient intake, and reschedule requests.

**Business Rationale**

Staff approval ensures the online workflow fits clinic scheduling rules and avoids unintended schedule changes.

**Acceptance Criteria**

1. Staff can see each pending request type and its submitted details.
2. Staff can approve a pending request and cause the corresponding appointment or schedule change to be reflected in the scheduling system.
3. Staff can reject a pending request without creating or changing an appointment.
4. Approved and rejected requests no longer appear as pending.

---

### PRD-FR-006 — Appointment Reminders

- **Priority:** Must
- **Feature:** FEAT-006
- **Persona:** PERSONA-001

**Requirement**

The system shall send appointment reminders by email and SMS 72 hours and 24 hours before confirmed appointments.

**Business Rationale**

Reminders support patient attendance and reduce reliance on staff phone follow-up.

**Acceptance Criteria**

1. A confirmed appointment has reminder events scheduled for 72 hours and 24 hours before the appointment time.
2. Reminder events are created for both email and SMS when those contact channels are available.
3. Reminder delivery failures are recorded for staff review or operational follow-up.
4. Pending, rejected, or unapproved requests do not trigger appointment reminders.

---

### PRD-FR-007 — Staff Daily Schedule View

- **Priority:** Must
- **Feature:** FEAT-007
- **Persona:** PERSONA-003

**Requirement**

The system shall provide a daily schedule view showing confirmed appointments plus pending booking and reschedule requests for front desk schedulers and providers.

**Business Rationale**

A single view reduces fragmented scheduling work and gives staff a shared understanding of the day’s confirmed and pending activity.

**Acceptance Criteria**

1. Front desk schedulers can view confirmed appointments for the selected day.
2. Providers can view confirmed appointments for the selected day.
3. Pending booking and reschedule requests are visible in the daily view or directly accessible from it.
4. The view distinguishes confirmed appointments from pending requests.

---

### PRD-FR-008 — Scheduling System Integration

- **Priority:** Must
- **Feature:** FEAT-008
- **Persona:** PERSONA-003

**Requirement**

The system shall integrate with the clinic’s existing scheduling system to reflect confirmed appointments, approved bookings, and approved reschedule changes.

**Business Rationale**

Integration prevents duplicate scheduling workflows and keeps staff working from the clinic’s source of truth.

**Acceptance Criteria**

1. Confirmed appointments from the scheduling system are available to the staff daily schedule view.
2. An approved online booking is reflected in the scheduling system.
3. An approved reschedule request updates the appointment in the scheduling system.
4. If integration fails, the system records the failure and does not silently mark the scheduling update as complete.

---

### PRD-FR-009 — After-Hours Submission Tracking

- **Priority:** Should
- **Feature:** FEAT-009
- **Persona:** PERSONA-003

**Requirement**

The system shall track whether booking requests are submitted outside clinic office hours.

**Business Rationale**

The first release needs to establish a baseline for the primary success metric: after-hours online booking submissions.

**Acceptance Criteria**

1. Each booking request stores its submission timestamp.
2. Each booking request is classified as inside or outside office hours based on clinic-defined office hours.
3. Staff or administrators can report the count of after-hours online booking submissions for a selected period.

## 9. Non-Functional Requirements

### PRD-NFR-001 — HIPAA-Compliant PHI Handling

- **Category:** Compliance
- **Priority:** Must

**Requirement**

The system shall handle PHI using HIPAA-eligible services with business associate agreements in place and encryption for PHI in transit and at rest.

**Acceptance Criteria**

1. No production PHI is processed by a vendor unless a BAA is in place before launch.
2. PHI is encrypted in transit using HTTPS/TLS.
3. PHI is encrypted at rest in production storage.
4. Access to PHI is limited to authorized staff roles.

### PRD-NFR-002 — Availability

- **Category:** Availability
- **Priority:** Must

**Requirement**

The system shall make online booking and reschedule request submission available 24/7 except during planned maintenance.

**Acceptance Criteria**

1. Patients can access the online booking flow outside clinic office hours when no planned maintenance is active.
2. Planned maintenance prevents submission with a clear patient-facing message.
3. Planned maintenance windows are distinguishable from unplanned outages in operational records.

### PRD-NFR-003 — Security Auditability

- **Category:** Security
- **Priority:** Must

**Requirement**

The system shall maintain audit records for patient request submission, staff approval or rejection, reminder scheduling, and scheduling-system update failures.

**Acceptance Criteria**

1. Each patient request submission has an audit record with timestamp and request type.
2. Each staff approval or rejection has an audit record with timestamp and staff user identifier.
3. Each reminder delivery failure has an audit record.
4. Each scheduling-system integration failure has an audit record.

### PRD-NFR-004 — Accessibility

- **Category:** Accessibility
- **Priority:** Should

**Requirement**

The patient-facing booking and reschedule flows shall be usable by patients relying on common keyboard and screen reader interactions.

**Acceptance Criteria**

1. A patient can complete verification and submit a booking request using keyboard navigation.
2. Form fields expose labels that are announced by screen readers.
3. Validation errors are visible and programmatically associated with the relevant fields.

## 10. User Journeys

### JOURNEY-001 — Existing Patient Books Online After Hours

1. Existing patient accesses online scheduling outside office hours.
2. Patient verifies using a one-time code sent by email or SMS.
3. Patient selects a preset appointment slot and submits a booking request.
4. Staff sees the pending booking request in the daily schedule or pending requests view.
5. Staff approves the request and the appointment is reflected in the scheduling system.
6. The patient receives email and SMS reminders 72 hours and 24 hours before the appointment.

**Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-005, PRD-FR-006, PRD-FR-008, PRD-FR-009, PRD-NFR-001, PRD-NFR-002

### JOURNEY-002 — New Patient Submits Intake Request

1. New patient opens the online intake request flow.
2. New patient verifies using a one-time code sent by email or SMS.
3. New patient submits an intake request.
4. Staff reviews the pending intake request.
5. Staff approves the request before scheduling an appointment.

**Related Requirements:** PRD-FR-001, PRD-FR-003, PRD-FR-005, PRD-NFR-001

### JOURNEY-003 — Existing Patient Requests Reschedule

1. Existing patient accesses the reschedule flow.
2. Patient verifies using a one-time code sent by email or SMS.
3. Patient selects a preset slot at least 24 hours before the existing appointment.
4. Patient submits a reschedule request.
5. Staff approves or rejects the request.
6. Approved changes are reflected in the scheduling system.

**Related Requirements:** PRD-FR-001, PRD-FR-004, PRD-FR-005, PRD-FR-008, PRD-NFR-001

### JOURNEY-004 — Staff Reviews the Day’s Schedule

1. Front desk scheduler or provider opens the daily schedule view.
2. Staff reviews confirmed appointments for the day.
3. Staff reviews pending booking and reschedule requests from the same view or directly linked pending area.
4. Staff acts on pending requests as appropriate.

**Related Requirements:** PRD-FR-005, PRD-FR-007, PRD-FR-008, PRD-NFR-003

## 11. Feature Breakdown

### FEAT-001 — Patient Verification

- **Summary:** Verifies patient access using one-time codes sent by email or SMS before request submission.
- **Priority:** Must
- **Related Requirements:** PRD-FR-001, PRD-NFR-001, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-001-patient-verification.spec.md`

### FEAT-002 — Existing Patient Booking

- **Summary:** Allows existing patients to submit booking requests from staff-managed preset appointment slots.
- **Priority:** Must
- **Related Requirements:** PRD-FR-002, PRD-FR-009, PRD-NFR-001, PRD-NFR-002
- **Expected Spec File:** `specs/FEAT-002-existing-patient-booking.spec.md`

### FEAT-003 — New-Patient Intake Requests

- **Summary:** Allows prospective patients to submit intake requests for staff approval.
- **Priority:** Must
- **Related Requirements:** PRD-FR-003, PRD-FR-005, PRD-NFR-001
- **Expected Spec File:** `specs/FEAT-003-new-patient-intake-requests.spec.md`

### FEAT-004 — Reschedule Requests

- **Summary:** Allows existing patients to request rescheduling to preset slots up to 24 hours before the appointment.
- **Priority:** Must
- **Related Requirements:** PRD-FR-004, PRD-FR-005, PRD-NFR-001
- **Expected Spec File:** `specs/FEAT-004-reschedule-requests.spec.md`

### FEAT-005 — Staff Approval Workflow

- **Summary:** Allows front desk schedulers to review, approve, and reject pending booking, intake, and reschedule requests.
- **Priority:** Must
- **Related Requirements:** PRD-FR-005, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-005-staff-approval-workflow.spec.md`

### FEAT-006 — Appointment Reminders

- **Summary:** Sends email and SMS reminders 72 hours and 24 hours before confirmed appointments.
- **Priority:** Must
- **Related Requirements:** PRD-FR-006, PRD-NFR-001, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-006-appointment-reminders.spec.md`

### FEAT-007 — Staff Daily Schedule View

- **Summary:** Shows confirmed appointments and pending booking or reschedule requests for front desk schedulers and providers.
- **Priority:** Must
- **Related Requirements:** PRD-FR-007, PRD-FR-005, PRD-FR-008
- **Expected Spec File:** `specs/FEAT-007-staff-daily-schedule-view.spec.md`

### FEAT-008 — Scheduling System Integration

- **Summary:** Synchronizes confirmed appointments and approved request outcomes with the clinic’s existing scheduling system.
- **Priority:** Must
- **Related Requirements:** PRD-FR-008, PRD-NFR-003
- **Expected Spec File:** `specs/FEAT-008-scheduling-system-integration.spec.md`

### FEAT-009 — Reporting and Baseline Tracking

- **Summary:** Tracks after-hours booking submissions and other first-release baseline metrics.
- **Priority:** Should
- **Related Requirements:** PRD-FR-009
- **Expected Spec File:** `specs/FEAT-009-reporting-and-baseline-tracking.spec.md`

## 12. Constraints

- CON-001 — The clinic is a small healthcare business and must comply with HIPAA.
- CON-002 — Production launch depends on HIPAA-compliant vendor agreements and BAAs being in place.
- CON-003 — Online booking must use preset appointment slots managed by staff.
- CON-004 — New-patient intake and reschedule requests require staff approval.
- CON-005 — Reschedule requests are allowed only up to 24 hours before the appointment.
- CON-006 — The product must integrate with the clinic’s existing scheduling system.
- CON-007 — Clinical triage and medical advice routing are out of scope for the first release.

## 13. Risks

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| RISK-001 | HIPAA-compliant vendor agreements are delayed | High | Medium | Make BAA completion a launch dependency and select eligible vendors early |
| RISK-002 | Existing scheduling system integration is more complex than expected | High | Medium | Validate integration capabilities before implementation planning and define fallback operational handling |
| RISK-003 | Preset slots are not maintained by staff | Medium | Medium | Assign slot-management ownership and include slot review in launch readiness |
| RISK-004 | Email or SMS reminders expose PHI inappropriately | High | Medium | Use HIPAA-eligible messaging services and approve reminder content rules before launch |
| RISK-005 | Staff approval queues create delays that reduce patient benefit | Medium | Medium | Monitor pending request aging and adjust staffing workflow after baseline data is collected |

## 14. Open Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| OQ-001 | What numeric targets should be set for after-hours online booking submissions after the first-release baseline is established? | Product Owner | Open | TBD |
| OQ-002 | Which existing scheduling system will be integrated, and what integration methods are available? | Product Owner / Technical Lead | Open | TBD |
| OQ-003 | Which HIPAA-eligible vendors will provide scheduling, email, SMS, hosting, and storage services? | Product Owner / Compliance Owner | Open | TBD |
| OQ-004 | Which appointment types, providers, and preset slot rules are included at launch? | Clinic Operations Owner | Open | TBD |
| OQ-005 | What exact PHI may appear in email and SMS reminder content? | Compliance Owner | Open | TBD |
| OQ-006 | What staff roles beyond front desk schedulers and providers, if any, require access? | Product Owner | Open | TBD |

## 15. Release and Rollout Considerations

- **Target Release:** Launch as soon as HIPAA-compliant vendor agreements are in place.
- **Rollout Approach:** First release for existing-patient booking, new-patient intake requests, reschedule requests, reminders, and staff daily schedule view; exact provider or appointment-type rollout remains an open question.
- **Migration Considerations:** Existing confirmed appointments must be available through the scheduling system integration; preset appointment slots must be configured before launch.
- **Training/Support Considerations:** Front desk schedulers and providers need training on daily schedule review, pending request approval, rejection handling, and integration failure follow-up.

## 16. Definition of Product Success

The product is successful when:

- Patients can submit online booking requests outside office hours.
- Existing patients can book from preset appointment slots online.
- New patients can submit intake requests for staff approval.
- Existing patients can request rescheduling online up to 24 hours before an appointment.
- Staff can see confirmed appointments and pending booking or reschedule requests in one place.
- Email and SMS reminders are scheduled for confirmed appointments at 72 hours and 24 hours before appointment time.
- First-release baselines are established for after-hours online booking submissions and related scheduling activity.
- All Must-priority requirements are accepted.
- Required quality, security, and compliance outcomes are evidenced.

## 17. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Product Owner | TBD | Pending | TBD |
| Business Sponsor | TBD | Pending | TBD |