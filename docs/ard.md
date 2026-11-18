# Architecture Requirements Document (ARD)

> Replace all `<PLACEHOLDER>` values. Every material architecture decision or constraint should have a stable ID that feature specifications can reference.

## Document Metadata

- **Product/Application:** SpendWise — Personal Expense Tracker
- **Document ID:** ARD-SPENDWISE
- **Version:** 0.1
- **Status:** Draft
- **Architecture Owner:** TBD
- **Contributors:** Product Owner, Stakeholders, ARD Agent
- **Last Updated:** 2026-11-18
- **Reference PRD:** `PRD.md`

---

## 1. Architecture Summary

SpendWise will be implemented as a static, browser-based React application using Vite, React, and TypeScript. The application has no backend runtime, no account system, no external financial integrations, and no server-side storage for expense or budget data. It runs fully in the user's browser and persists expenses and monthly budgets in `localStorage` using JSON serialization and schema versioning.

The architecture separates presentation, application state, domain rules, persistence, and export concerns through React components, custom hooks, and small TypeScript services. Static hosting serves the compiled app over HTTPS, while all financial data remains in the same browser profile. CSV export is generated client-side from the filtered expense dataset.

Key drivers are simplicity for a small-business delivery context, local-only privacy, responsive usability, accessible controls, acceptance-test-driven release readiness, and automated CI gates covering type checking, linting, tests, accessibility checks, production build, dependency/CVE scanning, and SBOM generation.

## 2. Architecture Drivers

- PRD-FR-001 — The architecture must support adding validated expenses with amount, date, category, and optional description.
- PRD-FR-002 — The architecture must display persisted expenses newest first with INR currency formatting and empty states.
- PRD-FR-003 — The architecture must support editing expenses with the same validation rules as creation.
- PRD-FR-004 — The architecture must support confirmed deletion and summary recalculation.
- PRD-FR-005 — The architecture must support month and category filtering with clearable filters.
- PRD-FR-006 — The architecture must calculate monthly total, record count, category totals, and budget usage from local records.
- PRD-FR-007 — The architecture must support one monthly budget per month and status thresholds below 80%, 80% to 100%, and above 100%.
- PRD-FR-008 — The architecture must export filtered records as CSV with `YYYY-MM-DD` dates.
- PRD-FR-009 — The architecture must persist expenses and budgets in the same browser profile after refresh or reopening.
- PRD-NFR-001 — The architecture must be local-only with no account, authentication, external financial integration, or server-side financial storage.
- PRD-NFR-002 — The architecture must remain usable on common desktop and mobile browser viewport sizes.
- PRD-NFR-003 — The architecture must support visible labels, keyboard-operable controls, and field-associated validation feedback.
- PRD-NFR-004 — The architecture must support visible feedback after save, update, delete, and export actions.
- CON-001 — The first release must remain small and simple for a small-business delivery context.
- CON-002 — The first release must be a browser-based React application.
- CON-003 — Expense and budget data must remain local to the browser profile in the first release.
- CON-004 — The first release must not include account creation, login, cloud sync, or external financial integrations.
- CON-005 — Currency display must be INR (₹).
- CON-006 — Amounts must be positive and support up to two decimal places.
- CON-007 — The first release must use the fixed categories Food, Transport, Shopping, Bills, Health, Entertainment, and Other.
- CON-008 — CSV export dates must use `YYYY-MM-DD` format.
- CON-009 — Release can occur when Must-priority requirements pass acceptance testing.

## 3. Architecture Principles

1. Keep financial data local to the browser profile and avoid network transmission of expense or budget records.
2. Prefer simple static-web architecture over backend services unless a future approved requirement requires server-side capabilities.
3. Reuse established repository and platform patterns where appropriate.
4. Keep presentation, orchestration, domain, integration, and persistence concerns separated.
5. Make validation, calculation, persistence, and export logic testable outside UI components.
6. Preserve accessibility and responsive behavior as first-class architectural concerns, not final-stage styling tasks.

## 4. System Context

### 4.1 Actors

- PERSONA-001 — Uses the application in one browser profile to add, edit, delete, filter, summarize, budget, export, and retain personal expense records.
- Browser Runtime — Executes the static React application, provides DOM APIs, download APIs, and `localStorage`.
- Static Hosting Platform — Serves the compiled application assets over HTTPS with no application backend runtime.
- CI/CD Platform — Runs quality, build, security, SBOM, and deployment checks before release.

### 4.2 External Systems

| ID | System | Purpose | Interface | Owner |
|---|---|---|---|---|
| EXT-001 | Browser `localStorage` | Persist expenses, monthly budgets, and schema metadata in the user's browser profile | Web Storage API | User's browser vendor |
| EXT-002 | Browser File Download Capability | Save filtered expense records as a CSV file generated in the browser | File | User's browser vendor |
| EXT-003 | Static Hosting Platform | Serve compiled HTML, CSS, JavaScript, and static assets over HTTPS | File/HTTPS | Hosting provider TBD |
| EXT-004 | CI/CD Platform | Run automated checks, produce build artifacts, generate SBOM, and deploy static assets | Pipeline | Engineering team |

## 5. Technology Stack

| Layer | Technology | Version/Constraint | Rationale |
|---|---|---|---|
| Frontend | Vite + React + TypeScript | Exact versions TBD by engineering; must support current target browsers | Satisfies PRD React constraint, provides fast static builds, and makes domain/storage/export logic type-safe. |
| Backend | None | No backend runtime in first release | Supports PRD-NFR-001 local-only privacy and avoids server-side financial data storage. |
| Data | Browser `localStorage` with JSON serialization and schema versioning | Same browser profile only; quota and clearing behavior controlled by browser | Simple local persistence aligned with PRD-FR-009 and first-release scope. |
| Hosting | Static site hosting | HTTPS required; provider TBD | Matches no-backend architecture and supports low-operational-overhead deployment. |
| CI/CD | Automated pipeline | Must run type check, lint, unit tests, integration tests, accessibility checks, production build, dependency/CVE scan, and SBOM generation | Provides release confidence for Must-priority acceptance criteria and security baseline. |

## 6. Logical Architecture

### 6.1 Components

#### COMP-001 — App Shell and Routing/Layout

- **Responsibility:** Compose dashboard, form, history, summary, budget, and export areas in a responsive layout.
- **Inputs:** Application state, selected month, selected category, user actions.
- **Outputs:** Rendered UI, navigation/layout state, action callbacks to hooks and services.
- **Dependencies:** COMP-002, COMP-003, COMP-004, COMP-005, COMP-006, COMP-007.
- **Repository Area:** `src/app/`

#### COMP-002 — Expense Form Component

- **Responsibility:** Capture create/edit expense inputs, show field-level validation messages, and submit valid expense commands.
- **Inputs:** Existing expense for edit mode, current date default, fixed category list, form input events.
- **Outputs:** Create/update expense command or cancel action.
- **Dependencies:** COMP-005, COMP-006.
- **Repository Area:** `src/features/expenses/`

#### COMP-003 — Expense History and Filters Component

- **Responsibility:** Display expenses newest first, filter by month and category, clear filters, and initiate edit/delete flows.
- **Inputs:** Expense records, selected filters, user filter/edit/delete actions.
- **Outputs:** Filter state changes, edit selection, delete confirmation command.
- **Dependencies:** COMP-005, COMP-006.
- **Repository Area:** `src/features/history/`

#### COMP-004 — Monthly Summary Component

- **Responsibility:** Display monthly total, record count, category-wise totals, and budget used/remaining/exceeded.
- **Inputs:** Selected month, expense records, monthly budget data.
- **Outputs:** Summary UI and budget status display.
- **Dependencies:** COMP-006.
- **Repository Area:** `src/features/summary/`

#### COMP-005 — Application State Hooks

- **Responsibility:** Orchestrate expense, budget, filter, persistence, and UI feedback state for components.
- **Inputs:** Component commands, persisted snapshots from storage, domain service outputs.
- **Outputs:** Current view model, state mutation functions, save/update/delete/export feedback.
- **Dependencies:** COMP-006, COMP-007, COMP-008.
- **Repository Area:** `src/hooks/`

#### COMP-006 — Domain and Validation Services

- **Responsibility:** Enforce amount, date, category, description, budget, sorting, filtering, summary, and budget-status rules.
- **Inputs:** Expense records, budget records, selected filters, form inputs.
- **Outputs:** Validation results, sorted/filtered lists, monthly summaries, budget status values.
- **Dependencies:** None.
- **Repository Area:** `src/domain/`

#### COMP-007 — Local Storage Repository

- **Responsibility:** Read and write versioned JSON expense and budget data in browser `localStorage`.
- **Inputs:** Expense collection, budget collection, schema version, persistence commands.
- **Outputs:** Persisted snapshots, loaded snapshots, storage error results.
- **Dependencies:** EXT-001.
- **Repository Area:** `src/infrastructure/storage/`

#### COMP-008 — CSV Export Service

- **Responsibility:** Convert filtered expenses to CSV with `YYYY-MM-DD` dates and trigger browser download when records exist.
- **Inputs:** Filtered expense records.
- **Outputs:** CSV content, downloadable file, success/no-record feedback.
- **Dependencies:** EXT-002.
- **Repository Area:** `src/infrastructure/export/`

#### COMP-009 — Static Build and Deployment Pipeline

- **Responsibility:** Build, validate, scan, generate SBOM, publish, and roll back static application artifacts.
- **Inputs:** Source code, tests, dependency manifests, hosting configuration.
- **Outputs:** Production build artifact, preview deployment, production deployment, SBOM, scan results.
- **Dependencies:** EXT-003, EXT-004.
- **Repository Area:** `.github/workflows/` or `ci/` and `hosting/` TBD

### 6.2 Data Flow

1. PERSONA-001 opens the static SpendWise app served by EXT-003 in a browser runtime.
2. COMP-001 initializes the UI and asks COMP-005 to load persisted state.
3. COMP-005 requests COMP-007 to read versioned JSON expense and budget data from EXT-001.
4. COMP-002 sends expense create/edit input to COMP-006 for validation and normalization.
5. COMP-005 applies valid create/edit/delete commands and asks COMP-007 to persist the updated expense collection.
6. COMP-003 sends selected month and category filters to COMP-005 and COMP-006.
7. COMP-006 filters expenses, sorts records newest first, calculates monthly totals, category totals, record counts, and budget status.
8. COMP-004 displays summary values for the selected month and current budget.
9. COMP-008 receives filtered expenses, creates CSV content with `YYYY-MM-DD` dates, and triggers EXT-002 to download the file.
10. COMP-005 surfaces visible feedback for save, update, delete, export, validation, empty state, and storage error outcomes.

## 7. Architecture Decisions

### ARD-DEC-001 — Static Client-Only React Architecture

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001, PRD-NFR-002, PRD-NFR-003

**Decision**

SpendWise will be delivered as a static Vite + React + TypeScript application with no backend runtime in the first release.

**Rationale**

The PRD requires a browser-based React app, no account system, no financial integrations, no cloud sync, and no server-side storage of expense or budget data. A static client-only architecture directly satisfies these constraints with minimal operational overhead.

**Consequences**

- Financial data remains in the browser profile instead of server systems.
- Hosting and rollback are simpler than backend deployments.
- Cross-device sync, server backup, account-based recovery, and centralized audit logs are not available in the first release.

**Alternatives Considered**

- Backend API with database — Not selected because PRD-NFR-001 prohibits server-side financial data storage in the first release.
- Next.js/server-rendered app — Not selected because the product does not need backend rendering or API functions.
- Plain CDN React without build step — Not selected because TypeScript and CI build checks improve maintainability and validation.

### ARD-DEC-002 — Versioned `localStorage` Persistence

- **Status:** Accepted
- **Related Requirements:** PRD-FR-009, PRD-NFR-001

**Decision**

Expense and monthly budget data will be persisted in browser `localStorage` as versioned JSON payloads.

**Rationale**

The first release is small, single-user, same-browser-profile, and local-only. `localStorage` is sufficient for simple structured records and has low implementation complexity. Schema versioning allows future migration handling without introducing a database.

**Consequences**

- Data survives refresh and reopening in the same browser profile when site data is not cleared.
- Data may be lost if the user clears browser site data or uses another browser/profile/device.
- Large-scale record volumes are not the primary optimization target.

**Alternatives Considered**

- IndexedDB — Not selected for first release because the data model is small and `localStorage` is simpler.
- In-memory only — Not selected because PRD-FR-009 requires persistence after refresh/reopen.
- Server database — Not selected because PRD-NFR-001 excludes server-side storage.

### ARD-DEC-003 — Layered Frontend Modules Without Global State Framework

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-003, PRD-FR-006, PRD-FR-008

**Decision**

Application state and business rules will be structured with React components, custom hooks, domain services, storage services, and export services rather than a global state framework.

**Rationale**

The first release has modest state needs. Custom hooks and small services keep code testable and separated without adding unnecessary dependencies or architectural weight.

**Consequences**

- Validation, summary, filter, budget, storage, and export logic can be unit tested independently.
- Component complexity is reduced by moving rules into domain services.
- If future features add complex cross-screen state, a state framework can be reconsidered.

**Alternatives Considered**

- Single global component — Not selected because it would mix UI, persistence, calculations, and export logic.
- Redux Toolkit — Not selected because it is more infrastructure than needed for first-release scope.
- Backend-style API client abstractions — Not selected because there is no backend API.

### ARD-DEC-004 — Client-Side CSV Export

- **Status:** Accepted
- **Related Requirements:** PRD-FR-008, PRD-FR-005, PRD-NFR-001

**Decision**

CSV export will be generated entirely in the browser from the currently filtered expense records and downloaded through browser file download capabilities.

**Rationale**

The PRD requires CSV export while preserving local-only privacy. Browser-side generation avoids sending financial records to a server.

**Consequences**

- Export works without backend services.
- CSV files reflect the current filter state.
- Import, cloud backup, and server-side report generation remain out of scope.

**Alternatives Considered**

- Server-generated CSV — Not selected because it would require transmitting financial data to a backend.
- Export all records regardless of filters — Not selected because PRD-FR-008 requires exports to respect selected filters.

### ARD-DEC-005 — Automated CI Quality and Security Gates

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-002, PRD-NFR-003, PRD-NFR-004

**Decision**

The release pipeline must pass type checking, linting, unit tests, integration tests, accessibility checks, production build, dependency/CVE scanning, and SBOM generation before production deployment.

**Rationale**

The PRD defines acceptance-test-based release readiness and required quality, security, and compliance evidence. Automated gates provide repeatable assurance for a small team.

**Consequences**

- Build failures block release earlier.
- High and critical dependency vulnerabilities block release unless explicitly risk-accepted by the owner.
- CI setup adds modest upfront effort but reduces regression risk.

**Alternatives Considered**

- Manual acceptance testing only — Not selected because it provides weak repeatability and security evidence.
- Type check and build only — Not selected because it omits accessibility, unit/integration behavior, and dependency risk.

### ARD-DEC-006 — No External Telemetry by Default

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-001, PRD-NFR-004

**Decision**

The first release will use console-safe client-side error logging and user-visible feedback without sending telemetry, analytics, expense data, or budget data to external services by default.

**Rationale**

The strongest product constraint is local-only privacy. Avoiding external telemetry reduces the risk of accidentally transmitting personal financial information.

**Consequences**

- The team has limited production diagnostics unless users report issues.
- No external telemetry privacy review is needed for first release.
- Future telemetry would require explicit architecture and product approval.

**Alternatives Considered**

- Third-party analytics and error tracking — Not selected because it could weaken the privacy posture and is not required by the PRD.
- Local diagnostic logs in storage — Not selected because it adds storage complexity and could retain sensitive context unnecessarily.

## 8. API and Integration Architecture

### ARD-API-001 — Browser Local Storage Interface

- **Type:** File
- **Operation/Topic:** `localStorage.getItem`, `localStorage.setItem`, `localStorage.removeItem` for versioned JSON keys
- **Provider:** COMP-007
- **Consumers:** COMP-005
- **Authentication:** Browser profile access; no application authentication
- **Versioning:** JSON payload contains schema version; incompatible future changes require migration handling before write

**Request/Input Contract**

Storage write payload contains a top-level schema version and collections for expenses and budgets. Expense records include an application-generated `id`, `amount`, `date`, `category`, optional `description`, `createdAt`, and `updatedAt`. Budget records include `month`, `amount`, `createdAt`, and `updatedAt`.

**Response/Output Contract**

Storage read returns a parsed application data snapshot when present and valid. Missing data returns an empty initialized snapshot. Invalid or incompatible data returns a handled storage error state without silently overwriting existing stored data.

**Error Handling**

Storage parsing, quota, and unavailable-storage failures must be caught and surfaced through user-visible feedback. Writes must be atomic at the payload level from the application's perspective: the app prepares the full next versioned snapshot before calling `setItem`. Operations are idempotent when writing the same snapshot repeatedly.

### ARD-API-002 — CSV Export File Interface

- **Type:** File
- **Operation/Topic:** Browser-generated CSV file download from filtered expense records
- **Provider:** COMP-008
- **Consumers:** PERSONA-001
- **Authentication:** None
- **Versioning:** CSV column contract versioned by documented column order for first release

**Request/Input Contract**

Input is the currently filtered list of expense records. Each record must provide date, description, category, and amount. Dates must be formatted as `YYYY-MM-DD`. Amounts must be positive values with up to two decimal places.

**Response/Output Contract**

Output is a CSV file containing headers `date`, `description`, `category`, and `amount`, followed by one row per filtered expense. If no records match the current filters, no file is generated and the UI explains that there is nothing to export.

**Error Handling**

CSV generation must escape fields according to CSV rules. Export must provide success feedback after the browser download is triggered. Export failure must show actionable feedback and must not modify expense or budget data.

### ARD-API-003 — Static Asset Hosting Interface

- **Type:** File
- **Operation/Topic:** HTTPS delivery of compiled `index.html`, JavaScript, CSS, and static assets
- **Provider:** EXT-003
- **Consumers:** Browser Runtime
- **Authentication:** Public static access; no application login
- **Versioning:** Build artifact version identified by CI build ID or release tag

**Request/Input Contract**

Browser requests static application files from the configured hosting URL over HTTPS.

**Response/Output Contract**

Hosting returns cacheable static assets and the application shell needed to run the React app locally in the browser.

**Error Handling**

Hosting must support redeploying the previous known-good build artifact. Cache policy must not prevent urgent rollback from becoming available to users after normal browser refresh behavior.

## 9. Data Architecture

### DATA-001 — Expense

- **Purpose:** Record a personal spending event for monthly and category review.
- **System of Record:** Browser `localStorage` in the user's browser profile.
- **Classification:** Confidential
- **Retention:** Retain indefinitely in browser storage until the user deletes the individual expense or clears browser site data.
- **Key Relationships:** Belongs to one category; contributes to one calendar month summary; contributes to budget status for that month.

### DATA-002 — Monthly Budget

- **Purpose:** Store one budget amount for a selected calendar month.
- **System of Record:** Browser `localStorage` in the user's browser profile.
- **Classification:** Confidential
- **Retention:** Retain indefinitely in browser storage until the user updates it or clears browser site data.
- **Key Relationships:** One budget per month; compared with total expenses for the same month.

### DATA-003 — Category

- **Purpose:** Classify each expense into one of the fixed first-release categories.
- **System of Record:** Application code/configuration.
- **Classification:** Internal
- **Retention:** Fixed for the first release unless a later approved requirement changes category behavior.
- **Key Relationships:** Each expense has exactly one category; monthly summaries group expenses by category.

### DATA-004 — Filter State

- **Purpose:** Represent selected calendar month and selected category for UI display and export.
- **System of Record:** In-memory React application state.
- **Classification:** Internal
- **Retention:** Active browser session only unless implementation explicitly persists UI preferences in a future approved change.
- **Key Relationships:** Determines visible expense list, monthly summary context, and CSV export dataset.

### Data Migration

- **Required:** No
- **Approach:** No existing product data source exists for the first release. The storage payload must include a schema version so future releases can introduce client-side migrations if needed.

## 10. Security Architecture

### ARD-SEC-001 — Local-Only Privacy Boundary

- **Related Requirements:** PRD-NFR-001

The implementation must:

- Store expense and budget data locally in the user's browser profile.
- Avoid transmitting expense and budget records to application servers, telemetry services, financial services, or third-party analytics by product functionality.
- Serve the static app over HTTPS in production.
- Communicate that clearing browser site data may remove locally stored records.
- Treat locally stored expense and budget records as Confidential personal financial data.

The implementation must not:

- Require account creation, login, or authentication for first-release Must-priority flows.
- Request bank, UPI, card, payment-provider, or external financial account connections.
- Store expense or budget records in server-side storage.
- Enable cloud synchronization across devices in the first release.
- Log expense descriptions, amounts, or budgets to external systems by default.

### ARD-SEC-002 — Client-Side Input and Output Safety

- **Related Requirements:** PRD-FR-001, PRD-FR-003, PRD-FR-007, PRD-FR-008, PRD-NFR-003

The implementation must:

- Validate amount, date, category, description, and budget inputs before persistence.
- Escape CSV fields during export.
- Render user-provided descriptions as text, not executable markup.
- Handle malformed local storage data without executing or trusting it.

The implementation must not:

- Use unsanitized user-provided descriptions in HTML injection sinks.
- Persist invalid expense or budget records after failed validation.
- Silently overwrite unreadable local storage data without surfacing an error.

### Security Controls

- **Authentication:** None for the application; first release has no accounts or login.
- **Authorization:** Not applicable for multi-user access; all data is scoped to the current browser profile.
- **Secrets Management:** No runtime application secrets in the frontend bundle. CI/CD deployment credentials, if needed, must be stored in the CI/CD platform secret store and never committed.
- **Encryption:** HTTPS/TLS required for production static asset delivery. Browser `localStorage` does not provide application-managed encryption at rest; no app-layer encryption is required for first release (assumption).
- **Audit Logging:** No centralized audit logging because there is no backend and no account identity. User-visible feedback and local UI state changes evidence actions in-session.
- **Dependency/CVE Scanning:** Automated dependency/CVE scanning is required; high and critical vulnerabilities block release unless explicitly risk-accepted by the product/engineering owner.
- **SBOM:** Required for release build artifacts.

## 11. Reliability, Performance, and Scalability

### ARD-OPS-001 — Local Interaction Performance

- **Target:** Add, edit, delete, filter, summary recalculation, and budget updates complete within 200 ms for 1,000 local expense records on target browsers.
- **Validation Method:** Automated or scripted browser performance test using a generated 1,000-record local dataset and timing around each core interaction.

### ARD-OPS-002 — Persistence Reliability

- **Target:** Saved expenses and monthly budgets remain available after refresh and reopening in the same browser profile when browser site data has not been cleared.
- **Validation Method:** Integration or end-to-end persistence tests that write data, refresh/reopen the app context, and verify records and budgets are restored.

### ARD-OPS-003 — Static Asset Availability

- **Target:** Production deployment serves the latest approved static build over HTTPS after release and supports redeploying the previous known-good build when rollback is needed.
- **Validation Method:** Deployment smoke test after publish and rollback drill or documented rollback verification against prior build artifacts.

### ARD-OPS-004 — Responsive and Browser Compatibility

- **Target:** Core flows are usable on common desktop and mobile browser viewport sizes, with validation on current Chrome, Edge, Firefox, and Safari mobile/desktop where available.
- **Validation Method:** Automated and/or manual smoke tests covering add, edit, delete, filter, budget, export, and persistence across target browsers and viewport sizes.

### ARD-OPS-005 — Data Loss Limitation

- **Target:** The application must not delete existing persisted expense or budget records except through confirmed individual expense deletion, budget update, or browser-controlled site data clearing.
- **Validation Method:** Unit and integration tests for delete/update flows and malformed-storage handling; manual verification of browser site-data warning copy.

## 12. Observability

- **Logging:** Use console-safe client-side error logging for development and diagnostics. Do not log expense amounts, descriptions, budgets, or complete local storage payloads by default.
- **Metrics:** No external telemetry metrics by default. Build/test/security metrics are collected in CI/CD only.
- **Tracing:** Not applicable for first release because there is no backend or distributed request path.
- **Alerting:** CI/CD failures for build, tests, accessibility checks, dependency/CVE scanning, and deployment must notify the engineering owner through the CI/CD platform mechanism.
- **Sensitive Data Restrictions:** Expense records, budget amounts, descriptions, and exported CSV content must not be sent to external telemetry, analytics, logs, or monitoring services by default.

## 13. Deployment Architecture

- **Environments:** Local development, preview/test static deployment, and production static deployment.
- **Deployment Model:** Static build artifacts generated by CI/CD and published to a static hosting platform over HTTPS.
- **CI/CD:** Pipeline runs type check, lint, unit tests, integration tests, accessibility checks, production build, dependency/CVE scanning, SBOM generation, preview deployment, and production deployment after approval.
- **Configuration Management:** Use environment-specific build and hosting configuration checked into the repository where safe; deployment credentials must stay in CI/CD secrets.
- **Rollback:** Static hosting keeps prior build artifacts and can redeploy the previous known-good build.
- **Infrastructure as Code:** Lightweight IaC or checked-in hosting configuration for repeatable static deployment; exact tool/provider remains TBD.

## 14. Testing Architecture

Required test levels:

- Unit tests
- Integration tests
- Contract tests
- End-to-end tests
- Security tests
- Performance tests, where applicable

Minimum quality expectations:

- **Coverage Threshold:** At least 80% line coverage for domain, validation, storage, summary, and export modules.
- **Blocking Security Severity:** High and critical dependency/CVE findings block release unless explicitly risk-accepted by the product/engineering owner.
- **Required Build/Lint Checks:** TypeScript type check, lint, unit tests, integration tests, accessibility checks, production build, dependency/CVE scan, and SBOM generation.

Contract test expectations:

- ARD-API-001 storage payload contract covers schema version, expense fields, budget fields, empty initialization, malformed payload handling, and idempotent writes.
- ARD-API-002 CSV contract covers header order, `YYYY-MM-DD` date format, field escaping, filtered record inclusion, and no-file behavior when no records match.
- ARD-API-003 static asset contract covers production asset availability over HTTPS and smoke loading of the application shell.

End-to-end test expectations:

- Cover PRD Must-priority flows for add, view, edit, delete with confirmation/cancel, month/category filtering, monthly summary, monthly budget status, CSV export, and local persistence after refresh.
- Include keyboard operation and visible-label checks for core controls from PRD-NFR-003.
- Include responsive smoke checks for common desktop and mobile viewport sizes from PRD-NFR-002.

## 15. Architecture Constraints

- ARC-CON-001 — The application must remain a browser-based React application for the first release.
- ARC-CON-002 — The first release must not include a backend runtime for product functionality.
- ARC-CON-003 — Expense and budget data must remain local to the user's browser profile.
- ARC-CON-004 — The first release must not include account creation, login, cloud sync, or external financial integrations.
- ARC-CON-005 — Static production hosting must serve the app over HTTPS.
- ARC-CON-006 — Persistence must use browser `localStorage` with JSON serialization and schema versioning.
- ARC-CON-007 — The implementation must use the fixed first-release category list: Food, Transport, Shopping, Bills, Health, Entertainment, and Other.
- ARC-CON-008 — Currency display must be INR (₹).
- ARC-CON-009 — CSV export dates must use `YYYY-MM-DD` format.
- ARC-CON-010 — The first release must not include CSV import.
- ARC-CON-011 — The first release must not include custom categories.
- ARC-CON-012 — The first release must not include a clear-all local data feature.
- ARC-CON-013 — High and critical dependency/CVE findings must block release unless explicitly risk-accepted by the product/engineering owner.
- ARC-CON-014 — SBOM generation is required for release builds.
- ARC-CON-015 — No specific regulatory compliance framework was provided for this small-business project; any future compliance requirement must be added as an architecture constraint before implementation.

## 16. Risks and Trade-offs

| ID | Risk/Trade-off | Impact | Mitigation |
|---|---|---|---|
| ARC-RISK-001 | Users can lose data by clearing browser site data. | High impact to user records. | Communicate the limitation in-app as required by PRD-FR-009 and avoid promising backup/sync. |
| ARC-RISK-002 | Local-only storage prevents cross-device access. | Users cannot access records across phone/laptop profiles. | Keep cloud sync out of scope and set expectations in product copy. |
| ARC-RISK-003 | `localStorage` is simpler but less robust than IndexedDB for large datasets. | Very large expense histories may hit browser limits or performance issues. | Validate 1,000-record performance target and use schema versioning so future migration to IndexedDB remains possible. |
| ARC-RISK-004 | No external telemetry limits production diagnostics. | Engineering may rely on user reports for runtime failures. | Use CI gates, local error handling, user-visible failures, and avoid telemetry to preserve privacy. |
| ARC-RISK-005 | Exact browser version matrix is unsettled. | Compatibility sign-off may be delayed. | Track as ARC-OQ-001 and run smoke tests on current Chrome, Edge, Firefox, and Safari mobile/desktop where available. |
| ARC-RISK-006 | Static hosting provider is not selected. | Deployment configuration and IaC details remain incomplete. | Track as ARC-OQ-002 and keep hosting requirements provider-neutral. |
| ARC-RISK-007 | No app-layer encryption for `localStorage`. | Local device/browser compromise could expose records. | Classify data as Confidential, avoid network transmission, require HTTPS for assets, and document app-layer encryption as not required for first release (assumption). |

## 17. Open Architecture Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| ARC-OQ-001 | What exact desktop and mobile browser versions are supported for final compatibility sign-off? | Product Owner | Open | TBD |
| ARC-OQ-002 | Which static hosting provider will be used for production and preview/test deployments? | Engineering Lead | Open | TBD |
| ARC-OQ-003 | Which lightweight IaC or checked-in hosting configuration standard will be used for repeatable static deployment? | Engineering Lead | Open | TBD |
| ARC-OQ-004 | Who is the named Architecture Owner for approval and future architecture decisions? | Product Owner | Open | TBD |
| ARC-OQ-005 | Which CI/CD platform will execute release gates, SBOM generation, and deployments? | Engineering Lead | Open | TBD |
| ARC-OQ-006 | Is app-layer encryption for local storage explicitly out of scope for the first release? Current architecture assumes yes because no key-management mechanism exists in a local-only no-login app. | Security Reviewer | Open | TBD |

## 18. Feature Mapping

| Feature | Relevant Components | Decision IDs | Interface IDs | Security IDs |
|---|---|---|---|---|
| FEAT-001 | COMP-002, COMP-005, COMP-006, COMP-007 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003 | ARD-API-001 | ARD-SEC-001, ARD-SEC-002 |
| FEAT-002 | COMP-003, COMP-005, COMP-006, COMP-007 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003 | ARD-API-001 | ARD-SEC-001, ARD-SEC-002 |
| FEAT-003 | COMP-004, COMP-005, COMP-006, COMP-007 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003 | ARD-API-001 | ARD-SEC-001 |
| FEAT-004 | COMP-004, COMP-005, COMP-006, COMP-007 | ARD-DEC-001, ARD-DEC-002, ARD-DEC-003 | ARD-API-001 | ARD-SEC-001, ARD-SEC-002 |
| FEAT-005 | COMP-003, COMP-005, COMP-006, COMP-008 | ARD-DEC-001, ARD-DEC-004, ARD-DEC-006 | ARD-API-002 | ARD-SEC-001, ARD-SEC-002 |
| FEAT-006 | COMP-005, COMP-007 | ARD-DEC-001, ARD-DEC-002 | ARD-API-001 | ARD-SEC-001 |

## 19. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Solution Architect | TBD | Pending | TBD |
| Security Reviewer | TBD | Pending | TBD |
| Engineering Lead | TBD | Pending | TBD |