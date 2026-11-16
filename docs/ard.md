# Architecture Requirements Document (ARD)

> Replace all `<PLACEHOLDER>` values. Every material architecture decision or constraint should have a stable ID that feature specifications can reference.

## Document Metadata

- **Product/Application:** Simple Todo Application
- **Document ID:** ARD-TODO
- **Version:** 0.1
- **Status:** Draft
- **Architecture Owner:** TBD; see ARC-OQ-001
- **Contributors:** TBD; see ARC-OQ-002
- **Last Updated:** 2026-11-12
- **Reference PRD:** `PRD.md`

---

## 1. Architecture Summary

The Simple Todo Application will be implemented as a browser-only, single-screen React single-page application using TypeScript, Vite, and plain scoped CSS or CSS modules. It will be statically hosted on GitHub Pages and delivered over HTTPS with no backend API, backend database, paid service dependency, user account, telemetry service, analytics service, or multi-device synchronization.

Task data will be stored only in browser `localStorage` under a versioned v1 key. The React application will keep current task state in built-in React state and custom hooks, while separating presentation components, task orchestration, pure domain utilities, runtime persistence validation, and the localStorage repository. The localStorage repository is the task system of record for one user on one browser/device.

The architecture prioritizes lightweight delivery, traceability to Must-priority PRD requirements, accessibility, responsive browser support, deterministic local persistence behavior, dependency security hygiene, SBOM generation, and automated quality gates with manual browser/device checks where automation is unavailable.

## 2. Architecture Drivers

- PRD-FR-001 — Users must be able to create tasks with required non-empty trimmed title, optional details, and default pending status.
- PRD-FR-002 — Task form validation must prevent invalid saves and provide inline field-level errors.
- PRD-FR-003 — Saved tasks must be visible in a task list with available task details.
- PRD-FR-004 — Default ordering must place pending tasks before completed tasks and sort by due date within each status group.
- PRD-FR-005 — Users must be able to edit existing task details.
- PRD-FR-006 — Deletion must require user confirmation.
- PRD-FR-007 — Users must be able to toggle task status between pending and completed.
- PRD-FR-008 — Users must be able to search tasks by title using case-insensitive partial matching.
- PRD-FR-009 — Users must be able to filter tasks by All, Pending, and Completed.
- PRD-FR-010 — Dashboard counts must show total, pending, completed, and overdue tasks.
- PRD-FR-011 — Task data must persist after browser refresh using browser localStorage.
- PRD-FR-012 — The app must clearly report unavailable, full, corrupted, or unreadable localStorage persistence failures.
- PRD-NFR-001 — Core task flows must be keyboard-operable with visible focus and labeled fields.
- PRD-NFR-002 — The UI must be responsive and usable on desktop and mobile browsers.
- PRD-NFR-003 — Task data must remain client-side and must not be transmitted to backend, analytics, telemetry, or third-party storage services.
- PRD-NFR-004 — Core interactions must complete within 200 ms for up to 500 stored tasks on supported browsers.
- PRD-NFR-005 — Core flows must pass on latest stable Chrome, Edge, Firefox, Safari, mobile Safari, and mobile Chrome.
- PRD-NFR-006 — CI/CD must block unresolved high and critical dependency/CVE findings and generate an SBOM.
- PRD-NFR-007 — User-entered task content must render as escaped text and must not use unsafe raw HTML injection.
- CON-001 — The product is for a small business and must remain lightweight for the initial demo.
- CON-002 — The initial version must use browser localStorage only for task persistence.
- CON-003 — The initial version must not require backend APIs, backend databases, paid services, or multi-device synchronization.
- CON-004 — The initial release has no fixed date and should ship when all Must-priority requirements pass acceptance testing and release gates.
- CON-005 — The initial product is limited to a single user on one browser/device.
- CON-006 — Task data must not be transmitted to backend APIs, analytics services, telemetry services, or third-party storage services.
- CON-007 — The implementation stack is a browser-only single-page application using React, TypeScript, and Vite.
- CON-008 — Static application assets must be served over HTTPS through GitHub Pages or equivalent static hosting.

## 3. Architecture Principles

1. Keep task data fully client-side for the initial version, with no backend or remote data transmission.
2. Make localStorage behavior explicit, versioned, validated, testable, and failure-aware.
3. Reuse established repository and platform patterns where appropriate.
4. Keep presentation, orchestration, domain, integration, and persistence concerns separated.
5. Prefer browser-native and React-native capabilities over extra libraries unless they materially reduce risk.
6. Treat accessibility, responsive layout, dependency hygiene, SBOM generation, and automated release evidence as release-gating architecture concerns.

## 4. System Context

### 4.1 Actors

- Individual Task User — Uses the browser application to create, view, edit, delete, search, filter, persist, and track personal tasks on one browser/device.
- Browser Runtime — Executes the React SPA, provides DOM APIs, supports keyboard/focus behavior, and exposes browser localStorage.
- GitHub Pages — Serves compiled static application assets over HTTPS.
- GitHub Actions — Runs install, typecheck, lint, tests, accessibility checks, dependency scanning, SBOM generation, build, deployment, and smoke checks.
- Browser Vendors — Provide supported desktop and mobile browser runtimes.
- npm Registry — Provides open-source package dependencies used during build and test.

### 4.2 External Systems

| ID | System | Purpose | Interface | Owner |
|---|---|---|---|---|
| EXT-001 | Browser localStorage / Web Storage API | Persist task data after browser refresh for one browser/device | Web API/File-like key-value storage | Browser vendor / user browser environment |
| EXT-002 | GitHub Pages | Serve compiled static SPA assets over HTTPS | File | Engineering / GitHub |
| EXT-003 | GitHub Actions | Run CI/CD quality gates and deploy static artifacts | Event | Engineering / GitHub |
| EXT-004 | Supported web browsers | Provide runtime environment for the SPA on desktop and mobile | Web runtime | Browser vendors |
| EXT-005 | npm Registry | Supply JavaScript dependencies for build, test, and tooling | Package registry | npm / package maintainers |

## 5. Technology Stack

| Layer | Technology | Version/Constraint | Rationale |
|---|---|---|---|
| Frontend | React with TypeScript and Vite | Current stable versions selected during implementation; must support latest stable Chrome, Edge, Firefox, Safari, mobile Safari, and mobile Chrome | Mandated by PRD; supports typed component-based SPA development and static builds |
| Backend | None | Backend APIs and backend databases are prohibited for task data flows | Satisfies client-side privacy boundary and lightweight demo constraints |
| Data | Browser localStorage / Web Storage API | Browser localStorage only; versioned v1 storage key; task data scoped to one browser/device | Satisfies refresh persistence without backend storage or paid services |
| Hosting | GitHub Pages | HTTPS static hosting compatible with Vite build artifacts | Low operational overhead and selected first-release static host |
| CI/CD | GitHub Actions | Runs release gates and deploys GitHub Pages artifact | Git-based automation aligned to hosting platform |
| State Management | React built-in state and custom hooks | No external state management library | Keeps the app lightweight while meeting single-screen state needs |
| Styling | Plain scoped CSS or CSS modules | No paid UI kit | Minimizes dependencies and supports responsive/accessibility requirements |
| Testing | Vitest, React Testing Library, Playwright, axe accessibility checks | Must support unit/component/E2E/accessibility gates | Provides fast domain/persistence tests and user-flow evidence |
| Security Tooling | `npm audit` and CycloneDX npm SBOM generation | High and critical CVEs block release; SBOM required | Meets PRD dependency security and SBOM requirements |

## 6. Logical Architecture

### 6.1 Components

#### COMP-001 — App Shell

- **Responsibility:** Compose dashboard, task form, search/filter controls, task list, delete dialog, application-level persistence errors, and single-screen SPA layout.
- **Inputs:** Browser load event, persisted tasks from COMP-008, user interactions from child components.
- **Outputs:** Rendered SPA, delegated task commands, visible error banners, updated UI state.
- **Dependencies:** COMP-002, COMP-003, COMP-004, COMP-005, COMP-006, COMP-007, COMP-008.
- **Repository Area:** `src/App.tsx`, `src/features/tasks/`

#### COMP-002 — Task Form Component

- **Responsibility:** Capture create/edit inputs for title, description, due date, and priority; show inline validation errors; clear create form after successful create.
- **Inputs:** User-entered task fields, existing task values for edit mode, validation results from COMP-007.
- **Outputs:** Create or update commands with normalized task input.
- **Dependencies:** COMP-007.
- **Repository Area:** `src/features/tasks/components/TaskForm.tsx`

#### COMP-003 — Task List Component

- **Responsibility:** Display ordered task rows, empty/no-results states, status toggle actions, edit actions, and delete initiation.
- **Inputs:** Visible task collection, task commands, current search/filter state.
- **Outputs:** Status toggle, edit, and delete-request commands.
- **Dependencies:** COMP-006, COMP-007.
- **Repository Area:** `src/features/tasks/components/TaskList.tsx`

#### COMP-004 — Accessible Delete Confirmation Dialog

- **Responsibility:** Confirm destructive delete operations with accessible in-app focus management and keyboard operation.
- **Inputs:** Selected task pending deletion and user confirmation/cancel actions.
- **Outputs:** Delete-confirmed or delete-cancelled command.
- **Dependencies:** COMP-001.
- **Repository Area:** `src/features/tasks/components/DeleteTaskDialog.tsx`

#### COMP-005 — Dashboard Component

- **Responsibility:** Display total, pending, completed, and overdue task counts.
- **Inputs:** Saved task collection and current local date.
- **Outputs:** Accessible dashboard count cards or summary.
- **Dependencies:** COMP-007.
- **Repository Area:** `src/features/tasks/components/Dashboard.tsx`

#### COMP-006 — Search and Filter Component

- **Responsibility:** Capture title search term and selected status filter for All, Pending, and Completed.
- **Inputs:** User search text and status filter selection.
- **Outputs:** Search/filter state updates.
- **Dependencies:** COMP-007.
- **Repository Area:** `src/features/tasks/components/SearchFilter.tsx`

#### COMP-007 — Task Domain Utilities

- **Responsibility:** Provide pure functions for task validation, normalization, ordering, title search, status filtering, overdue detection, dashboard count calculation, ID generation strategy wrapping, timestamp normalization, and v1 payload validation helpers.
- **Inputs:** Task input, task collection, search term, status filter, current local date, raw parsed payloads.
- **Outputs:** Validation results, normalized task values, sorted/filtered task collections, dashboard counts, overdue flags, validated v1 collections.
- **Dependencies:** Browser `crypto.randomUUID` where available for ID generation through an adapter.
- **Repository Area:** `src/features/tasks/domain/`

#### COMP-008 — localStorage Task Repository

- **Responsibility:** Read, validate, write, and remove serialized task collections in browser localStorage; detect unavailable, full, corrupted, or unreadable storage errors.
- **Inputs:** Task collection from orchestration, browser localStorage state, v1 storage key.
- **Outputs:** Persisted task collection, loaded task collection, structured persistence/load errors.
- **Dependencies:** EXT-001, COMP-007.
- **Repository Area:** `src/features/tasks/persistence/localStorageTaskRepository.ts`

#### COMP-009 — Task State Hook / Orchestrator

- **Responsibility:** Coordinate React state, task commands, domain utilities, persistence writes, persistence errors, and derived visible views without external state libraries.
- **Inputs:** User commands from components, loaded task data, domain utility results.
- **Outputs:** Updated in-memory task state, persisted task collection, derived dashboard/search/filter views.
- **Dependencies:** COMP-007, COMP-008.
- **Repository Area:** `src/features/tasks/hooks/useTasks.ts`

#### COMP-010 — Static Build Artifact

- **Responsibility:** Package the React/Vite application into static assets for GitHub Pages.
- **Inputs:** Source code, Vite configuration, GitHub Pages base path configuration.
- **Outputs:** Static HTML, JavaScript, CSS, and asset files.
- **Dependencies:** EXT-002, EXT-003, EXT-005.
- **Repository Area:** `dist/`

### 6.2 Data Flow

1. The Individual Task User loads the app from EXT-002 in a supported browser.
2. COMP-001 initializes COMP-009, and COMP-009 requests saved tasks from COMP-008.
3. COMP-008 reads the JSON payload from EXT-001 under the v1 storage key and asks COMP-007 to validate the parsed v1 task collection.
4. COMP-008 returns a valid task collection, an empty collection when no payload exists, or a structured load error when data is unavailable, corrupted, or unreadable.
5. COMP-001 displays a persistence error banner when load/write errors occur and passes task state to COMP-002 through COMP-006.
6. The user submits create/edit input through COMP-002.
7. COMP-009 uses COMP-007 to validate and normalize task input, generate or preserve identifiers, update ISO timestamps, and calculate derived state.
8. After every successful create, edit, delete, or status toggle, COMP-009 asks COMP-008 to write the full task collection immediately to localStorage.
9. COMP-008 writes the serialized v1 task collection to EXT-001 or returns a structured persistence error; the UI must not claim successful persistence when writes fail.
10. COMP-007 derives ordered, searched, filtered, overdue, and counted views from saved in-memory task state.
11. COMP-004 confirms deletion before COMP-009 removes the task and triggers persistence.
12. On browser refresh, the flow restarts from localStorage and dashboard counts are recalculated from persisted task data.

## 7. Architecture Decisions

### ARD-DEC-001 — Use React, TypeScript, and Vite for the Static SPA

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-003, PRD-FR-005, PRD-FR-007, PRD-FR-008, PRD-FR-009, PRD-FR-010, PRD-NFR-001, PRD-NFR-002, PRD-NFR-005, PRD-NFR-007

**Decision**

Implement the Todo application as a React single-page application using TypeScript and Vite, compiled to static assets.

**Rationale**

The PRD mandates React, TypeScript, and Vite. This stack supports accessible component composition, typed task models, fast builds, and static hosting on GitHub Pages.

**Consequences**

- Enables typed client-side development and automated unit/component testing.
- Produces static assets that do not require backend compute.
- Requires Node-based build tooling in GitHub Actions.

**Alternatives Considered**

- Vanilla HTML, CSS, and JavaScript — Rejected because the approved PRD mandates React, TypeScript, and Vite.
- Vue or Svelte with TypeScript and Vite — Rejected because the approved PRD mandates React.

### ARD-DEC-002 — Use Browser localStorage as the Task System of Record

- **Status:** Accepted
- **Related Requirements:** PRD-FR-011, PRD-FR-012, PRD-NFR-003

**Decision**

Persist the complete v1 task collection in browser localStorage and treat localStorage as the system of record for the initial version.

**Rationale**

The PRD requires persistence after browser refresh while prohibiting backend APIs, backend databases, paid services, and multi-device synchronization. localStorage satisfies the single-browser/device scope with minimal operational complexity.

**Consequences**

- Tasks persist after refresh in the same browser profile.
- Tasks are not available across browsers, devices, browser profiles, or after cleared browser/application data.
- Persistence and corruption failure handling must be explicit and user-visible.

**Alternatives Considered**

- Backend database with API — Rejected because backend APIs and backend databases are out of scope.
- IndexedDB — Rejected because PRD constraints specify browser localStorage only.
- Session storage — Rejected because it does not satisfy the persistence requirement as directly as localStorage.

### ARD-DEC-003 — Separate UI Components, State Orchestration, Domain Utilities, and Persistence

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-004, PRD-FR-008, PRD-FR-009, PRD-FR-010, PRD-FR-012, PRD-NFR-004

**Decision**

Organize the task feature into React presentation components, a task state/orchestration hook, pure task domain utilities, and a localStorage repository.

**Rationale**

Validation, ordering, search, filtering, counts, performance, and persistence failure behavior must be testable and traceable to acceptance criteria. Separation keeps business rules outside presentation components and enables focused tests.

**Consequences**

- Domain and persistence behavior can be covered with fast Vitest tests.
- UI components remain focused on rendering and user interaction.
- Slightly more structure than a single-file demo, but lower regression risk.

**Alternatives Considered**

- Single React component containing all logic — Rejected because it weakens maintainability and test isolation.
- Deep enterprise layering — Rejected because it adds unnecessary ceremony for a lightweight demo.

### ARD-DEC-004 — Use React Built-In State and Custom Hooks Only

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-005, PRD-FR-007, PRD-FR-008, PRD-FR-009, PRD-FR-010, PRD-NFR-004

**Decision**

Manage application state with React built-in state and custom hooks, without Redux, Zustand, or another external state management library.

**Rationale**

The application is single-user, single-screen, and lightweight. React state and custom hooks are sufficient for task state, derived views, persistence errors, and form/edit state while reducing dependencies and security surface area.

**Consequences**

- Reduces bundle complexity and dependency scanning scope.
- Keeps state management easy to reason about for the first release.
- Future multi-screen or multi-user needs may require reevaluation.

**Alternatives Considered**

- Redux Toolkit — Rejected as unnecessary for the initial single-screen app.
- Zustand — Viable, but rejected to avoid adding dependency surface for limited state needs.

### ARD-DEC-005 — Use Hand-Written v1 Runtime Validation for Persisted Payloads

- **Status:** Accepted
- **Related Requirements:** PRD-FR-002, PRD-FR-012, PRD-NFR-006

**Decision**

Validate parsed localStorage payloads with a small hand-written runtime validator for the v1 task collection schema.

**Rationale**

Persisted localStorage data can be corrupted, unreadable, or manually modified. A hand-written validator is sufficient for the small schema and avoids adding a validation library dependency.

**Consequences**

- Corrupted data can be detected deterministically.
- Dependency footprint remains small.
- Schema changes require updating validator tests alongside code.

**Alternatives Considered**

- Zod — Rejected because the schema is small and dependency minimization is preferred.
- TypeScript-only typing after `JSON.parse` — Rejected because it does not validate runtime payloads.

### ARD-DEC-006 — Deploy Through GitHub Actions to GitHub Pages

- **Status:** Accepted
- **Related Requirements:** PRD-NFR-005, PRD-NFR-006

**Decision**

Use GitHub Actions to install dependencies, typecheck, lint, test, scan, generate SBOM, build, deploy to GitHub Pages, and run a deployment smoke check.

**Rationale**

GitHub Pages is the selected static host and GitHub Actions is the selected CI/CD platform. The pairing provides lightweight static deployment and repeatable release evidence.

**Consequences**

- Deployments are reproducible from source control.
- Rollback can be performed by redeploying a prior commit/build.
- CI configuration must include GitHub Pages base path handling for Vite.

**Alternatives Considered**

- Manual deployment — Rejected because it weakens repeatability and release evidence.
- Netlify or Cloudflare Pages — Viable static hosts, but GitHub Pages was selected for the first release.

### ARD-DEC-007 — Gate Release With Vitest, React Testing Library, Playwright, axe, npm audit, and CycloneDX

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-002, PRD-FR-003, PRD-FR-004, PRD-FR-005, PRD-FR-006, PRD-FR-007, PRD-FR-008, PRD-FR-009, PRD-FR-010, PRD-FR-011, PRD-FR-012, PRD-NFR-001, PRD-NFR-002, PRD-NFR-004, PRD-NFR-005, PRD-NFR-006, PRD-NFR-007

**Decision**

Require Vitest and React Testing Library for unit/component tests, Playwright for end-to-end acceptance flows and supported browser automation, axe accessibility checks for core screens, `npm audit` for dependency scanning, and CycloneDX npm SBOM generation.

**Rationale**

The approved PRD requires automated quality gates, E2E acceptance evidence, accessibility checks, 80% domain/persistence utility coverage, zero unresolved high/critical dependency findings, and SBOM generation.

**Consequences**

- Critical flows are validated consistently in CI/CD.
- Security and SBOM evidence is available before release.
- Manual browser/device checks remain required where automation is unavailable.

**Alternatives Considered**

- Manual testing only — Rejected because the PRD requires automated release evidence.
- Playwright only — Rejected because unit/domain/persistence tests provide faster targeted feedback.
- Snyk — Viable, but `npm audit` and CycloneDX were selected for a no-paid-service first release.

### ARD-DEC-008 — Use a Single-Screen SPA With No Client Router

- **Status:** Accepted
- **Related Requirements:** PRD-FR-001, PRD-FR-003, PRD-FR-008, PRD-FR-009, PRD-FR-010, PRD-NFR-002

**Decision**

Implement the app as a single-screen SPA with no client-side router and configure the Vite base path for GitHub Pages.

**Rationale**

The PRD does not require multi-page navigation. Avoiding a router reduces dependency surface, simplifies GitHub Pages deployment, and keeps keyboard/accessibility testing focused.

**Consequences**

- Static hosting is simpler because no route fallback is required.
- Deep links to task detail pages are not supported in the initial release.
- Future multi-page features may introduce routing later.

**Alternatives Considered**

- React Router hash routing — Rejected because no separate routes are needed.
- React Router browser history routing — Rejected because it adds deployment fallback complexity for GitHub Pages.

### ARD-DEC-009 — Use Accessible In-App Delete Confirmation Dialog

- **Status:** Accepted
- **Related Requirements:** PRD-FR-006, PRD-NFR-001

**Decision**

Implement deletion confirmation as an accessible in-app dialog with keyboard support and focus management.

**Rationale**

The PRD requires deletion confirmation and keyboard-operable flows. An in-app dialog provides consistent styling, testability, and accessible focus control.

**Consequences**

- Requires focus trap/return behavior and accessible labels/descriptions.
- Avoids inconsistent native browser confirm behavior across browsers.
- Adds a small UI component but improves test coverage and user experience.

**Alternatives Considered**

- Native `window.confirm` — Rejected because it is less controllable and harder to test consistently.
- Inline confirmation controls — Viable, but a dialog better supports clear destructive action confirmation.

## 8. API and Integration Architecture

### ARD-API-001 — Browser localStorage Task Persistence Interface

- **Type:** File
- **Operation/Topic:** `localStorage.getItem('todo.tasks.v1')`, `localStorage.setItem('todo.tasks.v1', serializedTasks)`, `localStorage.removeItem('todo.tasks.v1')`
- **Provider:** COMP-008
- **Consumers:** COMP-009, COMP-001
- **Authentication:** None; scoped to the user’s browser profile and application origin
- **Versioning:** Storage key includes version suffix `v1`; future incompatible schema changes must use a new key or explicit migration routine

**Request/Input Contract**

The persisted payload is a JSON-serialized v1 task collection stored under `todo.tasks.v1`.

Task fields:

- `id`: stable client-generated string identifier created with `crypto.randomUUID` when available, with timestamp/random fallback.
- `title`: non-empty trimmed string.
- `description`: optional string.
- `dueDate`: optional ISO calendar date string.
- `priority`: optional enum value `Low`, `Medium`, or `High`.
- `status`: enum value `pending` or `completed`.
- `createdAt`: ISO timestamp string generated when the task is created.
- `updatedAt`: ISO timestamp string updated whenever the task changes.

**Response/Output Contract**

Read operations return one of the following:

- Validated task collection when valid persisted data exists.
- Empty task collection when no saved payload exists.
- Structured load error when localStorage is unavailable, unreadable, corrupted, or contains invalid data.

Write operations return one of the following:

- Success result after the serialized task collection is saved.
- Structured persistence error when localStorage is unavailable, quota is exceeded, or the write cannot be verified.

**Error Handling**

The repository must catch localStorage read/write exceptions and return structured errors to COMP-009 and COMP-001. On corrupted or unreadable data at load, the app must show an error banner, load an empty in-memory task list, and not automatically overwrite the existing stored data until the user performs a valid save. Writes replace the full task collection immediately after every successful create, edit, delete, or status toggle. The UI must not indicate that unsaved changes were persisted successfully. Retrying is user-driven through the next save action. Idempotency is achieved by writing the complete collection for the current application state.

### ARD-API-002 — GitHub Pages Static Asset Hosting Interface

- **Type:** File
- **Operation/Topic:** `GET /`, `GET /assets/*`
- **Provider:** EXT-002
- **Consumers:** EXT-004
- **Authentication:** None in application scope
- **Versioning:** Static build artifacts are versioned by source commit and GitHub Actions deployment history

**Request/Input Contract**

Supported browsers request static HTML, JavaScript, CSS, and asset files generated by the Vite build with the configured GitHub Pages base path.

**Response/Output Contract**

GitHub Pages returns static files over HTTPS with content types appropriate to each asset.

**Error Handling**

Hosting errors are handled by the browser as page or asset load failures. Rollback is performed by redeploying a previous known-good commit/build through GitHub Actions.

### ARD-API-003 — GitHub Actions CI/CD Workflow Interface

- **Type:** Event
- **Operation/Topic:** `push` or manual workflow dispatch for install, typecheck, lint, tests, scans, SBOM, build, deploy, and smoke check
- **Provider:** EXT-003
- **Consumers:** Engineering, EXT-002
- **Authentication:** GitHub repository permissions and GitHub Pages deployment permissions
- **Versioning:** Workflow definitions are versioned in source control

**Request/Input Contract**

A Git event or approved manual dispatch triggers the workflow against the repository source, lockfile, test suite, and deployment configuration.

**Response/Output Contract**

The workflow produces pass/fail status, test results, coverage report, `npm audit` result, CycloneDX SBOM artifact, Vite build artifact, GitHub Pages deployment, and deployment smoke result.

**Error Handling**

Any failed required step must fail the workflow and block release. High or critical dependency findings block release unless remediated or separately risk-accepted outside this ARD.

## 9. Data Architecture

### DATA-001 — Task

- **Purpose:** Represent a user-managed Todo item with details, priority, due date, and completion status.
- **System of Record:** Browser localStorage key `todo.tasks.v1`
- **Classification:** Internal
- **Retention:** Retained indefinitely in localStorage until the user deletes the task or clears browser/application data.
- **Key Relationships:** Belongs to the single browser/device application scope; contributes to dashboard counts, search results, filters, ordering, and overdue calculations.

### DATA-002 — Task Collection

- **Purpose:** Store the complete set of persisted tasks for the single browser/device scope.
- **System of Record:** Browser localStorage key `todo.tasks.v1`
- **Classification:** Internal
- **Retention:** Retained indefinitely in localStorage until tasks are deleted, a valid future clear action is added, or browser/application data is cleared.
- **Key Relationships:** Contains zero or more DATA-001 records; loaded into application state at startup and written after create, edit, delete, or status toggle operations.

### DATA-003 — UI State

- **Purpose:** Represent transient form, edit mode, search term, status filter, delete dialog, focus, and visible error state.
- **System of Record:** React in-memory state in COMP-009 and child components
- **Classification:** Internal
- **Retention:** Current browser session memory only; not persisted except task data saved in DATA-002.
- **Key Relationships:** Derived from and commands updates to DATA-002; not a durable record.

### Data Migration

- **Required:** No
- **Approach:** No migration is required for the initial release because no prior application or backend data source exists. The `todo.tasks.v1` storage key reserves a path for future schema migration if needed.

## 10. Security Architecture

### ARD-SEC-001 — Client-Side Privacy Boundary and No-Backend Data Transmission

- **Related Requirements:** PRD-NFR-003

The implementation must:

- Store task data only in browser localStorage for the initial version.
- Avoid transmitting task titles, descriptions, due dates, priorities, statuses, or task identifiers to backend APIs, analytics services, telemetry services, or third-party storage services.
- Serve static assets over HTTPS through GitHub Pages.
- Keep task data scoped to the application origin and browser profile.

The implementation must not:

- Add user accounts, authentication flows, backend APIs, backend databases, or multi-device synchronization for task data.
- Log task titles, descriptions, due dates, priorities, statuses, or task identifiers to remote services.
- Include application secrets in client-side source code, build artifacts, or static hosting configuration.

### ARD-SEC-002 — Client-Side Input Safety and Rendering Safety

- **Related Requirements:** PRD-FR-002, PRD-NFR-007

The implementation must:

- Validate task input before create or edit saves.
- Render user-entered task content through React text rendering or equivalent escaping behavior.
- Validate localStorage payloads before treating them as task data.

The implementation must not:

- Use `dangerouslySetInnerHTML` or equivalent unsafe HTML injection for task content.
- Execute user-entered task content as HTML, JavaScript, or scriptable markup.
- Store secrets, credentials, or tokens in localStorage.

### ARD-SEC-003 — Dependency Security and Supply Chain Evidence

- **Related Requirements:** PRD-NFR-006

The implementation must:

- Run `npm audit` in GitHub Actions and block unresolved high and critical severity findings.
- Generate a CycloneDX SBOM during CI/CD.
- Use lockfile-based dependency installation in CI/CD.

The implementation must not:

- Suppress high or critical dependency findings without documented remediation or risk acceptance outside this ARD.
- Depend on paid third-party UI, storage, analytics, telemetry, or infrastructure services for the first release.

### Security Controls

- **Authentication:** None in application scope; user accounts and authentication are out of scope for the initial version.
- **Authorization:** None in application scope; all task data is local to the user’s browser/device and application origin.
- **Secrets Management:** No application secrets are required or permitted in client-side code, GitHub Pages static artifacts, or runtime application configuration.
- **Encryption:** HTTPS is required in transit for static asset delivery; application-level at-rest encryption is not provided for localStorage and remains dependent on user device/browser environment.
- **Audit Logging:** No application audit logs are required because user accounts, backend APIs, manager reporting, and remote telemetry are out of scope.
- **Dependency/CVE Scanning:** GitHub Actions must run `npm audit`; unresolved high and critical severity findings block release.
- **SBOM:** Required; GitHub Actions must generate a CycloneDX SBOM artifact for the application dependencies/build.

## 11. Reliability, Performance, and Scalability

### ARD-OPS-001 — Client-Side Interaction Performance

- **Target:** Create, edit, delete, status toggle, search, filter, sort, and dashboard count updates complete within 200 ms for up to 500 stored tasks on supported browsers.
- **Validation Method:** Automated performance-oriented Vitest tests for domain utilities and representative Playwright E2E timing checks using a seeded 500-task dataset on supported automation targets.

### ARD-OPS-002 — Supported Browser Runtime Reliability

- **Target:** Core acceptance flows pass on latest stable Chrome, Edge, Firefox, Safari, mobile Safari, and mobile Chrome.
- **Validation Method:** Playwright E2E tests where supported by CI automation plus manual acceptance testing for supported browser/device combinations not covered by automation.

### ARD-OPS-003 — Deterministic Persistence Failure Handling

- **Target:** 100% of attempted persistent save operations show a clear error and avoid claiming successful persistence when localStorage is unavailable or full; 100% of corrupted/unreadable load cases show a clear error and do not automatically overwrite existing storage.
- **Validation Method:** Vitest tests mocking localStorage read/write exceptions and invalid payloads, plus Playwright E2E tests simulating storage failure where feasible.

### ARD-OPS-004 — Functional Acceptance Reliability

- **Target:** 100% of Must-priority functional acceptance flows pass before release.
- **Validation Method:** Playwright E2E tests covering create, view, edit, delete confirmation, search, filter, status toggle, dashboard count, refresh persistence, and failure handling, plus manual checks where automation is unavailable.

### ARD-OPS-005 — Static Hosting Availability Boundary

- **Target:** Application availability depends only on GitHub Pages serving compiled assets over HTTPS; no backend runtime dependencies are introduced.
- **Validation Method:** GitHub Actions deployment smoke test loads the deployed GitHub Pages URL and verifies the app shell renders after deployment.

## 12. Observability

- **Logging:** Browser console logging is limited to non-sensitive diagnostics for unexpected local runtime errors; task titles, descriptions, due dates, priorities, statuses, and task identifiers must not be intentionally logged to remote services.
- **Metrics:** No remote runtime product telemetry is required or allowed for task content; CI test results, coverage reports, scan results, SBOM generation, and deployment status provide operational evidence.
- **Tracing:** Distributed tracing is not applicable because the initial version has no backend services or cross-service calls.
- **Alerting:** GitHub Actions failures for install, typecheck, lint, tests, accessibility checks, build, dependency scan, SBOM generation, deployment, or smoke checks must fail the pipeline and be visible to engineering through GitHub workflow status.
- **Sensitive Data Restrictions:** DATA-001, DATA-002, and DATA-003 are Internal; task content must remain in browser localStorage/in-memory state and must not be sent in analytics, telemetry, remote logs, or backend payloads.

## 13. Deployment Architecture

- **Environments:** Local development, GitHub Actions test environment, and GitHub Pages production environment.
- **Deployment Model:** Static SPA deployment; Vite build artifacts are served by GitHub Pages over HTTPS with the correct GitHub Pages base path.
- **CI/CD:** GitHub Actions runs lockfile dependency install, TypeScript typecheck, lint, Vitest unit/component tests, React Testing Library checks, Playwright E2E tests, axe accessibility checks, production build, `npm audit`, CycloneDX SBOM generation, GitHub Pages deployment, and deployment smoke test.
- **Configuration Management:** Configuration is limited to static build settings such as Vite base path and GitHub Pages deployment settings; no runtime secrets are required.
- **Rollback:** Redeploy the previous known-good commit/build through GitHub Actions or revert to a prior source commit and rerun the deployment pipeline.
- **Infrastructure as Code:** GitHub Actions workflow files and GitHub Pages configuration in the repository define deployment automation; additional IaC is not required for the initial static host.

## 14. Testing Architecture

Required test levels:

- Unit tests
- Integration tests
- Contract tests
- End-to-end tests
- Security tests
- Performance tests, where applicable

Minimum quality expectations:

- **Coverage Threshold:** 80% line coverage for domain and persistence utilities, with critical PRD acceptance flows covered by Playwright E2E tests.
- **Blocking Security Severity:** Unresolved High and Critical dependency/CVE findings block release.
- **Required Build/Lint Checks:** Lockfile dependency install, TypeScript typecheck, lint, Vitest unit/component tests, React Testing Library component tests, Playwright E2E tests, axe accessibility checks for core screens, production build, `npm audit`, CycloneDX SBOM generation, GitHub Pages deployment, and deployment smoke test.

## 15. Architecture Constraints

- ARC-CON-001 — The first release must remain a browser-only React, TypeScript, and Vite SPA.
- ARC-CON-002 — Task data must use browser localStorage only and must not use backend APIs, backend databases, paid storage, or multi-device synchronization.
- ARC-CON-003 — Task data must not be transmitted to backend APIs, analytics services, telemetry services, third-party storage services, or remote logs.
- ARC-CON-004 — The deployment target for the first release is GitHub Pages served over HTTPS.
- ARC-CON-005 — The CI/CD platform for the first release is GitHub Actions.
- ARC-CON-006 — The app must support latest stable Chrome, Edge, Firefox, Safari, mobile Safari, and mobile Chrome, using manual checks where automation is unavailable.
- ARC-CON-007 — The release must pass all Must-priority requirements and release gates before rollout to the one-user pilot.
- ARC-CON-008 — The UI styling layer must use plain scoped CSS or CSS modules with no paid UI kit.
- ARC-CON-009 — The application must use a single-screen SPA model with no client router for the first release.
- ARC-CON-010 — Dependency scanning must block unresolved High and Critical findings, and CI/CD must generate a CycloneDX SBOM.

## 16. Risks and Trade-offs

| ID | Risk/Trade-off | Impact | Mitigation |
|---|---|---|---|
| ARC-RISK-001 | Browser localStorage can be unavailable, full, cleared, corrupted, blocked, or device-local only | User may lose task data or expect persistence that cannot be guaranteed | Show clear persistence/load errors, avoid false success states, validate payloads, document single-browser/device scope, and test failure paths |
| ARC-RISK-002 | No backend or accounts means no multi-device sync, sharing, or centralized recovery | Users may expect capabilities outside first-release scope | Keep product copy and release notes clear that the app is a lightweight single-user demo |
| ARC-RISK-003 | GitHub Pages static hosting has no backend observability or server-side control | Runtime failures may be visible only to the user unless reproduced | Use automated E2E/smoke tests and CI evidence; keep runtime diagnostics non-sensitive and local |
| ARC-RISK-004 | Hand-written runtime validation can drift from TypeScript types | Valid data may be rejected or invalid data may be accepted if tests are incomplete | Co-locate schema tests with domain/persistence tests and require coverage for valid and invalid v1 payloads |
| ARC-RISK-005 | Browser automation may not fully cover Safari and mobile Safari in CI | Release confidence may depend partly on manual checks | Require documented manual acceptance checks where Playwright automation is unavailable |
| ARC-RISK-006 | Dependency vulnerabilities can delay release | High/critical findings may block deployment close to release | Run `npm audit` early in CI and keep dependency set minimal |
| ARC-RISK-007 | Accessible in-app dialog adds implementation complexity | Delete confirmation could fail keyboard/focus requirements if implemented poorly | Include keyboard E2E checks and axe checks for core screens/dialog states |

## 17. Open Architecture Questions

| ID | Question | Owner | Status | Decision/Due Date |
|---|---|---|---|---|
| ARC-OQ-001 | Name the Architecture Owner for this ARD. | Product Owner | Open | TBD |
| ARC-OQ-002 | Name ARD contributors, including engineering, security, and product participants. | Product Owner | Open | TBD |
| ARC-OQ-003 | Name the Security Reviewer for approval. | Product Owner | Open | TBD |
| ARC-OQ-004 | Name the Engineering Lead for approval. | Product Owner | Open | TBD |

## 18. Feature Mapping

| Feature | Relevant Components | Decision IDs | Interface IDs | Security IDs |
|---|---|---|---|---|
| FEAT-001 | COMP-001, COMP-002, COMP-003, COMP-004, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-003, ARD-DEC-004, ARD-DEC-009 | ARD-API-001 | ARD-SEC-002 |
| FEAT-002 | COMP-001, COMP-006, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-003, ARD-DEC-004 | ARD-API-001 | ARD-SEC-001 |
| FEAT-003 | COMP-001, COMP-005, COMP-007, COMP-009 | ARD-DEC-001, ARD-DEC-003, ARD-DEC-004 | ARD-API-001 | ARD-SEC-001 |
| FEAT-004 | COMP-001, COMP-007, COMP-008, COMP-009 | ARD-DEC-002, ARD-DEC-003, ARD-DEC-005 | ARD-API-001 | ARD-SEC-001, ARD-SEC-002 |
| FEAT-005 | COMP-010 | ARD-DEC-001, ARD-DEC-006, ARD-DEC-007, ARD-DEC-008 | ARD-API-002, ARD-API-003 | ARD-SEC-003 |

## 19. Approval

| Role | Name | Decision | Date |
|---|---|---|---|
| Solution Architect | TBD; see ARC-OQ-001 | Pending | TBD |
| Security Reviewer | TBD; see ARC-OQ-003 | Pending | TBD |
| Engineering Lead | TBD; see ARC-OQ-004 | Pending | TBD |