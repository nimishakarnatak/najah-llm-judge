# Najah Stage 2 research workspace

A Replit-ready Python/Streamlit app for human adjudication, participant-separated splits, OpenAI/Anthropic judge comparison, protocol freeze and per-measure held-out evaluation. This package is a standalone app; it does not modify the existing Netlify rating service.

## Run in Replit

1. Create a Python app and upload the contents of this folder, with `app.py` at the project root.
2. In the Replit shell run `pip install -r requirements.txt`.
3. Add four Replit Secrets: `APP_PASSWORD`, `ADJUDICATOR_PASSWORD` (different passwords), `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY`. Do not paste API keys into source code or chats.
4. Click Run. The `.replit` file starts Streamlit on port 8501 and maps it to the web preview. Alternatively run `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`.
5. Sign in as Study analyst to import data and assign splits. Give the adjudicator password only to the independent human review team.
6. After adjudication, register candidate model IDs available to your accounts, run training/validation batches, compare validation metrics, choose the primary and secondary, and freeze. Then run test batches and download per-measure results.

Use one study per database. `NAJAH_DB_PATH` optionally changes the default `data/study.sqlite`. Freeze and splits cannot be undone through the UI. Preserve the database and downloads. This SQLite version is intended for one study workspace with modest concurrency. For deployed, multi-instance operation replace SQLite with persistent PostgreSQL, shared job scheduling and individual user accounts. Do not rely on ephemeral deployment storage for study records.

## Required data

The supplied rating CSVs can be imported directly. The importer accepts complete primary rows, excludes admin/judge/demo/draft rows and rater names beginning with test or dummy, deduplicates identical episode/rater records, rejects conflicting updates, and removes rater email/name from saved primary data. Review your actual rater roster; rename-based exclusions are a convenience, not a substitute for the study roster.

Episode JSON must be a list or an object with an `episodes` list. See `config/episode_example.json` (synthetic, not a Najah participant). Each episode needs:

| Field | Requirement |
|---|---|
| `episode_id` | Must match the rating CSV exactly |
| `participant_id` | Stable de-identified conversation-participant ID, **not evaluator/rater ID** |
| `module` | Module identifier |
| `language` | Verified original language, including mixed-language labels as appropriate |
| `sampling_weight` | Positive inverse inclusion weight appropriate to the target population |
| `turns` | Original text, ordered list of `turn_id`, `role`, `text`; optional `segment`: prior/episode/subsequent |
| `delivery_evidence` | Optional verified artifact/log evidence; empty means no independent verification supplied |
| `treatment` | Optional metadata used for later analysis, excluded from provider requests |

Do not fabricate participant IDs by treating every episode as a different person. If each sampled episode is demonstrably from a unique participant, document that fact. Keep necessary prior context and mark the focal episode explicitly. Include only permitted subsequent outcome evidence; do not leak subsequent task behavior into focal quality scoring. Review de-identification before provider calls.

The existing Netlify admin service exposes ratings CSV downloads and read-only rater queues. Its visible rating export does not contain conversation text or participant IDs. `docs/NETLIFY_EXPORT_SPEC.md` describes the export to add in its source project. This app can operate as soon as that JSON is supplied; no credentials from the existing app are bundled.

## Protocol and adjudication

The revised Word overview accompanies this package. Default allocation is 50% training, 25% validation, 25% test by participants, approximately 150/75/75 episodes when each episode comes from one participant. Random seed is saved. All episodes from the same participant remain together; module/language/treatment balance is displayed and must be checked. No stratification or quota guarantee is implied.

Training supports rubric/prompt development, not fine-tuning. Model versions and prompts are registered as immutable candidates. A changed prompt becomes a new candidate. Holdout results are locked until freeze and cannot be used to choose a new primary judge. Already exposed episodes are retrospective evidence, not a fresh prospective test.

Independent adjudicators review every disagreement, every missing/uncertain label, every safety/critical label, matching material-failure scores, legacy NA decisions and a reproducible approximately 10% agreement sample. Other matching primary 2/3 labels become a reference automatically, retaining their source. Adjudication saves reviewer ID, turn evidence, reason, prior values and time. Reviewers can leave unresolved labels explicitly. No LLM predictions appear in the human-review workspace; treatment labels are also hidden. A reviewer must not enter an original rater's ID.

Role passwords isolate ordinary app views but do not establish individual identity, employment roles, or independent expertise. The team must appoint qualified multilingual adjudicators and control password distribution. The app is not a full multi-user identity-management system. A data custodian with filesystem access can inspect the holdout; organizational separation remains necessary. Human adjudication should be completed before developers inspect LLM outputs; the UI alone cannot undo earlier exposure.

## Per-measure validation

`config/gates.json` registers a gate for each of nine ordinal criteria, seven critical-failure flags and task outcome. Thresholds are **proposed study choices**, not established standards. Review and approve before freezing. Other participant behavior/reaction and gender-handling measures remain human-coded in this version and must not be described as LLM-validated.

All gates require at least 60 evaluable cases, 40 participant clusters, 90% reference coverage and 95% prediction coverage. Ordinal gates require 10 scores below 3, five NA cases, quadratic kappa ≥0.60 (95% CI lower ≥0.40), MAE ≤0.35 (upper ≤0.50), and applicability balanced accuracy ≥0.85. A numeric-quality result may be informative even if NA decisions have insufficient evidence, but full automation remains withheld.

Critical flags require 20 positive and 20 negative human cases, recall ≥0.95 (lower ≥0.80), precision ≥0.80 (lower ≥0.65), specificity ≥0.90 (lower ≥0.80). Task outcome requires ten reference cases in each of four substantive outcome classes, macro F1 ≥0.80 (lower ≥0.70), minimum class recall ≥0.80 (lower ≥0.60). Unknown/unresolved reference labels are excluded from numeric/class metrics and reduce reference coverage. Model abstentions, invalid outputs and provider failures reduce prediction coverage; categorical abstentions on known cases count as errors.

Confidence intervals use 1,000 deterministic participant-cluster percentile bootstrap replicates with sampling weights retained. A metric CI is unavailable when more than 5% of its bootstrap replicates are undefined; that causes insufficient evidence. Critical-flag proportion intervals are widened with an approximate Wilson guard using Kish effective participant counts, preventing perfect detection from producing a zero-width CI. These are pointwise, not simultaneous confidence intervals. Metric exports include unweighted and weighted confusion counts. Subgroup checks remain part of the subsequent study analysis. Small, sparse classes and highly unequal weights require additional assessment; a bootstrap does not create missing positive cases.

A held-out set of 75 episodes is unlikely to validate all rare flags or NA decisions. Do not lower thresholds after seeing test results. Gather fresh criterion-specific positive cases in a separate challenge study, retain cohort identity, and report enriched challenge metrics separately. This version implements the existing-cohort train/validation/test workflow, not challenge-cohort recruitment, full-corpus scoring or 30-day follow-up analysis. Statistical passes are provisional until evidence citations are independently audited and language/module applicability is confirmed. Safety score success does not establish detection of critical failures.

## Provider calls and reproducibility

The app uses the [OpenAI Responses API](https://developers.openai.com/api/docs/quickstart) and [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create). Enter explicit model IDs rather than relying on a changing latest alias. Requests omit human labels, treatment, participant/rater IDs, weights, and other judges' responses. They include rubric, module, language, original turns and permitted delivery evidence. Evidence turn IDs and values are validated. Output schema violations and truncations remain errors, never synthetic scores.

Runs store requested and returned model identifiers, response ID, token usage, hashes and timestamps. OpenAI requests set `store:false`; this is not a claim about all provider retention policies. Judge calls use provider-default sampling behavior because not every model accepts the same temperature controls; selection is reproducible at the recorded configuration level, not guaranteed bit-for-bit. Check model stability and repeated-run variability on training before freeze. Calls retry transient HTTP failures up to three times; successful judgments are never overwritten.

Provider/API end-to-end behavior and deployment on the user's Replit account require your API keys and actual episode export. Local tests use mocked provider responses and synthetic records; no paid calls are made during packaging.

## Tests and backup

Optional app-provided English translations may be imported as `text_en` on each turn. Human adjudicators can enable English reading support beside the original text. These machine translations are not human verified; consult the original when meaning is uncertain. Provider requests omit `text_en` so judge evaluation uses original-language text consistently. Translations do not establish the source language or participant identity.

Run `python -m unittest discover -s tests -v`. Tests exercise leakage prevention, immutable freeze, adjudication, missing-data behavior, critical-class gates, provider request blinding, JSON/evidence validation and a mocked complete workflow. Run `python -m compileall app.py najah` for syntax checks.

Before freeze, analyst downloads exclude test reference labels; full data-custodian backups can be taken directly from SQLite. After freeze, Exports provides study backup JSON, split manifest, reference labels and frozen protocol. The complete backup JSON is an audit export, not an automated restore format; to resume, preserve the original SQLite file. Never commit participant data, study passwords or API keys.
