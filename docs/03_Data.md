# Data

The raw data files are **not committed**. Only the small data version file (the pointer to the data) and small samples for tests may be committed.

## Source (decided: option C, both)

| Option | What it is | Pros | Cons |
| --- | --- | --- | --- |
| A. Own labeling | Collect postings by hand (copy the text of vacancies you read anyway) and label the role and level | Matches the real use case, Ukrainian and English, you know the label rules | Slow; a few hundred examples at most; label noise |
| B. Public dataset | A job postings dataset from Kaggle or similar; derive the role label from the title | Large, ready | Mostly English; labels come from titles, so the model may only learn the titles; check the license |
| C. Both | Public data for pre-training the baseline, own labeled set for validation and test | Best evaluation honesty | More work |

Rules for any source:

- Check the license and the site terms before using or storing postings. If scraping is not allowed, do not scrape; collect by hand or use a dataset that permits the use.
- Store only the text and the labels, no personal data, no contact details, no company internals.
- Credit the source in the README and do not redistribute the raw files.
- The test set is labeled from the duties (v0.1: by Claude, without a human review) and never used for tuning.

## Labeling rules

Change a rule only together with a note here: every change can shift the labels and needs a new data version.

- Label by the **duties and requirements**, not by the title. If a posting mixes two roles, label by the main duty; if still unclear, label `Other` and note it. One label per posting; ignore company name, perks and salary; do not edit the text.

| Role | Label it when the main duties are |
| --- | --- |
| Data Analyst | dashboards, reports, ad-hoc analysis, SQL, business metrics |
| Data Scientist | statistical and ML models for a business question, experiments |
| ML / AI Engineer | models in production, training pipelines, MLOps, LLM and deep learning systems |
| Python Backend | services and APIs in Python; data and ML only as a side topic |
| Other | anything else: sales, QA, frontend, pure data engineering, management, unclear postings |

- Level: `junior` if the posting asks for up to about one year or says junior, trainee, intern or "no experience required"; otherwise `not_junior`. If the title says junior but the requirements ask for 3 or more years, trust the requirements.

Hard cases (the 55 postings with reasons are in the `note` column of `data/processed/djinni.csv`):

- Title says one role, duties another: label by the duties.
- Business Analyst (requirements, workshops, 1C, Odoo): `Other`, no dashboards, SQL or metrics.
- Backend in Java, Node.js, Go, PHP or C#, QA automation in Python: `Other`; Python Backend means services built in Python.
- Full-stack Python with LLM features: Python Backend when the requirements are FastAPI, REST, React; ML / AI Engineer when the duties are prompts, agents, evals.
- Forward-deployed AI engineer: ML / AI Engineer, AI and LLM systems are the deliverable.
- Pure data engineering (ETL, pipelines, data lake): `Other`.

## Schema of the own dataset

| Column | Meaning |
| --- | --- |
| `id` | stable identifier of the posting (not a link to a person) |
| `title` | job title |
| `description` | full text of duties and requirements |
| `language` | `uk` or `en` |
| `role` | label from the class list |
| `level` | `junior` or `not_junior` |
| `source` | where the text came from (site name or dataset name) |

Verify the real columns after loading if a public dataset is used.

## Data versioning

Every dataset change (new rows, relabeling, a new split) creates a new data version. A trained model in MLflow must record the data version it used, so that the run can be reproduced. The approach is DVC with an S3 remote (see Decisions).

## Facts about the data

Real numbers for v0.2 (v0.1 differs only in the LinkedIn weak labels, see Decisions):

- LinkedIn: 123,220 usable postings, all English; the weak role is `Other` for 99.1%. After exact dedupe the target classes shrink from 1,118 to 991. After cleaning and downsampling "Other": 2,191 postings (Other 1,200, Data Analyst 377, ML / AI 234, Data Scientist 216, Python Backend 164). Level is unknown for 23.7% (623 of the 2,191).
- Djinni: 1,497 collected (uk 762, en 735; junior 351 by the years field), 397 labeled (Other 215, ML / AI 73, Data Analyst 45, Python Backend 42, Data Scientist 22). Non-Other duplicates 253 to 247 (2.4%). 5 levels were overridden because the years contradicted the title; the 55 hard cases are in the `note` column.
- Splits: `train` 2,488 (Other 1,361, Data Analyst 411, ML / AI 289, Data Scientist 232, Python Backend 195; level not_junior 1,291, junior 574, unknown 623); `djinni_dev` 297 (not_junior 257, junior 40); `djinni_test` 100 (Other 54, ML / AI 18, Python Backend 11, Data Analyst 11, Data Scientist 6; not_junior 88, junior 12).
- Description length (median and extremes): not measured yet.
- Share of postings where the title alone gives the label: the v0.1 title keyword rule agreed with the duty-based label in about 79% of the 397 labeled Djinni postings (not representative: all non-Other plus 150 Other).
- Agreement on a re-labeled sample (label noise estimate): not measured; the labels have one annotator (Claude) and no human review.

Files (not committed, tracked by DVC): `data/raw` (scraped and downloaded sources), `data/processed` (`linkedin.csv`, `djinni.csv` with the labels), `data/splits` (`train.csv`, `djinni_dev.csv`, `djinni_test.csv`). Versions: v0.1 (weak labels from title keywords, tag `v0.1-data`) and v0.2 (weak labels follow the labeling rules).

## Split and evaluation

- Split (`src/data/split.py`): `StratifiedGroupKFold`, 4 folds, one fold is the test set (25%), seed 42, stratified by role. The group is the first 80 characters of the description (a rough proxy for one company), so such postings never land on both sides. Duplicates are removed first and no test description appears in the LinkedIn set. Level is not stratified.
- `train` = LinkedIn (weak labels) plus `djinni_dev`; `djinni_test` is not touched until the end.
- Models are compared by cross-validation on the 297 Djinni dev rows: 5 folds with the same group key, LinkedIn always in training, every fold validated only on Djinni rows it has not seen. In training a Djinni row weighs more than a LinkedIn row (weight 12-25).
- Limits: the test set is small (6 Data Scientist and 12 junior postings), so per-class metrics are noisy. The Djinni weight and the v0.2 label rules were chosen after looking at dev errors, so dev scores are optimistic; only `djinni_test` is an honest estimate.

## Results and promotion (v0.2)

Macro-F1: cross-validation on Djinni dev, and `djinni_test` (used once, for the final model).

| Model | dev CV | test |
| --- | --- | --- |
| rule baseline (title keywords) | 0.934 | 0.890 |
| TF-IDF, title (Djinni weight 1) | 0.913 | 0.882 |
| TF-IDF, title + description (weight 1) | 0.920 | not run |
| embeddings, title (weight 25) | 0.885 | not run |
| TF-IDF, description (weight 25) | 0.776 | not run |

- The description adds nothing over the title under these labels: 0.920 against 0.913 is within noise. Titles and the v0.2 weak labels come from the same rules, and the Djinni labels were set with the title keyword hint visible.
- On the test set the model is on the level of the rule (0.882 against 0.890, with 100 postings the noise is about 0.05). The rule encodes the labeling rules the labels were made with, so it is not an independent reference.
- Registered model: `vacancy-role-classifier` version 1 (TF-IDF on the title, weight 1, trained on the whole `train.csv`), alias `production`. The service loads `models:/vacancy-role-classifier@production`.
- Promotion rule: a version gets the `production` alias when `test_macro_f1` is at least 0.85 (the rule baseline 0.890 minus about 0.04, the noise of a 100-posting test set); otherwise it stays `candidate`. The threshold was set after the first test result (0.882), so for version 1 it is not independent; for later models it works as a real gate.

## Decisions

- Source: option C. LinkedIn Job Postings 2023-2024 (Kaggle, arshkon, CC BY-SA 4.0) gives volume but its role labels are weak (title keywords); own Djinni vacancies (public `/jobs/` pages) give the labeled evaluation set.
- DOU is dropped: its terms (clause 2.5) forbid automated collection without the administration's consent. LinkedIn data is scraped from the site, so its terms may apply: private learning project, raw files not redistributed, source credited in the README.
- `djinni_dev` (297 rows, 9% of train) goes into training so the model sees Ukrainian text; only `djinni_test` stays out. Parameters are chosen by cross-validation on dev.
- The DVC remote is S3: paid, about $0.01 per month for this size; close it at the end of the project.
- v0.2: after the error analysis the weak LinkedIn labels follow the labeling rules (`business analyst`, QA and automation titles are `Other`; Python Backend needs python, django, fastapi or flask, a plain "backend" is `Other`). With the v0.1 labels the models had learned that every business analyst is a Data Analyst and every backend is a Python Backend, which the duty-based labels contradict.
- Fine-tuning is skipped: the errors come from weak labels, not from model capacity.
