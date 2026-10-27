# Labeling Rules

Draft v1. Change a rule only together with a note in `02_Experiment_Log.md`, because every change can shift the labels and needs a new data version.

## General rules

- Label by the **duties and requirements**, not by the title. The title may be misleading.
- If a posting mixes two roles, label by the **main duty** (the one that takes most of the described work). If still unclear, label `Other` and add it to the hard examples.
- Ignore the company name, perks and salary; they carry no role signal.
- One label per posting. Do not edit the text, store it as it was read.

## Classes

| Role | Main duties (label it when most of these are true) | Typical signals |
| --- | --- | --- |
| Data Analyst | Dashboards, reports, ad-hoc analysis, SQL, business metrics, A/B test reading | SQL, BI tools (Power BI, Tableau, Looker), Excel, stakeholders, KPIs |
| Data Scientist | Building statistical and ML models to answer a business question, experiments, feature work | scikit-learn, statistics, hypothesis testing, modeling, notebooks |
| ML / AI Engineer | Putting models into production, training pipelines, MLOps, LLM and deep learning systems | PyTorch, TensorFlow, MLOps, model serving, LLM, deployment, GPU |
| Python Backend | Building services and APIs in Python; data and ML only as a side topic | Django, FastAPI, Flask, REST, databases, microservices |
| Other | Anything else: lead generation, sales, QA, frontend, pure data engineering, management, unclear postings | duties do not match the four classes above |

## Level

- `junior`: the posting asks for up to about 1 year of experience, or says junior, trainee, intern or "no experience required".
- `not_junior`: everything else, including middle, senior, lead, and postings that do not state the level but ask for 2 or more years.
- If the title says "Junior" but the requirements ask for 3 or more years, trust the **requirements** and label `not_junior`; note it as a hard example.

## Hard examples

Add each unclear posting here with the label chosen and the reason. This list explains the label noise in the README.

The postings are stored with their reasons in the `note` column of `data/processed/djinni.csv` (55 postings). Recurring patterns:

| Pattern | Label chosen | Reason |
| --- | --- | --- |
| Title says one role, duties say another (a "Senior Python Backend Engineer" doing analytics, a "Senior Data Scientist" doing data engineering, a "Data Analyst" who is a sports trader) | by the duties | general rule 1 |
| Business Analyst (requirements, workshops, 1C, Odoo, BAS) | Other | no dashboards, SQL or metrics work |
| Backend in Java, Node.js, Go, PHP or C#; QA automation in Python | Other | Python Backend means services built in Python |
| Full-stack Python with LLM features | Python Backend when the requirements are FastAPI, REST, React; ML / AI Engineer when the duties are prompts, agents, evals | main duty |
| Forward-deployed AI engineer (client-facing, builds AI solutions) | ML / AI Engineer | AI and LLM systems are the main deliverable |
| Pure data engineering (ETL, pipelines, data lake) | Other | pure data engineering is Other in the class table |
| Level: the title and the years field disagree | trust the requirements | level rule above; 5 postings changed |

## Label noise check

- The v0.1 Djinni labels were set by Claude in one pass from the duties, without a human review and without a re-label check. The label noise is therefore unknown; metrics on the test set measure agreement with one annotator.
- If time allows later: re-label a random sample of about 30 postings a few days apart without looking at the first labels and record the agreement in `05_Data.md`. Any label change creates a new data version.
- For the public dataset, labels derived from titles are treated as weak labels; only the labeled Djinni set is used for the test set (its dev part also goes into training, see log row 3).
