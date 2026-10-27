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

| id | Why it was hard | Label chosen | Reason |
| --- | --- | --- | --- |
| | | | |

## Label noise check

- After labeling, re-label a random sample of about 30 postings a few days later without looking at the first labels.
- Record the agreement (share of equal labels) in `05_Data.md`, section "Facts about the data".
- For the public dataset, labels derived from titles are treated as weak labels; only the own hand-labeled set is used for validation and test.
