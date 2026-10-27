# Data

The raw data files are **not committed**. Only the small data version file (the pointer to the data) and small samples for tests may be committed.

## Source (open decision, record the choice in log row 1)

| Option | What it is | Pros | Cons |
| --- | --- | --- | --- |
| A. Own labeling | Collect postings by hand (copy the text of vacancies you read anyway) and label the role and level | Matches the real use case, Ukrainian and English, you know the label rules | Slow; a few hundred examples at most; label noise |
| B. Public dataset | A job postings dataset from Kaggle or similar; derive the role label from the title | Large, ready | Mostly English; labels come from titles, so the model may only learn the titles; check the license |
| C. Both | Public data for pre-training the baseline, own labeled set for validation and test | Best evaluation honesty | More work |

Rules for any source:

- Check the license and the site terms before using or storing postings. If scraping is not allowed, do not scrape; collect by hand or use a dataset that permits the use.
- Store only the text and the labels, no personal data, no contact details, no company internals.
- Credit the source in the README and do not redistribute the raw files.
- The test set is labeled by hand and never used for tuning.

## Labeling rules (write the final version before labeling)

Draft classes: Data Analyst, Data Scientist, ML / AI Engineer, Python Backend, Other.

- Label by the **duties and requirements**, not by the title alone (this is the point of the project).
- If a posting mixes two roles, label by the main duty; if unclear, label Other and note it.
- Level: `junior` if the posting asks for up to about one year or says junior, trainee or intern; otherwise `not_junior`.
- Keep a short list of hard examples and how they were labeled (it helps to explain label noise in the README).

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

Every dataset change (new rows, relabeling, a new split) creates a new data version. A trained model in MLflow must record the data version it used, so that the run can be reproduced. The approach is chosen in step 3 of the work plan and recorded in the log.

## Facts about the data

To fill after checking (write real numbers only):

- Number of postings, per class and per language
- Duplicate share before and after cleaning
- Description length (median and extremes)
- Share of postings where the title alone gives the label
- Agreement on a re-labeled sample (label noise estimate)
