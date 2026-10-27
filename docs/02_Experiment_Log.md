Experiment Log

Log each experiment, check and decision. Add a row after each run and commit the row together with the code or config that produced it. Put the MLflow run id in the Result column.

| # | Experiment or check | Setup | Result (MLflow run id) | Notes |
| --- | --- | --- | --- | --- |
| 0 | MLflow setup check | docker compose, MLflow 3.17.2 (client and server pinned to the same version), SQLite metadata store and local artifact store in `mlflow-data/`, experiment `smoke-test` | 100251e1f91641a29fb544ec3253de66 | parameter, metric and artifact (`note.txt`) logged and visible in the UI at `localhost:5000` |
| 1 | Data source and labeling decision | Option C (both). Public: LinkedIn Job Postings 2023-2024 (Kaggle, arshkon, CC BY-SA 4.0, 123,849 rows, columns `title` and `description` present). Own: Djinni vacancies, public pages only, `/jobs/` allowed by robots.txt, polite sequential collection. Classes and labeling rules: `docs/06_Labeling_Rules.md`. Data versioning: DVC with an S3 remote (gz-compressed raw data) | n/a (decision) | Public data gives volume for train, but its role labels come from title keywords (weak labels); the hand-labeled Djinni set is used for validation and test only. DOU was dropped: its terms (clause 2.5) forbid automated collection without the administration's consent. LinkedIn data is scraped from the site, so its terms may apply; used for a private learning project, raw files not redistributed, source credited in the README. Experience level: derived from `formatted_experience_level` (public) and the "years of experience" field (Djinni). S3 chosen for DVC although paid (about $0.01 per month for this size); to be closed at the end of the project |
| 2 | Data checks | row counts, duplicates, empty texts, class balance, languages | | |
| 3 | Split | stratified train, validation, test; seed; duplicates removed first | | |
| 4 | Baseline: title only | TF-IDF + Logistic Regression on the title | | |
| 5 | Baseline: description | TF-IDF + Logistic Regression on the description | | compare with row 4 |
| 6 | Embeddings + classifier | embedding model name, classifier, parameters | | test Ukrainian and English |
| 7 | Optional fine-tuning | small transformer, learning rate, epochs | | stop if gain is small |
| 8 | Error analysis | confusion matrix, read 30 wrong predictions | | |
| 9 | Level (junior or not) | separate classifier or a second head | | |
| 10 | Registry rule | promotion Staging to Production, the metric and the threshold | | |
| 11 | Reproducibility check | re-run from the data version and commit | | metrics must match |
| 12 | Service checks | prediction latency, tests, CI | | |

# Conclusions

**What the data shows**
- 

**What did not work or had to be fixed, and how it was found**
- 

**What to improve**
- 
