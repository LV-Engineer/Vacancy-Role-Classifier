# Work Plan and Checklist

No deadline. Work in short stages of one to three evenings; each stage ends with something that can be shown (a commit, a table, a run in MLflow). If a stage grows too large, cut the scope of that stage, not the quality: a finished project with two models is better than an unfinished one with five.

## Stage 1: setup and decisions

- [x] 1. Create the repo, the `docs/` folder and the four documents
- [x] 2. Write the class list and the labeling rules (one page); decide the data source and record it in `docs/03_Data.md`
- [x] 3. Choose the data versioning approach, set it up on an empty dataset, record the choice in `docs/03_Data.md`
- [x] 4. Start MLflow (docker compose) and log a test run to check the setup

## Stage 2: data

- [x] 5. Collect or download the postings; label them (aim for a few hundred, balanced enough that every class has examples)
- [x] 6. Data checks and the split; record the facts in `docs/03_Data.md`
- [x] 7. Version the dataset; commit the version file, not the raw data

## Stage 3: experiments

- [x] 8. Baselines: TF-IDF + Logistic Regression on the title only and on the description; compare macro-F1
- [x] 9. Embeddings + classifier; test Ukrainian and English
- [x] 10. Error analysis (confusion matrix, read 30 wrong predictions) and level prediction (separate classifier or a second head)

## Stage 4: registry and service

- [x] 12. Register the best model, write and apply the promotion rule
- [ ] 13. Reproducibility check: re-run from the data version and commit, the metrics must match
- [ ] 14. FastAPI endpoint, Pydantic schemas, Dockerfile, docker compose with MLflow
- [ ] 15. Tests and CI

## Stage 5: finish

- [ ] 16. Insights for each of the five questions in the brief
- [ ] 17. README: problem, how to reproduce, results table, limitations