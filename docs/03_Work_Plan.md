# Work Plan and Checklist

No deadline. Work in short stages of one to three evenings; each stage ends with something that can be shown (a commit, a table, a run in MLflow). If a stage grows too large, cut the scope of that stage, not the quality: a finished project with two models is better than an unfinished one with five.

## Stage 1: setup and decisions

- [x] 1. Create the repo, the `docs/` folder and the five documents
- [x] 2. Write the class list and the labeling rules (one page); decide the data source and record it in log row 1
- [x] 3. Choose the data versioning approach, set it up on an empty dataset, record the choice in the log
- [x] 4. Start MLflow (docker compose) and log a test run to check the setup

## Stage 2: data

- [ ] 5. Collect or download the postings; label them (aim for a few hundred, balanced enough that every class has examples)
- [ ] 6. Data checks and the split; fill log rows 2 and 3
- [ ] 7. Version the dataset; commit the version file, not the raw data

## Stage 3: experiments

- [ ] 8. Baselines: title only and description (rows 4 and 5)
- [ ] 9. Embeddings + classifier (row 6)
- [ ] 10. Error analysis, level prediction (rows 8 and 9)
- [ ] 11. Optional: fine-tuning (row 7); skip if time is short

## Stage 4: registry and service

- [ ] 12. Register the best model, write and apply the promotion rule (row 10)
- [ ] 13. Reproducibility check (row 11)
- [ ] 14. FastAPI endpoint, Pydantic schemas, Dockerfile, docker compose with MLflow
- [ ] 15. Tests and CI (row 12)

## Stage 5: finish

- [ ] 16. Insights for each of the five questions in the brief
- [ ] 17. README: problem, how to reproduce, results table, limitations