Project 5: Vacancy Role Classifier (MLflow, reproducible pipeline, FastAPI)

Repo: https://github.com/LV-Engineer/Vacancy-Role-Classifier

# Business problem

* Who uses the model: a job seeker (and, in the same way, a recruiting team) who has to sort hundreds of job postings
* Why a model: job titles are misleading (for example "Data Analyst" in the title while the duties are lead generation), so the title alone is not a reliable filter; the description text carries the real signal
* Decisions supported: which postings to open and apply to, which to skip, and how to rank new postings by fit with the target roles

Questions the project must answer (each one must end with a number and a short conclusion):

1. Does the description text predict the role better than the title alone? (macro-F1 of both)
2. Which model gives the best trade-off between quality, speed and cost: TF-IDF + Logistic Regression, sentence embeddings + classifier, or a fine-tuned small transformer?
3. Where does the model fail? Which role pairs are confused most (for example Data Analyst vs Product Analyst)?
4. Can the level (junior or not) be predicted, and what does a wrong "junior" call cost the user?
5. Can every registered model be reproduced from its data version, parameters and code commit?

# Data

* Source and labels: decided: both (a public LinkedIn dataset for training, own labeled Djinni postings for evaluation); see `docs/03_Data.md`
* Classes (draft, change after looking at the data): Data Analyst, Data Scientist, ML / AI Engineer, Python Backend, Other
* Language: postings may be Ukrainian or English; the choice of the embedding model must be tested on both
* The raw files are not committed; versions of the dataset are tracked by the data versioning approach chosen in the work plan (record the choice in the log)
* Data issues to check and record: duplicate postings, empty or very short descriptions, mixed languages, class imbalance, label noise (two people would label the same posting differently), titles that leak the label

# Tech stack

| Area | Tools |
| --- | --- |
| Language and libraries | Python, pandas, scikit-learn, sentence-transformers (or a similar embedding library), PyTorch only if fine-tuning is done |
| Experiment tracking | MLflow: parameters, metrics (macro-F1, per-class F1), artifacts (confusion matrix), models |
| Model registry | MLflow Model Registry with stages Staging and Production, the promotion rule written in the log |
| Data versioning | one data versioning approach, chosen in step 2 of the work plan, so that a model can be tied to the exact data it was trained on |
| Serving | FastAPI, Pydantic validation, one prediction endpoint (role, level, probabilities) and a health endpoint |
| Packaging and quality | Docker and docker compose, pytest, CI on GitHub Actions, type checks |
| Version control | Git, GitHub |

# Approach

* Define the classes and the labeling rules first (one page), then label or load the data
* Split before any tuning: stratified train, validation and test, with a fixed seed; duplicates removed before the split so the same posting is not in two parts
* Baseline: TF-IDF + Logistic Regression; then embeddings + a classifier; optionally a fine-tuned small transformer; every run is logged in MLflow
* Compare on macro-F1 and per-class F1, not accuracy; look at the confusion matrix and read the wrong predictions
* Register the best model, promote it Staging to Production by a written rule, and check reproducibility by re-running from the data version and the commit
* Serve the Production model with FastAPI in Docker; add tests and CI
* README: the problem, how to reproduce, the results table, honest limitations

# Results

To fill after the experiments (real numbers only, each with the MLflow run id that produced it):

* Title only vs description: macro-F1 of both
* Model comparison table: quality, training time, prediction latency
* Per-class F1 and the main confusions
* Level prediction quality (junior or not)
* Reproducibility check: same metrics when re-run from the data version and commit
* Limitations: small hand-labeled set, one source of postings, label noise, no live data, class definitions are subjective

# Takeaway

To fill: what the numbers say about the task, which model is worth keeping and why, and what would be done with more data (more sources, active learning on the uncertain postings, retraining on a schedule).
