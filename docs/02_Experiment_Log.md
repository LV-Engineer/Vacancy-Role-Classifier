Experiment Log

Log each experiment, check and decision. Add a row after each run and commit the row together with the code or config that produced it. Put the MLflow run id in the Result column.

| # | Experiment or check | Setup | Result (MLflow run id) | Notes |
| --- | --- | --- | --- | --- |
| 1 | Data source and labeling decision | own labels, public dataset or both; class list | | decision and reason |
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
