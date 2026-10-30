# Insights

Numbers are macro-F1 unless noted. "dev CV" is 5-fold cross-validation on the 297 Djinni dev postings with LinkedIn always in training; "test" is `djinni_test` (100 postings), used once per final model. Data version v0.2 unless stated. Run ids are MLflow runs.

## 1. Does the description predict the role better than the title?

No. TF-IDF + Logistic Regression, dev CV: title 0.913, description 0.762, title + description 0.920 (runs `305f027a`, `50625b01`, `2e4366ef`). On v0.1 it was the same: 0.776 and 0.691. The gap between 0.913 and 0.920 is within noise.

**Conclusion:** under these labels the description adds no signal over the title. Caveat: the LinkedIn labels come from title rules and the Djinni labeler saw the title hint, so this partly measures how the labels were made, not what a posting says.

## 2. Which model is the best trade-off?

| Model | dev CV (Djinni weight 1) | Cost |
| --- | --- | --- |
| TF-IDF + Logistic Regression, title | 0.913 | under 1 s, a few KB |
| Embeddings (multilingual-e5-small) + Logistic Regression, title | 0.827 (0.885 with weight 25) | about 12 s to encode, a 470 MB model, PyTorch |
| Fine-tuned transformer | skipped | errors come from weak labels, not from model capacity |

Prediction latency of the production service (role and level models in one request, a 1,200-character description, 200 calls on a laptop after 10 warm-up calls, timed with `time.perf_counter` around `PredictionService.predict` and around `POST /predict`): in the process p50 4.6 ms and p95 7.4 ms; through Docker over HTTP p50 52.7 ms and p95 60.8 ms. The extra ~48 ms is most likely the HTTP and Docker network layer on Windows, not the models (not investigated). The latency of the embeddings model was not measured.

**Conclusion:** TF-IDF is the best and the cheapest, and it is the model in production: a prediction takes about 5 ms of compute.

## 3. Where does the model fail?

With the v0.1 labels there were 71 errors out of 297, 68 of them `Other` predicted as a target class: the weak LinkedIn labels called every "business analyst" a Data Analyst and every "backend" a Python Backend. After the labels were made to follow the labeling rules (v0.2) the same model made 23 errors (macro-F1 0.776 to 0.913). The most confused pair is `Other` and ML / AI Engineer (9 of 23), then `Other` predicted as Python Backend (5). On the test set Python Backend is the weakest class (F1 0.741), mostly `Other` postings predicted as Python Backend (4).

**Conclusion:** the remaining errors are titles that say AI, Data Scientist or Python while the duties say `Other`. Fixing the labels gave a bigger gain (+0.14) than any model change.

## 4. Can the level (junior or not) be predicted?

Weakly. Test: macro-F1 0.695 (junior F1 0.444, precision 0.67, recall 0.33), the title keyword rule gets 0.728. The model finds 4 of 12 junior postings and wrongly calls 2 of 88 not-junior postings junior.

**Conclusion:** a missed junior (8 of 12) costs a junior job seeker most, because that posting never reaches them; a false "junior" is rare (2 of 88) and costs one wasted click. The model barely passed the promotion gate (0.695 against 0.688) and is no better than the rule.

## 5. Can every registered model be reproduced?

Yes, for both models. Each production run was re-trained from the same data version (`data_splits_md5`) and code, and 4 of 4 test metrics matched exactly (tolerance 1e-9): the role run `c767e088` (check `71f0b12b`) and the level run `f4c2c41c` (check `489990b3`).

**Conclusion:** training is deterministic. Limit: the checks ran on one machine, not on a clean clone.
