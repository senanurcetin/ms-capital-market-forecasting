# NEXUS Kaggle publications

These folders contain the source and metadata of the two existing public notebooks.
Both were updated with the validation correction and re-executed on Kaggle CPU on
21 October 2026. The main analysis reads the existing derived-results dataset; the
experiment summary re-renders its historical figure. Neither retrains the full model
pipeline or creates a competition submission.

- [My Hold-out Was a Lucky Stretch](https://www.kaggle.com/code/senanuretin/ms-capital-my-hold-out-was-a-lucky-stretch): correction published as version 4.
- [Four Experiments That Did Not Help](https://www.kaggle.com/code/senanuretin/ms-capital-four-experiments-that-did-not-help): correction published as version 2.

`python scripts/check_kaggle_publications.py` performs an opt-in readback using the
owner's normal Kaggle authentication. It checks completed execution and hashes the
ordered cell sources against these files, ignoring cell IDs and execution outputs.
It does not publish or start a run. Credentials and competition data are excluded.
