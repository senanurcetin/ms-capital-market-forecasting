"""Hyperparameter search - and how much of its gain is real.

The search itself is routine. The part worth building carefully is the accounting, because
in this project the noise floor is unusually well characterised and it is HIGH:

    fold-to-fold cosine std       0.0041      (walk-forward, ensemble)
    block-to-block period std     0.0091      (notebook 04)

Against that, a tuning gain of 0.002 means nothing on its own. Two things follow.

SELECTION OPTIMISM IS THE MAIN RISK, NOT OVERFITTING THE TRAINING DATA

Running N trials and keeping the best does not only find good parameters - it also finds
favourable noise. The reported best-of-N score is biased upward by roughly the expected
maximum of N draws from the fold-noise distribution, and with N in the dozens that term is
comparable to any real effect being chased.

So the winner is re-scored under a DIFFERENT resampling of the same protocol: fresh
bagging and feature-sampling seeds, everything else identical. Whatever survives is
attributable to the parameters; whatever evaporates was the search fitting noise. Both
numbers are reported.

THE HOLD-OUT IS NOT THE JUDGE

Notebook 04 established that months 65-70 are an unusually favourable period, so an
absolute hold-out score cannot arbitrate anything. It is used only as a PAIRED comparison
- tuned against default, same rows, same protocol - where the period effect is common to
both arms and cancels.

Searching on a subsample and confirming the winner on full data is deliberate: a full
5-fold pass costs minutes, and dozens of them would buy less than the accounting above.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.metrics import cosine_similarity
from src.evaluation.temporal_validation import iter_folds
from src.models.base import feature_columns
from src.models.lightgbm_model import DEFAULT_PARAMS
from src.models.train import assert_fold_integrity, load_dataset

log = logging.getLogger(__name__)

SEARCH_FOLDS = 3          # the last three; the early folds train on comparatively little
SEARCH_FRAC = 0.35
SEARCH_ROUNDS = 600
EARLY_STOPPING = 60
TIME_BUDGET_MIN = 45.0   # wall-clock cap on the search; see search()


def suggest(trial) -> dict:
    """The search space.

    Ranges bracket the hand-chosen defaults rather than starting from scratch. Those were
    picked for defensible reasons - min_data_in_leaf guards 1.26M rows against
    over-splitting, max_bin keeps memory down - and the question is whether they can be
    improved on, not whether an unconstrained search turns up something exotic.

    THE RANGES ARE ALSO A COST DECISION, learned the hard way. The first version allowed
    num_leaves to 511, max_bin to 255, min_data_in_leaf down to 100 and learning_rate down
    to 0.01. Any one of those is affordable; together they are not, and the low end of the
    learning rate is the trap - early stopping never fires, so such a trial runs every one
    of its rounds. Trials in that corner cost roughly ten times the baseline, and a search
    budgeted at an hour ran for three.

    A search space is a compute budget written in another notation. These bounds keep the
    worst trial within about 2x the baseline instead of 10x, and TIME_BUDGET below caps the
    whole run regardless.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 50.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127]),
    }


def cv_score(
    df: pd.DataFrame,
    params: dict,
    *,
    folds: int = SEARCH_FOLDS,
    rounds: int = SEARCH_ROUNDS,
    seed: int = 42,
    trial=None,
) -> tuple[float, list[float]]:
    """Walk-forward cosine, averaged over the last `folds` folds.

    `seed` drives bagging and feature sampling, so re-scoring the same parameters with a
    different seed resamples the noise without changing anything about the protocol.
    """
    import lightgbm as lgb

    cols = feature_columns(df)
    months, y = df["month"].to_numpy(), df["target"].to_numpy()
    full = {**DEFAULT_PARAMS, **params, "seed": seed, "bagging_seed": seed,
            "feature_fraction_seed": seed}

    def cosine_eval(pred, dataset):
        return "cosine", cosine_similarity(dataset.get_label(), pred), True

    scores: list[float] = []
    for i, (fold, tr, va) in enumerate(list(iter_folds(months))[-folds:]):
        assert_fold_integrity(months, fold, tr, va)
        dtr = lgb.Dataset(df.iloc[tr][cols], label=y[tr])
        dva = lgb.Dataset(df.iloc[va][cols], label=y[va], reference=dtr)
        booster = lgb.train(
            full, dtr, num_boost_round=rounds, valid_sets=[dva], feval=cosine_eval,
            callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True,
                                          verbose=False)],
        )
        scores.append(cosine_similarity(y[va], booster.predict(df.iloc[va][cols])))
        if trial is not None:
            # Prune trials that are already hopeless so the budget goes to live ones.
            import optuna
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), scores


def search(df: pd.DataFrame, n_trials: int = 40, seed: int = 42,
           time_budget_min: float = TIME_BUDGET_MIN):
    """Run the search under a wall-clock cap, logging every trial as it lands.

    Both of those are corrections. An earlier version silenced optuna and left the budget
    open, so a run that was expected to take an hour took three with no way to see how far
    along it was. A long job you cannot observe is indistinguishable from a hung one.
    """
    import optuna

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.perf_counter()

    def objective(trial):
        return cv_score(df, suggest(trial), seed=seed, trial=trial)[0]

    def report(study_, trial):
        best = study_.best_value if study_.best_trial else float("nan")
        log.info("  trial %2d/%d  %-8s %+.5f  best %+.5f  [%.0f min]",
                 trial.number + 1, n_trials, trial.state.name,
                 trial.value if trial.value is not None else float("nan"),
                 best, (time.perf_counter() - t0) / 60)

    study.optimize(objective, n_trials=n_trials, callbacks=[report],
                   timeout=time_budget_min * 60, show_progress_bar=False)

    done = len(study.trials)
    if done < n_trials:
        log.warning("time budget of %.0f min reached after %d/%d trials - reporting what "
                    "the search found by then", time_budget_min, done, n_trials)
    log.info("search finished: %d trials (%d pruned) in %.0f min", done,
             sum(t.state.name == "PRUNED" for t in study.trials),
             (time.perf_counter() - t0) / 60)
    return study


def run(*, n_trials: int = 40, sample_frac: float = SEARCH_FRAC, seed: int = 42,
        time_budget_min: float = TIME_BUDGET_MIN) -> dict:
    cfg = load_config()
    df = load_dataset("train")
    if sample_frac and sample_frac < 1.0:
        df = df.groupby("month", group_keys=False).sample(frac=sample_frac,
                                                          random_state=seed)
        df = df.sort_values("sample_id", ignore_index=True)
    log.info("search set: %s rows (%.0f%% of train)", f"{len(df):,}", sample_frac * 100)

    base_score, base_folds = cv_score(df, {}, seed=seed)
    log.info("baseline (current defaults): %+.5f  folds %s", base_score,
             [f"{s:+.5f}" for s in base_folds])

    study = search(df, n_trials=n_trials, seed=seed,
                   time_budget_min=time_budget_min)
    best = study.best_params
    log.info("best trial CV: %+.5f  (gain %+.5f)", study.best_value,
             study.best_value - base_score)
    log.info("best params: %s", json.dumps(best, sort_keys=True))

    # How much of that gain survives a different draw of the same noise?
    check_seeds = (seed + 1, seed + 2, seed + 3)
    tuned_re = [cv_score(df, best, seed=s)[0] for s in check_seeds]
    base_re = [cv_score(df, {}, seed=s)[0] for s in check_seeds]
    honest_gain = float(np.mean(tuned_re) - np.mean(base_re))
    claimed = study.best_value - base_score

    log.info("")
    log.info("re-scored on fresh seeds %s:", list(check_seeds))
    log.info("  tuned    %+.5f  (std %.5f)", np.mean(tuned_re), np.std(tuned_re))
    log.info("  baseline %+.5f  (std %.5f)", np.mean(base_re), np.std(base_re))
    log.info("  gain claimed by the search %+.5f", claimed)
    log.info("  gain that survives         %+.5f", honest_gain)
    log.info("  selection optimism         %+.5f", claimed - honest_gain)

    out = {
        "n_trials": n_trials, "trials_completed": len(study.trials),
        "sample_frac": sample_frac, "search_folds": SEARCH_FOLDS,
        "time_budget_min": time_budget_min,
        "baseline_cv": base_score, "best_cv": study.best_value, "claimed_gain": claimed,
        "tuned_rescored_mean": float(np.mean(tuned_re)),
        "baseline_rescored_mean": float(np.mean(base_re)),
        "tuned_rescored_std": float(np.std(tuned_re)),
        "baseline_rescored_std": float(np.std(base_re)),
        "honest_gain": honest_gain, "selection_optimism": claimed - honest_gain,
        "check_seeds": list(check_seeds), "best_params": best,
    }
    dst = Path(cfg.paths.features)
    (dst / "tuning_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(dst / "tuning_trials.csv", index=False)
    return out


def confirm(best: dict | None = None, *, seed: int = 42) -> dict:
    """Does the gain transfer to full data and the real 5-fold protocol?

    Searching on 35% of the rows and three folds is a cost decision, and it is not free:
    several of these knobs scale with n - min_data_in_leaf most obviously - so an optimum
    found on a third of the data need not be one on all of it. This re-runs both arms on
    the full dataset under the protocol the reported numbers use.

    The comparison is PAIRED: identical folds, identical rows, identical seeds, only the
    parameters differ. Per-fold differences are reported alongside the mean, because a
    mean gain smaller than the fold spread is not evidence of anything.
    """
    cfg = load_config()
    if best is None:
        best = json.loads(
            (Path(cfg.paths.features) / "tuning_result.json").read_text())["best_params"]

    df = load_dataset("train")
    log.info("confirming on full data: %s rows, all folds", f"{len(df):,}")

    n_folds = len(list(iter_folds(df["month"].to_numpy())))
    tuned_mean, tuned = cv_score(df, best, folds=n_folds, rounds=3000, seed=seed)
    base_mean, base = cv_score(df, {}, folds=n_folds, rounds=3000, seed=seed)
    diffs = [t - b for t, b in zip(tuned, base)]

    log.info("  tuned    %+.5f  folds %s", tuned_mean, [f"{s:+.5f}" for s in tuned])
    log.info("  baseline %+.5f  folds %s", base_mean, [f"{s:+.5f}" for s in base])
    log.info("  paired difference %+.5f  (per fold %s)", tuned_mean - base_mean,
             [f"{d:+.5f}" for d in diffs])
    log.info("  folds improved: %d of %d", sum(d > 0 for d in diffs), len(diffs))

    out = {
        "full_data_rows": len(df), "n_folds": n_folds, "seed": seed,
        "tuned_mean": tuned_mean, "baseline_mean": base_mean,
        "paired_gain": tuned_mean - base_mean, "per_fold_diff": diffs,
        "folds_improved": int(sum(d > 0 for d in diffs)),
        "tuned_folds": tuned, "baseline_folds": base, "best_params": best,
    }
    (Path(cfg.paths.features) / "tuning_confirm.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="LightGBM hyperparameter search")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--sample-frac", type=float, default=SEARCH_FRAC)
    ap.add_argument("--budget-min", type=float, default=TIME_BUDGET_MIN,
                    help="wall-clock cap on the search, in minutes")
    ap.add_argument("--confirm", action="store_true",
                    help="skip the search; re-check the saved winner on full data")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.confirm:
        confirm()
    else:
        run(n_trials=args.trials, sample_frac=args.sample_frac,
            time_budget_min=args.budget_min)


if __name__ == "__main__":
    main()
