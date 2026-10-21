"""Revalidate two fixed-budget models; scored fold labels never enter model.fit()."""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

from src.config import REPO_ROOT, load_config
from src.models.base import Model
from src.models.baseline import RidgeModel
from src.models.lightgbm_model import LightGBMModel
from src.models.run_record import fingerprint, record_run, write_record
from src.models.train import load_dataset, results_frame, run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".artifacts/revalidation.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    protocol_path = REPO_ROOT / "configs/revalidation.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cfg = load_config()
    source_path = Path(cfg.paths.features) / "dataset_train.parquet"
    before = fingerprint(source_path)
    frame = load_dataset("train")
    if len(frame) != cfg.samples.train or frame.sample_id.duplicated().any():
        raise ValueError("Full-data revalidation requires the complete unique training table")
    factories: dict[str, Callable[[], Model]] = {
        "lightgbm": lambda: LightGBMModel(params={"num_threads": protocol["lightgbm_threads"]},
                                           num_boost_round=protocol["lightgbm_rounds"]),
        "ridge": lambda: RidgeModel(alpha=protocol["ridge_alpha"]),
    }
    summary = run_walk_forward(factories, frame, log_mlflow=False, fit_validation=False)
    if fingerprint(source_path) != before:
        raise RuntimeError("Input changed during revalidation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    results_frame(summary).to_csv(args.output.with_suffix(".csv"), index=False)
    record = record_run(frame=frame, source=before, config=dict(cfg.validation), options=protocol,
                        models={"lightgbm": {**LightGBMModel().params, "num_threads": protocol["lightgbm_threads"],
                                             "num_boost_round": protocol["lightgbm_rounds"]},
                                "ridge": {"alpha": protocol["ridge_alpha"]}},
                        summary_path=args.output, root=REPO_ROOT)
    record["protocol_source"] = fingerprint(protocol_path)
    write_record(args.output.with_suffix(".run.json"), record)
    print(results_frame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
