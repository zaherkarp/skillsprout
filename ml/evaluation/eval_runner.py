"""CLI runner for SkillSprout offline evaluation.

Runs the evaluation pipeline for v1 (baseline), v2 (calibration), or both,
and outputs a JSON report plus a Markdown summary.

Usage
-----
    # Evaluate v1 baseline only
    python -m ml.evaluation.eval_runner --model v1

    # Evaluate v2 calibration model only
    python -m ml.evaluation.eval_runner --model v2

    # Evaluate both and compare
    python -m ml.evaluation.eval_runner --model both

    # Custom synthetic data size
    python -m ml.evaluation.eval_runner --model v1 --n-users 1000 --seed 123

    # Save reports to a specific directory
    python -m ml.evaluation.eval_runner --model both --output-dir reports/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.ml.scoring import BaselineScorer, OccupationScore, SkillGap
from app.ml.calibration import CalibrationModel, CalibrationFeatures
from app.core.config import settings

from ml.evaluation.eval_framework import (
    EvaluationFramework,
    EvaluationReport,
    ProxyLabelDefinition,
    TemporalSplitter,
)
from ml.evaluation.generate_synthetic_interactions import (
    GeneratorConfig,
    SyntheticDataGenerator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scorer adapters
# ---------------------------------------------------------------------------


def score_v1_baseline(
    interactions: pd.DataFrame,
) -> np.ndarray:
    """Produce v1 baseline scores for all interactions.

    The v1 model uses match_score directly (normalized to [0, 1]) as its
    predicted probability of a positive outcome. This is a naive but
    reasonable baseline since higher match_score should correlate with
    higher engagement.

    Args:
        interactions: DataFrame with ``match_score`` column (0-100 scale).

    Returns:
        Array of predicted probabilities in [0, 1].
    """
    scores = interactions["match_score"].values.astype(float) / 100.0
    return np.clip(scores, 0.0, 1.0)


def score_v2_calibrated(
    interactions: pd.DataFrame,
    train_df: pd.DataFrame,
) -> np.ndarray:
    """Train a v2 calibration model on train data and predict on interactions.

    The v2 model is a logistic regression that learns from user feedback to
    produce calibrated P(positive) predictions.

    Args:
        interactions: DataFrame to score (test/val set).
        train_df: Training DataFrame with labels to fit the model on.

    Returns:
        Array of predicted probabilities in [0, 1].
    """
    # Prepare training data from the labeled train split
    train_labeled = train_df[train_df["label"].notna()].copy()

    if len(train_labeled) < settings.model_training_min_samples:
        logger.warning(
            "Insufficient labeled training data for v2 (%d samples, need %d). "
            "Falling back to v1 scores.",
            len(train_labeled),
            settings.model_training_min_samples,
        )
        return score_v1_baseline(interactions)

    model = CalibrationModel(model_version="v2_calibrated_eval")

    # Build training tuples
    training_pairs: List[Tuple[CalibrationFeatures, int]] = []
    for _, row in train_labeled.iterrows():
        features = CalibrationFeatures(
            match_score=float(row["match_score"]),
            gap_severity=float(row["gap_severity"]),
            job_zone_diff=float(row.get("job_zone_diff", 0.0)),
            target_job_zone=float(row.get("target_job_zone", 0)),
            num_missing_skills=int(row.get("num_missing_skills", 0)),
            sum_missing_weights=float(row.get("sum_missing_weights", 0.0)),
            mean_rating=float(row.get("mean_rating", 0.0)),
            rating_variance=float(row.get("rating_variance", 0.0)),
            num_rated_skills=int(row.get("num_rated_skills", 0)),
            user_id=int(row["user_id"]),
            target_onet_code=str(row["target_onet_code"]),
            event_id=int(row["event_id"]),
        )
        label = int(row["label"])
        training_pairs.append((features, label))

    try:
        model.train(training_pairs, test_size=0.2)
    except ValueError as exc:
        logger.warning("v2 training failed: %s. Falling back to v1 scores.", exc)
        return score_v1_baseline(interactions)

    # Predict on evaluation set
    predictions: List[float] = []
    for _, row in interactions.iterrows():
        features = CalibrationFeatures(
            match_score=float(row["match_score"]),
            gap_severity=float(row["gap_severity"]),
            job_zone_diff=float(row.get("job_zone_diff", 0.0)),
            target_job_zone=float(row.get("target_job_zone", 0)),
            num_missing_skills=int(row.get("num_missing_skills", 0)),
            sum_missing_weights=float(row.get("sum_missing_weights", 0.0)),
            mean_rating=float(row.get("mean_rating", 0.0)),
            rating_variance=float(row.get("rating_variance", 0.0)),
            num_rated_skills=int(row.get("num_rated_skills", 0)),
            user_id=int(row["user_id"]),
            target_onet_code=str(row["target_onet_code"]),
            event_id=int(row["event_id"]),
        )
        pred = model.predict(features)
        predictions.append(pred.predicted_probability)

    return np.array(predictions)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_evaluation(
    model_name: str,
    n_users: int = 200,
    seed: int = 42,
    output_dir: str = "eval_reports",
) -> Dict[str, Any]:
    """Run the full evaluation pipeline.

    Steps:
        1. Generate synthetic interaction data.
        2. Assign proxy labels.
        3. Split data temporally.
        4. Score test set with specified model(s).
        5. Compute metrics.
        6. Write JSON report and Markdown summary.

    Args:
        model_name: One of "v1", "v2", or "both".
        n_users: Number of synthetic users.
        seed: Random seed.
        output_dir: Directory for report files.

    Returns:
        Dict mapping model version to its serialized EvaluationReport.
    """
    logger.info("=== SkillSprout Offline Evaluation ===")
    logger.info("Model: %s | Users: %d | Seed: %d", model_name, n_users, seed)

    # Step 1: Generate synthetic data
    logger.info("Step 1/5: Generating synthetic interaction data...")
    config = GeneratorConfig(n_users=n_users, seed=seed)
    generator = SyntheticDataGenerator(config)
    interactions = generator.generate()
    logger.info("  Generated %d interactions", len(interactions))

    # Step 2: Assign proxy labels
    logger.info("Step 2/5: Assigning proxy labels...")
    framework = EvaluationFramework()
    labeled = framework.assign_proxy_labels(interactions)

    n_labeled = labeled["label"].notna().sum()
    logger.info(
        "  %d / %d interactions labeled (%.1f%%)",
        n_labeled,
        len(labeled),
        100 * n_labeled / len(labeled),
    )

    # Step 3: Temporal split
    logger.info("Step 3/5: Splitting data temporally...")
    splitter = TemporalSplitter(
        train_frac=0.70, val_frac=0.15, test_frac=0.15
    )
    splits = splitter.split(labeled)

    # Filter to labeled examples only for scoring
    train_labeled = splits.train[splits.train["label"].notna()].copy()
    test_labeled = splits.test[splits.test["label"].notna()].copy()

    if len(test_labeled) < 5:
        raise ValueError(
            f"Only {len(test_labeled)} labeled test examples. Need at least 5 "
            "for meaningful metrics. Try increasing n_users."
        )

    y_true = test_labeled["label"].values.astype(float)
    buckets = test_labeled["bucket"].values
    query_groups = test_labeled["event_id"].values

    # Step 4 & 5: Score and evaluate
    results: Dict[str, Any] = {}
    models_to_eval: List[str] = []

    if model_name in ("v1", "both"):
        models_to_eval.append("v1")
    if model_name in ("v2", "both"):
        models_to_eval.append("v2")

    for m in models_to_eval:
        logger.info("Step 4/5: Scoring with model '%s'...", m)

        if m == "v1":
            y_score = score_v1_baseline(test_labeled)
            version = "v1_baseline"
        else:
            y_score = score_v2_calibrated(
                test_labeled, train_labeled
            )
            version = "v2_calibrated"

        logger.info("Step 5/5: Computing metrics for '%s'...", version)
        report = framework.compute_metrics(
            y_true=y_true,
            y_score=y_score,
            buckets=buckets,
            model_version=version,
            query_groups=query_groups,
        )

        report.metadata = {
            "n_users": n_users,
            "seed": seed,
            "train_size": len(splits.train),
            "val_size": len(splits.val),
            "test_size": len(splits.test),
            "test_labeled_size": len(test_labeled),
            "split_dates": {
                k: str(v) for k, v in splits.split_dates.items()
            },
            "evaluated_at": datetime.utcnow().isoformat(),
        }

        report_dict = EvaluationFramework.report_to_dict(report)
        report_md = EvaluationFramework.report_to_markdown(report)

        results[version] = report_dict

        # Write files
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(output_dir, f"{version}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        logger.info("  JSON report: %s", json_path)

        md_path = os.path.join(output_dir, f"{version}_{timestamp}.md")
        with open(md_path, "w") as f:
            f.write(report_md)
        logger.info("  Markdown summary: %s", md_path)

        # Print summary to stdout
        print(f"\n{'=' * 60}")
        print(report_md)
        print(f"{'=' * 60}\n")

    # Comparison summary if running both
    if model_name == "both" and "v1_baseline" in results and "v2_calibrated" in results:
        _print_comparison(results["v1_baseline"], results["v2_calibrated"])

    return results


def _print_comparison(v1: Dict[str, Any], v2: Dict[str, Any]) -> None:
    """Print a side-by-side comparison of v1 and v2 results."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: v1_baseline vs v2_calibrated")
    print("=" * 60)

    metrics = ["overall_auc_roc", "overall_auc_pr", "mrr", "expected_calibration_error"]
    labels = ["AUC-ROC", "AUC-PR", "MRR", "ECE"]

    for label, metric in zip(labels, metrics):
        val1 = v1.get(metric)
        val2 = v2.get(metric)
        val1_str = f"{val1:.4f}" if val1 is not None else "N/A"
        val2_str = f"{val2:.4f}" if val2 is not None else "N/A"

        # Determine which is better (lower ECE is better, higher is better
        # for the rest)
        if val1 is not None and val2 is not None:
            if metric == "expected_calibration_error":
                winner = "v2" if val2 < val1 else "v1"
            else:
                winner = "v2" if val2 > val1 else "v1"
            indicator = f"  <-- {winner} wins"
        else:
            indicator = ""

        print(f"  {label:>8s}:  v1={val1_str}  v2={val2_str}{indicator}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run SkillSprout offline model evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ml.evaluation.eval_runner --model v1
  python -m ml.evaluation.eval_runner --model both --n-users 500
  python -m ml.evaluation.eval_runner --model v2 --output-dir ./my_reports
        """,
    )

    parser.add_argument(
        "--model",
        choices=["v1", "v2", "both"],
        default="v1",
        help="Which model(s) to evaluate: v1 (baseline), v2 (calibration), "
        "or both for side-by-side comparison. Default: v1",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=200,
        help="Number of synthetic users to generate. Default: 200",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_reports",
        help="Directory for report output files. Default: eval_reports/",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the evaluation CLI.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        results = run_evaluation(
            model_name=args.model,
            n_users=args.n_users,
            seed=args.seed,
            output_dir=args.output_dir,
        )

        # Print final summary
        for version, report in results.items():
            auc_roc = report.get("overall_auc_roc")
            auc_pr = report.get("overall_auc_pr")
            ece = report.get("expected_calibration_error")
            print(
                f"[{version}] AUC-ROC={auc_roc}, AUC-PR={auc_pr}, ECE={ece}"
            )

        return 0

    except Exception:
        logger.exception("Evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
