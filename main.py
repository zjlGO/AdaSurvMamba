from __future__ import annotations

import argparse
from timeit import default_timer as timer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AdaSurvMamba public survival training entry point.")

    parser.add_argument("--data_root_dir", type=str, required=True, help="Root directory for pre-extracted WSI features.")
    parser.add_argument("--dataset_dir", type=str, default="dataset_csv", help="Directory containing metadata CSV files.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for experiment outputs.")
    parser.add_argument("--which_splits", type=str, default="5foldcv", help="Split collection under splits/.")
    parser.add_argument("--split_dir", type=str, default="tcga_blca", help="Cancer cohort split name.")
    parser.add_argument("--model_type", type=str, default="adasurvmamba", choices=["adasurvmamba"])
    parser.add_argument("--mode", type=str, default="coattn", choices=["coattn", "pathomic", "path", "omic"])

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--k", type=int, default=5, help="Number of folds.")
    parser.add_argument("--k_start", type=int, default=-1, help="First fold to run.")
    parser.add_argument("--k_end", type=int, default=-1, help="Fold index after the last fold to run.")
    parser.add_argument("--n_bins", type=int, default=4, help="Number of discrete survival bins.")

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gc", type=int, default=32, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--reg", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--bag_loss", type=str, default="nll_surv", choices=["ce_surv", "nll_surv", "cox_surv"])
    parser.add_argument("--alpha_surv", type=float, default=0.0)

    parser.add_argument("--apply_sig", action="store_true", default=True, help="Use genomic signature groups.")
    parser.add_argument("--no_apply_sig", action="store_false", dest="apply_sig", help="Disable genomic signature groups.")
    parser.add_argument("--weighted_sample", action="store_true", default=True, help="Use weighted sampling.")
    parser.add_argument("--no_weighted_sample", action="store_false", dest="weighted_sample")
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--log_data", action="store_true", default=False, help="Write TensorBoard logs.")
    parser.add_argument("--overwrite", action="store_true", default=False)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    from utils.training import run_cross_validation

    start = timer()
    summary = run_cross_validation(args)
    elapsed = timer() - start
    print(summary)
    print(f"Finished in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
