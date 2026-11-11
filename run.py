"""Run one research question (or all) and write its CSV results to results/rq_<N>_result/.

    python run.py --rq 0      data, folds and representation screening
    python run.py --rq 1      predictive comparability and posterior structure
    python run.py --rq 2      operational reliability
    python run.py --rq 3      construct validity of the uncertainty components
    python run.py --rq 4      controlled ECG corruption and zero-shot transfer to LTSTDB
    python run.py --rq all    everything, in order
"""
import argparse
import importlib
import time

STEPS = ["0", "1", "2", "3", "4"]


def main():
    ap = argparse.ArgumentParser(description="ECG signal-quality uncertainty study")
    ap.add_argument("--rq", required=True, choices=STEPS + ["all"])
    args = ap.parse_args()
    for rq in (STEPS if args.rq == "all" else [args.rq]):
        t0 = time.time()
        print(f"\n=== RQ{rq} ===", flush=True)
        importlib.import_module(f"ecguq.rq.rq{rq}").run()
        print(f"=== RQ{rq} done in {time.time() - t0:.0f} s ===", flush=True)


if __name__ == "__main__":
    main()
