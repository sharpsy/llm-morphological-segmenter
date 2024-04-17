import argparse
import json

import pandas as pd


def _calc_accuracy(gold, pred):
    acc = 0
    assert len(gold) == len(
        pred
    ), f"Error! Files contain {len(gold)} gold entries and {len(pred)} predictions."
    total = len(gold)
    for g, p in zip(gold, pred):
        g_in, g_out = g.split("\t")
        g_list = g_out.split(",")
        p_in, p_out = p.split("\t")
        p_list = p_out.split(",")
        p_list = [p.strip().lower() for p in p_list]
        g_list = [g.strip().lower() for g in g_list]
        assert p_in == g_in, f"Error, {p_in} should be the same as {g_in}"
        if set(p_list).intersection(g_list):
            acc += 1
    return acc / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "goldfile", type=argparse.FileType("r"), help="gold standard analysis file"
    )
    parser.add_argument(
        "predfile", type=argparse.FileType("r"), help="predicted analysis file"
    )
    parser.add_argument(
        "output",
        type=argparse.FileType("w"),
        nargs="?",
        default="-",
        help="output file",
    )
    args = parser.parse_args()
    gold_df = pd.Series(args.goldfile.read().strip().split("\n"))
    pred_df = pd.Series(args.predfile.read().strip().split("\n"))
    acc = _calc_accuracy(gold_df, pred_df)
    output = {
        "metric": "accuracy-any",
        "files": {"reference": args.goldfile.name, "predictions": args.predfile.name},
        "score": round(acc, 4),
    }

    json.dump(output, args.output, indent=2)
