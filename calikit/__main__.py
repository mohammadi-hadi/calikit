"""Command-line interface: calikit audit | fit | apply."""

from __future__ import annotations

import argparse
import json
import sys

from calikit import __version__
from calikit.binning import bin_predictions, ece, mce
from calikit.ci import bootstrap_ci
from calikit.data import _as_prob, extract, parse_rescale, read_jsonl
from calikit.fit import Mapping, fit_isotonic, fit_platt, fit_temperature
from calikit.metrics import auc, brier, decomposition, log_loss, spiegelhalter
from calikit.svg import write_svg

FITTERS = {"temperature": fit_temperature, "platt": fit_platt, "isotonic": fit_isotonic}


def _add_data_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prob-key", default="p", help="field holding the predicted probability")
    parser.add_argument("--label-key", default="y", help="field holding the 0/1 outcome")
    parser.add_argument(
        "--rescale",
        metavar="LO,HI",
        help="map scores from [LO, HI] to [0, 1] first (e.g. --rescale 1,10 for judge scores)",
    )
    parser.add_argument("--bins", type=int, default=10, help="number of confidence bins")
    parser.add_argument(
        "--scheme",
        choices=["mass", "width"],
        default="mass",
        help="equal-mass bins (default) or equal-width bins",
    )


def _load(path: str, args: argparse.Namespace) -> tuple[list[float], list[int]]:
    rescale = parse_rescale(args.rescale) if args.rescale else None
    records = read_jsonl(path)
    return extract(records, prob_key=args.prob_key, label_key=args.label_key, rescale=rescale)


def _metrics(
    probs: list[float], labels: list[int], k: int, scheme: str
) -> dict[str, float | None]:
    bins = bin_predictions(probs, labels, k=k, scheme=scheme)
    try:
        auc_value: float | None = auc(probs, labels)
    except ValueError:
        auc_value = None
    return {
        "brier": brier(probs, labels),
        "log_loss": log_loss(probs, labels),
        "auc": auc_value,
        "ece": ece(bins),
        "mce": mce(bins),
    }


def cmd_audit(args: argparse.Namespace) -> int:
    probs, labels = _load(args.file, args)
    bins = bin_predictions(probs, labels, k=args.bins, scheme=args.scheme)
    m = _metrics(probs, labels, args.bins, args.scheme)
    z, p_value = spiegelhalter(probs, labels)
    rel, res, unc = decomposition(labels, bins)
    base = sum(labels) / len(labels)
    significant = p_value < 1.0 - args.level

    def ece_stat(ps: list[float], ys: list[int]) -> float:
        return ece(bin_predictions(ps, ys, k=args.bins, scheme=args.scheme))

    brier_ci = bootstrap_ci(probs, labels, brier, level=args.level, reps=args.reps, seed=args.seed)
    ece_ci = bootstrap_ci(probs, labels, ece_stat, level=args.level, reps=args.reps, seed=args.seed)

    if args.svg:
        write_svg(args.svg, bins)

    if args.json:
        payload = {
            "n": len(probs),
            "base_rate": base,
            **m,
            "brier_ci": list(brier_ci),
            "ece_ci": list(ece_ci),
            "spiegelhalter_z": z,
            "p_value": p_value,
            "significant": significant,
            "reliability": rel,
            "resolution": res,
            "uncertainty": unc,
            "bins_k": args.bins,
            "scheme": args.scheme,
            "level": args.level,
            "seed": args.seed,
            "reps": args.reps,
            "bins": [vars(b) for b in bins],
            "svg": args.svg,
        }
        print(json.dumps(payload, indent=2))
        return 0

    pct = 100.0 * args.level
    auc_text = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a (single-class labels)"
    over = sum(1 for b in bins if b.conf > b.acc)
    print(f"n = {len(probs)}   base rate: {100 * base:.1f}%")
    print(
        f"Brier: {m['brier']:.4f}  [{brier_ci[0]:.4f}, {brier_ci[1]:.4f}]"
        f"   (always-predict-base-rate: {unc:.4f})"
    )
    print(f"log loss: {m['log_loss']:.4f}   AUC: {auc_text}")
    print(
        f"ECE ({len(bins)} {args.scheme} bins): {m['ece']:.4f}"
        f"  [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]   MCE: {m['mce']:.4f}"
    )
    print(f"over-confident in {over}/{len(bins)} bins (confidence above observed frequency)")
    print(
        f"decomposition: reliability {rel:.4f}   resolution {res:.4f}   uncertainty {unc:.4f}"
    )
    print(f"Spiegelhalter's Z: {z:.2f}   p: {p_value:.4f}")
    if significant:
        print(f"verdict: miscalibration is significant at the {pct:g}% level")
    else:
        print(f"verdict: no significant miscalibration at the {pct:g}% level")
    if args.svg:
        print(f"wrote reliability diagram to {args.svg}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    probs, labels = _load(args.file, args)
    mapping = FITTERS[args.method](probs, labels)

    if args.eval:
        eval_probs, eval_labels = _load(args.eval, args)
        held_out = True
    else:
        eval_probs, eval_labels = probs, labels
        held_out = False
    before = _metrics(eval_probs, eval_labels, args.bins, args.scheme)
    after = _metrics(mapping.apply(eval_probs), eval_labels, args.bins, args.scheme)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(mapping.as_dict(), fh, indent=2)
            fh.write("\n")

    if args.json:
        payload = {
            "method": mapping.method,
            "params": mapping.params,
            "blocks": len(mapping.blocks) if mapping.blocks is not None else None,
            "n_fit": len(probs),
            "held_out": held_out,
            "before": before,
            "after": after,
            "out": args.out,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if mapping.method == "temperature":
        t = mapping.params["T"]
        note = "over-confident" if t > 1 else "under-confident"
        print(f"temperature scaling on {len(probs)} items: T = {t:.3f}  (predictions were {note})")
    elif mapping.method == "platt":
        print(
            f"Platt scaling on {len(probs)} items: "
            f"a = {mapping.params['a']:.3f}, b = {mapping.params['b']:.3f}"
        )
    else:
        print(f"isotonic regression on {len(probs)} items: {len(mapping.blocks or [])} blocks")
    where = "held-out data" if held_out else "the fitting data"
    print(f"scores on {where}:")
    print("              before     after")
    for key, name in (("brier", "Brier"), ("log_loss", "log loss"), ("ece", "ECE")):
        print(f"{name:<12}{before[key]:>10.4f}{after[key]:>10.4f}")
    if not held_out:
        print("(in-sample numbers flatter the fit; pass --eval FILE for an honest read)")
    if args.out:
        print(f"wrote mapping to {args.out}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    with open(args.mapping, encoding="utf-8") as fh:
        try:
            mapping = Mapping.from_dict(json.load(fh))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{args.mapping}: invalid JSON ({exc.msg})") from exc
    records = read_jsonl(args.file)
    rescale = parse_rescale(args.rescale) if args.rescale else None
    out_lines = []
    for i, rec in enumerate(records, start=1):
        if args.prob_key not in rec:
            raise ValueError(f"record {i}: missing field {args.prob_key!r}")
        p = _as_prob(rec[args.prob_key], f"record {i} ({args.prob_key!r})", rescale)
        rec = dict(rec)
        rec[f"{args.prob_key}_raw"] = rec[args.prob_key]
        rec[args.prob_key] = mapping.apply_one(p)
        out_lines.append(json.dumps(rec))
    text = "\n".join(out_lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"calibrated {len(records)} records -> {args.out} (method: {mapping.method})")
    else:
        sys.stdout.write(text)
        print(
            f"calibrated {len(records)} records (method: {mapping.method})",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calikit",
        description="Calibration auditing for probabilistic predictions.",
    )
    parser.add_argument("--version", action="version", version=f"calikit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    _add_data_options(common)
    common.add_argument("--level", type=float, default=0.95, help="confidence level")
    common.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    common.add_argument("--reps", type=int, default=1000, help="bootstrap resamples")
    common.add_argument("--json", action="store_true", help="machine-readable output")

    p_audit = sub.add_parser(
        "audit", parents=[common], help="report calibration metrics for one prediction file"
    )
    p_audit.add_argument("file", help="JSONL file with predictions and outcomes")
    p_audit.add_argument("--svg", metavar="PATH", help="write a reliability diagram SVG")
    p_audit.set_defaults(fn=cmd_audit)

    p_fit = sub.add_parser(
        "fit", parents=[common], help="fit a recalibration mapping and report before/after"
    )
    p_fit.add_argument("file", help="JSONL file to fit on")
    p_fit.add_argument(
        "--method", choices=sorted(FITTERS), default="temperature", help="mapping to fit"
    )
    p_fit.add_argument("--eval", metavar="FILE", help="score before/after on this held-out file")
    p_fit.add_argument("--out", metavar="PATH", help="write the mapping as JSON")
    p_fit.set_defaults(fn=cmd_fit)

    p_apply = sub.add_parser(
        "apply", parents=[common], help="run a saved mapping over a prediction file"
    )
    p_apply.add_argument("mapping", help="mapping JSON written by `calikit fit --out`")
    p_apply.add_argument("file", help="JSONL file to calibrate")
    p_apply.add_argument("--out", metavar="PATH", help="write calibrated JSONL here")
    p_apply.set_defaults(fn=cmd_apply)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
