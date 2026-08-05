"""calikit: calibration auditing for probabilistic predictions."""

from calikit.binning import Bin, bin_predictions, ece, mce
from calikit.ci import bootstrap_ci
from calikit.data import extract, parse_rescale, read_jsonl
from calikit.fit import Mapping, fit_isotonic, fit_platt, fit_temperature
from calikit.metrics import auc, brier, decomposition, log_loss, normal_cdf, spiegelhalter
from calikit.svg import reliability_svg, write_svg

__version__ = "0.1.0"

__all__ = [
    "Bin",
    "Mapping",
    "auc",
    "bin_predictions",
    "bootstrap_ci",
    "brier",
    "decomposition",
    "ece",
    "extract",
    "fit_isotonic",
    "fit_platt",
    "fit_temperature",
    "log_loss",
    "mce",
    "normal_cdf",
    "parse_rescale",
    "read_jsonl",
    "reliability_svg",
    "spiegelhalter",
    "write_svg",
]
