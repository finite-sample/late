"""Receipt misclassification, corrected LATE, and Monte Carlo evidence."""

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Design:
    n: int = 500
    pi: float = 0.75
    tau: float = 3.0
    dose: float = 2.0
    rho: float = 0.0
    pre: int = 30
    delivery: int = 6
    post: int = 30
    sigma: float = 2.0
    baseline_mean: float = 5.0
    baseline_sd: float = 3.0
    alpha: float = 0.05
    validation_per_class: int = 500


def windows(receipt, design, rng):
    """Gaussian daily outcomes with stationary AR(1) noise across all windows."""
    n = len(receipt)
    total = design.pre + design.delivery + design.post
    baseline = rng.normal(design.baseline_mean, design.baseline_sd, n)
    noise = rng.normal(0, design.sigma, (n, total))
    if design.rho:
        scale = np.sqrt(1 - design.rho**2)
        for day in range(1, total):
            noise[:, day] = design.rho * noise[:, day - 1] + scale * noise[:, day]
    values = baseline[:, None] + noise
    end = design.pre + design.delivery
    values[:, design.pre : end] += design.dose * receipt[:, None]
    values[:, end:] += design.tau * receipt[:, None]
    return values[:, : design.pre], values[:, design.pre : end], values[:, end:]


def receipt_flag(pre, delivery, alpha):
    """Fixed upper-tail rule; its nominal level need not equal its actual FPR."""
    se = pre.std(axis=1, ddof=1) * np.sqrt(1 / pre.shape[1] + 1 / delivery.shape[1])
    statistic = (delivery.mean(axis=1) - pre.mean(axis=1)) / se
    return (statistic > stats.t.ppf(1 - alpha, pre.shape[1] - 1)).astype(float)


def gaussian_rates(design):
    """Exact sensitivity and FPR under independent Gaussian daily noise."""
    if design.rho != 0:
        raise ValueError("Gaussian t calibration requires independent daily noise")
    nc = design.dose / (design.sigma * np.sqrt(1 / design.pre + 1 / design.delivery))
    cutoff = stats.t.ppf(1 - design.alpha, design.pre - 1)
    return float(stats.nct.sf(cutoff, design.pre - 1, nc)), design.alpha


def calibration(design, rng):
    """Independent validation samples with verified receipt, one per class."""
    m = design.validation_per_class
    receipt = np.repeat([1.0, 0.0], m)
    pre, delivery, _ = windows(receipt, design, rng)
    flags = receipt_flag(pre, delivery, design.alpha)
    sensitivity, false_positive = flags[:m].mean(), flags[m:].mean()
    variance = np.array([flags[:m].var(ddof=1) / m, flags[m:].var(ddof=1) / m])
    return sensitivity, false_positive, variance


def ratio_gradient(a, q, s, f):
    return np.array(
        [
            (s - f) / (q - f),
            -a * (s - f) / (q - f) ** 2,
            a / (q - f),
            a * (s - q) / (q - f) ** 2,
        ]
    )


def estimate_late(y1, y0, flags, s, f, calibration_variance=(0.0, 0.0)):
    """Joint delta variance for independent units and independent calibration."""
    y1, y0, flags = map(np.asarray, (y1, y0, flags))
    if len(y1) != len(flags) or min(len(y1), len(y0)) < 2:
        raise ValueError("Need aligned treated flags and at least two units per arm")
    if not (0 <= f < s <= 1):
        raise ValueError("Need 0 <= false-positive rate < sensitivity <= 1")
    a, q = y1.mean() - y0.mean(), flags.mean()
    pi = (q - f) / (s - f)
    if not 0 < pi <= 1:
        raise ValueError("Corrected compliance outside (0, 1]; no Wald interval")
    covariance = np.zeros((4, 4))
    covariance[:2, :2] = np.cov(y1, flags, ddof=1) / len(y1)
    covariance[0, 0] += y0.var(ddof=1) / len(y0)
    covariance[2, 2], covariance[3, 3] = calibration_variance
    gradient = ratio_gradient(a, q, s, f)
    variance = float(gradient @ covariance @ gradient)
    estimate = a / pi
    se = np.sqrt(max(variance, 0))
    return {
        "estimate": float(estimate),
        "se": float(se),
        "pi": float(pi),
        "lo": float(estimate - stats.norm.ppf(0.975) * se),
        "hi": float(estimate + stats.norm.ppf(0.975) * se),
        "cov_itt_flag": float(covariance[0, 1]),
    }


def sensitivity_bounds(itt, q, f, s_lower, s_upper):
    """Population bounds with known FPR and a sensitivity interval."""
    if not (0 <= f < q <= 1 and 0 <= s_lower <= s_upper <= 1):
        raise ValueError("Invalid probability restrictions")
    lower = max(s_lower, q)
    if lower > s_upper:
        raise ValueError("Sensitivity range incompatible with observed flag rate")
    effects = [itt * (s - f) / (q - f) for s in (lower, s_upper)]
    shares = [(q - f) / (s - f) for s in (s_upper, lower)]
    return tuple(shares), tuple(sorted(effects))


def run_scenario(design, mode, repetitions, seed):
    rng = np.random.default_rng(seed)
    estimates, widths, covered, covariances = [], [], [], []
    rates = []
    failures = 0
    for _ in range(repetitions):
        complier = rng.binomial(1, design.pi, design.n)
        assigned = np.zeros(design.n, dtype=bool)
        assigned[rng.choice(design.n, design.n // 2, replace=False)] = True
        receipt = assigned * complier
        pre, delivery, post = windows(receipt, design, rng)
        flags = receipt_flag(pre[assigned], delivery[assigned], design.alpha)
        if mode == "validated":
            s, f, rate_variance = calibration(design, rng)
        elif mode == "perfect sensitivity":
            s, f, rate_variance = 1.0, design.alpha, (0.0, 0.0)
        else:
            s, f = gaussian_rates(replace(design, rho=0))
            rate_variance = (0.0, 0.0)
        rates.append((s, f))
        y = post.sum(axis=1)
        try:
            result = estimate_late(
                y[assigned], y[~assigned], flags, s, f, rate_variance
            )
        except ValueError:
            failures += 1
            covered.append(False)
            continue
        estimates.append(result["estimate"])
        widths.append(result["hi"] - result["lo"])
        covered.append(result["lo"] <= design.tau * design.post <= result["hi"])
        covariances.append(result["cov_itt_flag"])
    coverage = np.mean(covered)
    return {
        "repetitions": repetitions,
        "true_late": design.tau * design.post,
        "mean": float(np.mean(estimates)) if estimates else float("nan"),
        "bias": (
            float(np.mean(estimates) - design.tau * design.post)
            if estimates
            else float("nan")
        ),
        "coverage": float(coverage),
        "coverage_mcse": float(np.sqrt(coverage * (1 - coverage) / repetitions)),
        "mean_width": float(np.mean(widths)) if widths else float("nan"),
        "unavailable": failures / repetitions,
        "mean_s": float(np.mean(rates, axis=0)[0]),
        "mean_f": float(np.mean(rates, axis=0)[1]),
        "mean_cov_itt_flag": (
            float(np.mean(covariances)) if covariances else float("nan")
        ),
    }


def run_study(repetitions=2000, seed=20261009, output=ROOT / "note" / "generated"):
    base = Design()
    scenarios = [
        ("Strong signal", replace(base, dose=5), "known"),
        ("Moderate signal", base, "known"),
        ("Moderate, validated", base, "validated"),
        ("Assume perfect detection", base, "perfect sensitivity"),
        ("Serial noise, validated", replace(base, rho=0.6), "validated"),
        ("Serial noise, iid rates", replace(base, rho=0.6), "iid rates"),
        ("Weak signal, validated", replace(base, dose=0.5), "validated"),
        ("Null effect, validated", replace(base, tau=0), "validated"),
    ]
    rows = []
    for index, (name, design, mode) in enumerate(scenarios):
        result = run_scenario(
            design, mode, repetitions, np.random.SeedSequence([seed, index])
        )
        row = {"scenario": name, "calibration": mode, **asdict(design), **result}
        rows.append(row)
        print(
            name,
            {k: round(result[k], 3) for k in ("bias", "coverage", "unavailable")},
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = []
    for row in rows:
        lines.append(
            f"{row['scenario']} & {row['true_late']:.0f} & {row['bias']:.1f} & "
            f"{row['coverage']:.3f} & {row['mean_width']:.1f} & "
            f"{row['unavailable']:.3f} \\\\"
        )
    table = (
        r"\begin{tabular}{@{}lrrrrr@{}}"
        + "\n"
        + r"\toprule"
        + "\n"
        + r"Scenario & True LATE & Bias & Coverage & Width & Unavailable\\"
        + "\n"
        + r"\midrule"
        + "\n"
        + "\n".join(lines)
        + "\n"
        + r"\bottomrule"
        + "\n"
        + r"\end{tabular}"
        + "\n"
    )
    (output / "coverage_table.tex").write_text(table)
    macros = {
        "Replications": str(repetitions),
        "UntreatedMean": f"{base.baseline_mean * base.post:.0f}",
    }
    for label, idx in [
        ("Moderate", 2),
        ("Perfect", 3),
        ("Serial", 4),
        ("WrongSerial", 5),
        ("Weak", 6),
    ]:
        macros[label + "Coverage"] = f"{rows[idx]['coverage']:.3f}"
        macros[label + "Bias"] = f"{rows[idx]['bias']:.1f}"
        macros[label + "Unavailable"] = f"{rows[idx]['unavailable']:.3f}"
    macros["ModerateSensitivity"] = f"{gaussian_rates(base)[0]:.3f}"
    (output / "results.tex").write_text(
        "\n".join(
            f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
        )
        + "\n"
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "seed": seed,
                "repetitions": repetitions,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--output", type=Path, default=ROOT / "note" / "generated")
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    run_study(args.repetitions, args.seed, args.output)
