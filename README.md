# LATE with unobserved treatment receipt

This technical note studies delivery-window measurements as imperfect proxies for treatment receipt in randomized experiments with **one-sided noncompliance**. A known mechanical delivery signature may help measure receipt when ordinary delivery records are unavailable.

If the detector's sensitivity is `s`, its false-positive rate is `f`, and the fraction flagged among assigned units is `q`, then

```text
compliance = (q - f) / (s - f)
LATE = assignment effect / compliance
```

The compliance formula is the established [Rogan–Gladen misclassification correction](https://doi.org/10.1093/oxfordjournals.aje.a112510). Its use requires known detection rates or suitable validation data, together with random assignment, exclusion, consistency, no interference, and a positive compliance fraction. The note does not claim a new identification result, a test of monotonicity, or empirical validation of the delivery-window idea.

[Read the paper](note/inferred_compliance_note.pdf).

The estimator retains the covariance between outcome and receipt-flag estimates and, for an independent validation sample, uncertainty in sensitivity and false positives. Separate delivery and outcome windows do not imply independence. Negative outcome changes do not identify IV defiers. The note also gives population sensitivity bounds and explains why comparisons within predicted-complier subgroups generally remain assignment effects.

## Replication

Python 3.11 or later, a LaTeX installation with `latexmk` and BibTeX, and the dependencies below are required. The recorded run environment is in [metadata.json](note/generated/metadata.json).

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
make all PYTHON=python
```

`make all` runs Black, isort, Flake8, statistical regression tests, the complete simulation, and the paper build. `make simulate` and `make paper` can be run separately. Tables and repeated numerical results are generated directly from the simulation; the paper does not maintain a second copy of those numbers.

The study compares known calibration, independent receipt validation, incorrect perfect-sensitivity assumptions, serial dependence, weak detection, and a zero treatment effect. Confidence intervals are ordinary joint delta-method Wald intervals. They are not guaranteed to work with weak detection. Infeasible estimated compliance is reported as unavailable; coverage counts unavailable intervals as failures. Bias and width are conditional on an available estimate.

- [Simulation and estimator](sim/coverage_simulation.py)
- [Generated results](note/generated/coverage.csv), including Monte Carlo uncertainty and unavailable fractions
- [Statistical tests](tests/test_inference.py)
- [Paper source](note/inferred_compliance_note.tex)

A smaller diagnostic run can use a separate output directory, preserving the paper's full-study results:

```bash
python sim/coverage_simulation.py --repetitions 100 --output /tmp/late-smoke
```

No application data or independently verified receipt records are included. Establishing an empirical contribution requires such evidence and validation that the detection rates transport to the intended experiment.
