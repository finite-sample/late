# Inferring Treatment Compliance from Delivery-Window Data

In randomized experiments with imperfect compliance, the LATE requires observing treatment receipt. When receipt is unobserved but the experiment has a pre-treatment time series and a distinct delivery window, compliance can be inferred by testing for a structural break during delivery. The false positive rate of this test is pinned down by the significance level, which makes the bias in the inferred compliance rate analytically tractable and correctable.

The corrected compliance rate is:

$$\pi_c = \frac{\hat{\pi} - \alpha}{1 - \alpha}$$

where $\hat{\pi}$ is the share of treated units that pass the break test and $\alpha$ is the test size.

### Paper

[Inferring Treatment Compliance from Delivery-Window Data](note/inferred_compliance_note.pdf) develops the estimator, derives the delta method variance, and shows in simulations that the 95% CI achieves coverage between 0.947 and 0.974 across twelve parameter configurations.

### Replication

The simulation code is in Python and requires `numpy`, `scipy`, and `pandas`.

```bash
# Coverage simulations (Table 1 in the paper)
python sim/coverage_simulation.py
```

The script runs 1,000 Monte Carlo replications at each of 12 parameter configurations and reports bias, coverage, and CI width for the corrected LATE estimator.

### Repository Structure

```
├── note/
│   ├── inferred_compliance_note.tex
│   ├── inferred_compliance_note.pdf
│   └── references.bib
├── sim/
│   └── coverage_simulation.py
└── README.md
```

### References

- Abadie, A., Gu, J., and Shen, S. (2024). Instrumental variable estimation with first-stage heterogeneity. *Journal of Econometrics* 240(2): 105425.
- Angrist, J.D., Imbens, G.W., and Rubin, D.B. (1996). Identification of causal effects using instrumental variables. *JASA* 91: 444-455.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal* 21: C1-C68.
- Hazard, Y. and Löwe, S. (2023). Improving LATE estimation in experiments with imperfect compliance. Working paper.
- Imbens, G.W. and Angrist, J.D. (1994). Identification and estimation of local average treatment effects. *Econometrica* 62: 467-475.

### License

MIT
