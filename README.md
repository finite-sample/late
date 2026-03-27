# Exploiting Delivery-Window Data to Classify Compliance Types and Estimate LATE

In randomized experiments with imperfect compliance, the LATE requires observing treatment receipt and the Wald estimator requires the monotonicity assumption. When receipt is unobserved but the experiment has a pre-treatment time series and a distinct delivery window, a break test applied to the delivery window can classify treated units into compliers, never-takers, and defiers. This yields three things the standard LATE setup does not provide:

1. An inferred compliance rate with closed-form bias correction: $\pi_c = (\hat{\pi} - \alpha/2)/(1 - \alpha/2)$
2. An empirical test of monotonicity via defier detection
3. A characterization of the complier subpopulation that can be projected onto the control group

### Paper

[Exploiting Delivery-Window Data to Classify Compliance Types and Estimate LATE](note/inferred_compliance_note.pdf)

### Replication

```bash
# Coverage simulations (Table 1)
python sim/coverage_simulation.py

# Defier detection simulations (Table 2)
python sim/defier_simulation.py
```

### Repository Structure

```
├── note/
│   ├── inferred_compliance_note.tex
│   ├── inferred_compliance_note.pdf
│   └── references.bib
├── sim/
│   ├── coverage_simulation.py
│   └── defier_simulation.py
└── README.md
```

### References

- Abadie, A., Gu, J., and Shen, S. (2024). Instrumental variable estimation with first-stage heterogeneity. *Journal of Econometrics* 240(2): 105425.
- Angrist, J.D., Imbens, G.W., and Rubin, D.B. (1996). Identification of causal effects using instrumental variables. *JASA* 91: 444-455.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal* 21: C1-C68.
- Frangakis, C.E. and Rubin, D.B. (2002). Principal stratification in causal inference. *Biometrics* 58: 21-29.
- Hazard, Y. and Löwe, S. (2023). Improving LATE estimation in experiments with imperfect compliance. Working paper.
- Imbens, G.W. and Angrist, J.D. (1994). Identification and estimation of local average treatment effects. *Econometrica* 62: 467-475.
- Nickerson, D.W. (2005). Scalable protocols offer efficient design for field experiments. *Political Analysis* 13: 233-252.

### License

MIT