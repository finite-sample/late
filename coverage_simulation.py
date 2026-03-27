"""
CI coverage for corrected LATE estimator.
Two approaches: delta method and bootstrap.
"""

import numpy as np
from scipy import stats
import pandas as pd

np.random.seed(42)

def generate_data(N=500, T_pre=30, T_treat=6, T_post=30, mu_base=5,
                  sigma_base=3, sigma_noise=2, compliance_rate=0.75,
                  mechanical_per_day=17, tau=3):
    """Generate one experiment's data. Returns dict of arrays."""
    N_treat = N // 2
    n_compliers = int(N_treat * compliance_rate)
    
    base = np.maximum(np.random.normal(mu_base, sigma_base, N), 0.5)
    pre = np.maximum(base[:, None] + np.random.normal(0, sigma_noise, (N, T_pre)), 0)
    tw = np.maximum(base[:, None] + np.random.normal(0, sigma_noise, (N, T_treat)), 0)
    tw[:n_compliers] += mechanical_per_day
    post = np.maximum(base[:, None] + np.random.normal(0, sigma_noise, (N, T_post)), 0)
    post[:n_compliers] += tau
    Y = post.sum(axis=1)
    
    Z = np.zeros(N, dtype=int)
    Z[:N_treat] = 1
    
    return {'Y': Y, 'Z': Z, 'pre': pre, 'tw': tw, 'N': N, 'N_treat': N_treat,
            'T_pre': T_pre, 'T_treat': T_treat}


def infer_compliance(pre_treat, tw_treat, T_pre, T_treat, alpha=0.05):
    """Return D_hat array for treated units."""
    pre_m = pre_treat.mean(axis=1)
    pre_s = pre_treat.std(axis=1, ddof=1)
    tw_m = tw_treat.mean(axis=1)
    se = np.sqrt(pre_s**2 / T_pre + pre_s**2 / T_treat)
    se = np.maximum(se, 1e-10)
    t = (tw_m - pre_m) / se
    p = 1 - stats.t.cdf(t, df=T_pre - 1)
    return (p < alpha).astype(int)


def estimate_late(Y, Z, D_hat_treat, alpha=0.05):
    """
    Returns: (corrected_late, se_delta, ci_lo_delta, ci_hi_delta)
    using delta method.
    """
    N_treat = Z.sum()
    N_ctrl = len(Z) - N_treat
    
    Y1 = Y[Z == 1]
    Y0 = Y[Z == 0]
    
    itt = Y1.mean() - Y0.mean()
    var_itt = Y1.var(ddof=1) / N_treat + Y0.var(ddof=1) / N_ctrl
    
    pi_hat = D_hat_treat.mean()
    # pi_hat is a sample mean of binary, so var = pi_hat*(1-pi_hat)/N_treat
    var_pi_hat = pi_hat * (1 - pi_hat) / N_treat
    
    pi_c = (pi_hat - alpha) / (1 - alpha)
    
    if pi_c <= 0:
        return np.nan, np.nan, np.nan, np.nan
    
    late_c = itt / pi_c
    
    # Delta method for f(ITT, pi_hat) = (1-alpha)*ITT / (pi_hat - alpha)
    # df/dITT = (1-alpha)/(pi_hat - alpha) = 1/pi_c
    # df/dpi_hat = -(1-alpha)*ITT/(pi_hat - alpha)^2 = -late_c / (pi_hat - alpha)
    
    denom = pi_hat - alpha
    df_ditt = (1 - alpha) / denom
    df_dpi = -(1 - alpha) * itt / denom**2
    
    var_late = df_ditt**2 * var_itt + df_dpi**2 * var_pi_hat
    se_late = np.sqrt(var_late)
    
    ci_lo = late_c - 1.96 * se_late
    ci_hi = late_c + 1.96 * se_late
    
    return late_c, se_late, ci_lo, ci_hi


def bootstrap_ci(Y, Z, pre, tw, T_pre, T_treat, alpha=0.05, n_boot=500):
    """
    Nonparametric bootstrap: resample units (package-level), recompute everything.
    Returns (corrected_late, ci_lo, ci_hi).
    """
    N = len(Y)
    N_treat = Z.sum()
    
    # Point estimate
    D_hat = infer_compliance(pre[:N_treat], tw[:N_treat], T_pre, T_treat, alpha)
    pi_hat = D_hat.mean()
    pi_c = (pi_hat - alpha) / (1 - alpha)
    itt = Y[Z == 1].mean() - Y[Z == 0].mean()
    if pi_c <= 0:
        return np.nan, np.nan, np.nan
    late_c = itt / pi_c
    
    boot_lates = np.zeros(n_boot)
    treat_idx = np.where(Z == 1)[0]
    ctrl_idx = np.where(Z == 0)[0]
    
    for b in range(n_boot):
        # Resample within treatment and control
        bt = np.random.choice(N_treat, N_treat, replace=True)
        bc = np.random.choice(len(ctrl_idx), len(ctrl_idx), replace=True)
        
        Y1_b = Y[treat_idx[bt]]
        Y0_b = Y[ctrl_idx[bc]]
        pre_b = pre[treat_idx[bt]]
        tw_b = tw[treat_idx[bt]]
        
        itt_b = Y1_b.mean() - Y0_b.mean()
        D_hat_b = infer_compliance(pre_b, tw_b, T_pre, T_treat, alpha)
        pi_hat_b = D_hat_b.mean()
        pi_c_b = (pi_hat_b - alpha) / (1 - alpha)
        
        if pi_c_b > 0:
            boot_lates[b] = itt_b / pi_c_b
        else:
            boot_lates[b] = np.nan
    
    valid = boot_lates[~np.isnan(boot_lates)]
    if len(valid) < 50:
        return late_c, np.nan, np.nan
    
    # Percentile CI
    ci_lo = np.percentile(valid, 2.5)
    ci_hi = np.percentile(valid, 97.5)
    
    return late_c, ci_lo, ci_hi


def run_coverage(n_sims=1000, do_bootstrap=False, **kwargs):
    """Run coverage simulation. Returns DataFrame."""
    tau = kwargs.get('tau', 3)
    T_post = kwargs.get('T_post', 30)
    true_late = tau * T_post
    
    results = []
    for s in range(n_sims):
        data = generate_data(**kwargs)
        D_hat = infer_compliance(data['pre'][:data['N_treat']],
                                 data['tw'][:data['N_treat']],
                                 data['T_pre'], data['T_treat'])
        
        late_c, se_d, ci_lo_d, ci_hi_d = estimate_late(
            data['Y'], data['Z'], D_hat)
        
        covers_delta = (ci_lo_d <= true_late <= ci_hi_d) if not np.isnan(ci_lo_d) else np.nan
        
        row = {'late_c': late_c, 'se_delta': se_d,
               'ci_lo_delta': ci_lo_d, 'ci_hi_delta': ci_hi_d,
               'covers_delta': covers_delta}
        
        if do_bootstrap:
            _, ci_lo_b, ci_hi_b = bootstrap_ci(
                data['Y'], data['Z'], data['pre'], data['tw'],
                data['T_pre'], data['T_treat'], n_boot=300)
            covers_boot = (ci_lo_b <= true_late <= ci_hi_b) if not np.isnan(ci_lo_b) else np.nan
            row.update({'ci_lo_boot': ci_lo_b, 'ci_hi_boot': ci_hi_b,
                        'covers_boot': covers_boot})
        
        results.append(row)
    
    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("COVERAGE OF 95% CIs FOR CORRECTED LATE")
print("=" * 70)

# --- Baseline with bootstrap ---
print("\n--- BASELINE (pi=0.75, tau=3, N=500) ---")
print("Running with bootstrap (this takes a moment)...")
df = run_coverage(n_sims=500, do_bootstrap=True)
true_late = 90

print(f"\nTrue LATE: {true_late}")
print(f"Corrected LATE mean: {df['late_c'].mean():.2f} (sd={df['late_c'].std():.2f})")
print(f"\nDelta method CI:")
print(f"  Mean width: {(df['ci_hi_delta'] - df['ci_lo_delta']).mean():.2f}")
print(f"  Coverage: {df['covers_delta'].mean():.3f}")
print(f"\nBootstrap CI:")
print(f"  Mean width: {(df['ci_hi_boot'] - df['ci_lo_boot']).dropna().mean():.2f}")
print(f"  Coverage: {df['covers_boot'].dropna().mean():.3f}")

# --- Delta method coverage across scenarios (faster) ---
print("\n\n--- COVERAGE ACROSS SCENARIOS (delta method, 1000 sims each) ---")
print(f"{'Scenario':<35} {'LATE':>6} {'Bias':>6} {'Coverage':>8} {'Width':>8}")
print("-" * 68)

scenarios = [
    ("Baseline (pi=.75, tau=3)", dict(compliance_rate=0.75, tau=3)),
    ("Low compliance (pi=.50)", dict(compliance_rate=0.50, tau=3)),
    ("High compliance (pi=.90)", dict(compliance_rate=0.90, tau=3)),
    ("Small effect (tau=1)", dict(compliance_rate=0.75, tau=1)),
    ("Large effect (tau=10)", dict(compliance_rate=0.75, tau=10)),
    ("Null effect (tau=0)", dict(compliance_rate=0.75, tau=0)),
    ("Small N (N=100)", dict(compliance_rate=0.75, tau=3, N=100)),
    ("Large N (N=2000)", dict(compliance_rate=0.75, tau=3, N=2000)),
    ("High noise (sig=5)", dict(compliance_rate=0.75, tau=3, sigma_noise=5)),
    ("Low mechanical (mech=5)", dict(compliance_rate=0.75, tau=3, mechanical_per_day=5)),
    ("Short pre (T_pre=7)", dict(compliance_rate=0.75, tau=3, T_pre=7)),
    ("alpha=0.01", dict(compliance_rate=0.75, tau=3)),  # handle separately
]

for name, kw in scenarios:
    alpha = 0.05
    if 'alpha=0.01' in name:
        alpha = 0.01
    
    tau_val = kw.get('tau', 3)
    T_post_val = kw.get('T_post', 30)
    true_l = tau_val * T_post_val
    
    # Need to pass alpha through... let me handle it
    if 'alpha=0.01' in name:
        # Custom run with alpha=0.01
        n_sims_here = 1000
        covers = []
        widths = []
        lates = []
        for s in range(n_sims_here):
            data = generate_data(**{k:v for k,v in kw.items()})
            D_hat = infer_compliance(data['pre'][:data['N_treat']],
                                     data['tw'][:data['N_treat']],
                                     data['T_pre'], data['T_treat'], alpha=0.01)
            late_c, se_d, ci_lo, ci_hi = estimate_late(
                data['Y'], data['Z'], D_hat, alpha=0.01)
            if not np.isnan(ci_lo):
                covers.append(ci_lo <= true_l <= ci_hi)
                widths.append(ci_hi - ci_lo)
                lates.append(late_c)
        mean_late = np.mean(lates)
        print(f"{name:<35} {mean_late:>6.1f} {mean_late-true_l:>6.1f} "
              f"{np.mean(covers):>8.3f} {np.mean(widths):>8.1f}")
    else:
        df = run_coverage(n_sims=1000, do_bootstrap=False, **kw)
        mean_late = df['late_c'].dropna().mean()
        cov = df['covers_delta'].dropna().mean()
        wid = (df['ci_hi_delta'] - df['ci_lo_delta']).dropna().mean()
        print(f"{name:<35} {mean_late:>6.1f} {mean_late-true_l:>6.1f} "
              f"{cov:>8.3f} {wid:>8.1f}")
