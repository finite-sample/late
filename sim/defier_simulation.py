import numpy as np
from scipy import stats

np.random.seed(42)

def sim_defiers(N=500, T_pre=30, T_treat=6, T_post=30, mu_base=5,
                sigma_base=3, sigma_noise=2, pi=0.75, delta=0.0,
                mechanical=17, tau=3, alpha=0.05, n_sims=1000):
    N_treat = N // 2
    n_compliers = int(N_treat * pi)
    n_defiers = int(N_treat * delta)
    # rest are never-takers
    
    results = []
    for _ in range(n_sims):
        base = np.maximum(np.random.normal(mu_base, sigma_base, N), 0.5)
        pre = np.maximum(base[:, None] + np.random.normal(0, sigma_noise, (N, T_pre)), 0)
        tw = np.maximum(base[:, None] + np.random.normal(0, sigma_noise, (N, T_treat)), 0)
        
        # Compliers: positive mechanical shift
        tw[:n_compliers] += mechanical
        # Defiers: negative mechanical shift
        tw[n_compliers:n_compliers+n_defiers] -= mechanical
        tw = np.maximum(tw, 0)
        
        # Post-treatment
        post = np.maximum(base[:, None] + np.random.normal(0, sigma_noise, (N, T_post)), 0)
        post[:n_compliers] += tau
        post[n_compliers:n_compliers+n_defiers] -= tau
        post = np.maximum(post, 0)
        
        # Two-sided classification for treated units
        pre_m = pre[:N_treat].mean(axis=1)
        pre_s = pre[:N_treat].std(axis=1, ddof=1)
        tw_m = tw[:N_treat].mean(axis=1)
        se = np.sqrt(pre_s**2/T_pre + pre_s**2/T_treat)
        se = np.maximum(se, 1e-10)
        t = (tw_m - pre_m) / se
        
        t_crit = stats.t.ppf(1 - alpha/2, df=T_pre-1)
        
        complier_hat = (t > t_crit).sum()
        defier_hat = (t < -t_crit).sum()
        
        pi_hat_c = complier_hat / N_treat
        delta_hat = defier_hat / N_treat
        
        # Monotonicity test
        z_mono = (delta_hat - alpha/2) / np.sqrt((alpha/2)*(1-alpha/2)/N_treat)
        reject_mono = z_mono > 1.645  # one-sided 5%
        
        results.append({
            'pi_hat_c': pi_hat_c,
            'delta_hat': delta_hat,
            'reject_mono': reject_mono,
        })
    
    return results

print("DEFIER DETECTION SIMULATIONS")
print(f"{'True delta':>10} {'delta_hat':>10} {'alpha/2':>8} {'Reject H0':>10} {'pi_hat_c':>10} {'pi_hat_d':>10}")
print("-" * 65)

for d in [0.00, 0.05, 0.10, 0.15]:
    r = sim_defiers(delta=d)
    dh = np.mean([x['delta_hat'] for x in r])
    ph = np.mean([x['pi_hat_c'] for x in r])
    rej = np.mean([x['reject_mono'] for x in r])
    print(f"{d:>10.2f} {dh:>10.3f} {0.025:>8.3f} {rej:>10.3f} {ph:>10.3f} {dh:>10.3f}")
