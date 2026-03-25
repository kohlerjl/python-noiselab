# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: noiselab
#     language: python
#     name: noiselab
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
from numpy.random import PCG64
from scipy import signal, special, stats

from noiselab.generators import RelaxationProcess

rng = PCG64(seed=1)

# %% [raw]
# rng = None

# %%
# alpha = 1.7
alpha = 1
gen = RelaxationProcess(alpha=alpha, f_min=0.01, f_max=10.0, gaussianity=10.0, decay_max=3)

# %%
gen.t_last, gen.list_length, gen.mean_list_length, gen.max_list_length, gen.transition_rate

# %%
gen.gaussianity, gen.skewness

# %%
noise = gen.get_samples(1_000_000, dt=0.1)

# %% [raw]
# %%timeit
# noise = gen.get_samples(2**16, dt=1)

# %%
plt.plot(noise, ',')

# %%
print(f"Average: {np.mean(noise):.4f}, expected {gen.mean:.4g}")
print(f"Variance: {np.var(noise):.4f}, expected {gen.var:.4g}")
print(f"Skewness: {stats.skew(noise):.4f}, expected {gen.skewness:.4g}")

# %%
sample_size = 2**20
nperseg = sample_size >> 4
dt = 0.001

noise = gen.get_samples(sample_size, dt=dt)
f, P = signal.welch(noise, fs=1 / dt, nperseg=nperseg)

# %% [markdown]
# For a generated signal scaled to unity variance, the general form of the (double-sided) PSD is given by
# $$
# S(\omega) = \frac{1}{\pi \langle 1/\lambda \rangle \left(\lambda_\text{max}^{1-\beta} - \lambda_\text{min}^{1-\beta}\right)}
# \frac{1}{\omega^2} \left[
#     \lambda_\text{max}^{1-\beta} {}_2 F_1 \left( 1, \frac{1-\beta}{2}; \frac{3-\beta}{2}; -\frac{\lambda_\text{max}^2}{\omega^2}  \right)
#     - \lambda_\text{min}^{1-\beta} {}_2 F_1 \left( 1, \frac{1-\beta}{2}; \frac{3-\beta}{2}; -\frac{\lambda_\text{min}^2}{\omega^2}  \right)
# \right]
# $$
# in terms of the Gauss hyper-geometric function ${}_2 F_1(a_1, a_2; b_1; x)$ (https://mathworld.wolfram.com/HypergeometricFunction.html)

# %% [markdown]
# In the range $\lambda_\text{min} \ll \omega \ll \lambda_\text{max}$, this function can be approximated by a power law (double-sided) PSD $S(f) \approx A f^{-\alpha}$. The scaling constant $A$ can be determined by expanding the full PSD around $\omega = 1$ for $\lambda_\text{max} = x$, $\lambda_\text{min} = 1/x$, and  $x \rightarrow \infty$, giving
# $$
# A = (2\pi)^{-(\beta+1)} \Gamma\left(\frac{3-\beta}{2}\right) \Gamma\left(\frac{1 + \beta}{2}\right)
# \frac{2}{\langle 1/\lambda \rangle \left(\lambda_\text{max}^{1-\beta} - \lambda_\text{min}^{1-\beta}\right)}, \quad \text{for} \quad -1 < \beta < +1
# $$

# %% [markdown]
# For the case of $1/f$ noise ($\alpha=1, \beta=0$), the transition decay rates have a uniform distribution, and the PSD simplifies to
# $$
# S(\omega) = \frac{1}{\pi \langle 1/\lambda \rangle (\lambda_\text{max} - \lambda_\text{min})} \frac{1}{\omega} \left(
#     \rm{arctan} \frac{\lambda_\text{max}}{\omega} - \rm{arctan} \frac{\lambda_\text{min}}{\omega}
# \right)
# $$

# %% [markdown]
# and the power-law approximation is given with
# $$
# A = \frac{1}{2\langle 1/\lambda \rangle \left(\lambda_\text{max} - \lambda_\text{min}\right)} = \frac{1}{2} \rm{ln} \frac{\lambda_\text{max}}{\lambda_\text{min}}
# $$

# %%
plt.loglog(f, P, '.')

f_min = 1 / (nperseg * dt)
f_max = 1 / dt / 2
f_th = np.logspace(np.log10(f_min), np.log10(f_max), num=1000)

omega = 2 * np.pi * f_th
scale = gen.mean_inv_lambda / 2 * (gen.lb_max - gen.lb_min)
th = 2 / scale / omega**2 * (
        gen.lb_max * special.hyp2f1(1, (1 - gen.beta0) / 2, (3 - gen.beta0) / 2, -gen.lambda_max**2 / omega**2)
        - gen.lb_min * special.hyp2f1(1, (1 - gen.beta0) / 2, (3 - gen.beta0) / 2, -gen.lambda_min**2 / omega**2)
    )
plt.loglog(f_th, th, 'r-', label='Theory')

lin_scale = special.gamma((3 - gen.beta0) / 2) * special.gamma((1 + gen.beta0) / 2) * (2 * np.pi)**-alpha
plt.plot(f_th, 2 * lin_scale * (f_th**-alpha) / scale, 'k--')

th = 2 * gen.psd(f_th)  # Convert to one-sided
plt.loglog(f_th, th, 'k-', label='Theory')

plt.legend()
plt.axvline(gen.lambda_max / 2 / np.pi, ls=':', c='k')
plt.axvline(gen.lambda_min / 2 / np.pi, ls=':', c='k')

# %%
sample_size = 2**12
nperseg = sample_size >> 4

dts = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
for dt in dts:
    noise = gen.get_samples(sample_size, dt=dt)
    f, P = signal.welch(noise, fs=1 / dt, nperseg=nperseg, noverlap=3 * nperseg // 4)

    plt.loglog(f, P, '.', label=f'dt={dt:.1e}')

f_min = 4 / (nperseg * max(dts))
f_max = 1 / min(dts) / 2
f_th = np.logspace(np.log10(f_min), np.log10(f_max), num=1000)

scale = gen.mean_inv_lambda / 2 * (gen.lb_max - gen.lb_min)
# lin_scale = -special.gamma(-(gen.beta0+1)/2)*special.gamma((3-gen.beta0)/2)/special.gamma((1-gen.beta0)/2)**2
lin_scale = special.gamma((3 - gen.beta0) / 2) * special.gamma((1 + gen.beta0) / 2) * (2 * np.pi)**-alpha
plt.plot(f_th, 2 * lin_scale * (f_th**-alpha) / scale, 'k--')

omega = 2 * np.pi * f_th
th = 2 / scale / omega * (np.arctan(gen.lambda_max / omega) - np.arctan(gen.lambda_min / omega))
# plt.loglog(f_th, th, 'r-')

th = 2 / scale / omega**2 * (
        gen.lb_max * special.hyp2f1(1, (1 - gen.beta0) / 2, (3 - gen.beta0) / 2, -gen.lambda_max**2 / omega**2)
        - gen.lb_min * special.hyp2f1(1, (1 - gen.beta0) / 2, (3 - gen.beta0) / 2, -gen.lambda_min**2 / omega**2)
    )
plt.loglog(f_th, th, 'r-', label='Theory')

th = 2 * gen.psd(f_th)  # Convert to one-sided
plt.loglog(f_th, th, 'k-', label='Theory')

plt.axvline(gen.lambda_max / 2 / np.pi, ls=':', c='k')
plt.axvline(gen.lambda_min / 2 / np.pi, ls=':', c='k')

plt.legend()
plt.xlim(f_min, f_max)

# %% [markdown]
# Algorithm improvements:
# * Normalize amplitude for PSD=1 at 1 Hz?
# * Parametrize $\lambda_\text{min} = 2\pi / T$ by maximum coherence time $T$
# * Optimize for generating blocks of samples by generating all transitions through interval, iterating 'list' only once, and calculating contribution of each pulse to all samples, then cleaning/sorting the remaining transition list.
# * Try discrete approximation of exponential decay function? (might not be valid for very fast decay rates)

# %%
