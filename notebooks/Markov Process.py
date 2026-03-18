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
from scipy import signal

from noiselab.adev import oadev
from noiselab.generators import MarkovProcess

# %%
s = 1
Hz = 1 / s

# %% [markdown]
# # First-order Markov process

# %% [markdown]
# A (continuous) random noise process described by
# $$
# \frac{\partial}{\partial t} x = - \Gamma x + \sqrt{D} \xi(t)
# $$
# Where $\Gamma=1/\tau_c$ is given by the correlation time $\tau_c$, $D$ is the diffusion rate, and $\xi$ is a random variable Wiener increment (white noise) satisfying $\langle \xi(t) \xi(t')\rangle = \delta(t-t')$.
#
# The solution to this differential equation is given by
# $$
# x(t) = x(0)e^{-\Gamma t} + \sqrt{D} \int_0^t d \tau e^{\Gamma(\tau - t)} \xi(\tau)
# $$
#
# For a discrete process sampled at intervals $\Delta t$, with $\Gamma \ll 1/\Delta t$, this can be appoximated as
# $$
# X_{i+1} = X_i - \Gamma X_i + \sqrt{D \Delta t} \xi_i
# $$
# where $\langle \xi_i \xi_j \rangle = \delta_{ij}$

# %%
num = int(10e6)
dt_sample = 1 / (400 * Hz)
t_vec = np.arange(num) * dt_sample

diffusion_rate = 1 / np.sqrt(s)
correlation_time = 10 * s

gen = MarkovProcess(diffusion_rate=diffusion_rate, correlation_time=correlation_time)
noise = gen.sample(num=num, dt=dt_sample)

# %%
sl = slice(0, 1000)
plt.plot(t_vec[sl], noise[sl], '.')

# %% [markdown]
# ## Statistics

# %% [markdown]
# The process mean is given by
# $$
# \langle x(t) \rangle = \langle x(0) \rangle e^{-\Gamma t}
# $$
# and 2nd-moment
# $$
# \langle x^2(t) \rangle = \langle x^2(0) \rangle e^{-2 \Gamma t} + \frac{D}{2 \Gamma} \left[ e^{-2 \Gamma t} - 1 \right]
# $$
#
# In steady state ($t \gg 1/\Gamma$), the mean vanishes and the variance is $\langle x^2 \rangle \rightarrow D/2\Gamma$.
# Therefore, if the initial value is defined as a random variable with distribution $x(0) \sim N(0, D/2\Gamma)$, then the process mean and variance are constant everywhere.
#

# %%
print(f'Predicted mean: {gen.mean:.3f}')
print(f'Actual mean: {np.mean(noise):.3f}')

print(f'Predicted variance: {gen.var:.3f}')
print(f'Actual variance: {np.var(noise):.3f}')

# %%
plt.hist(noise, bins=100)

# %% [markdown]
# ## Auto-correlation

# %%
corr = signal.correlate(noise, noise, mode='same') / len(noise)

# %%
N2 = len(corr) // 2

# %%
t_corr = t_vec - t_vec[N2]

# %%
N2 = len(corr) // 2
rng = int(20 * correlation_time / dt_sample)
sl = slice(N2 - rng // 2, N2 + rng // 2)

plt.figure(figsize=(6, 3))
plt.plot(t_corr[sl], corr[sl], '.', label='Simulated')

plt.plot(t_corr[sl], gen.auto_correlation(t_corr[sl]), 'k', label='Theory')
plt.legend()
plt.ylabel(r'$R_{xx}(\tau)$')
plt.xlabel(r'$\tau$ (s)')

# %% [markdown]
# ## PSD

# %% [markdown]
# The Markov-process auto-correlation function is
# $$
# R(t, t') = \langle x(t) x(t') \rangle =
# \langle x^2(0) \rangle e^{-\Gamma (t + t')} + \frac{D}{2\Gamma} \left[e^{-\Gamma |t - t'|} - e^{-\Gamma(t + t')} \right]
# $$
# If we again draw the initial state from the equilibrium distribution, then the auto-correlation function is stationary everywhere $R(t, t+\tau) = R(0, \tau)$
# $$
# R(\tau) = \frac{D}{2\Gamma} e^{-\Gamma |\tau|}
# $$
# The (single-sided) PSD can be determined from the Fourier transform of this auto-correlation function
# $$
# S_{xx}[f] = \int_{-\infty}^\infty d\tau e^{2 \pi f \tau} R(\tau) = \frac{D}{4 \pi f^2 + \Gamma^2}
# $$

# %%
fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(8, 3))

nperseg = 2**18
f_psd, P_noise = signal.welch(noise, fs=1 / dt_sample, nperseg=nperseg)
ax1.loglog(f_psd, np.sqrt(P_noise), label='Simulated')

P_theory = 2 * gen.psd(f_psd)  # Convert to single-sided PSD
ax1.loglog(f_psd, np.sqrt(P_theory), lw=1, ls='-', c='r', label='Theory')

ax1.legend()
ax1.set_ylabel('Noise PSD (x/rt. Hz)')
ax1.set_xlabel('Frequency (Hz)')

# %% [markdown]
# # ADEV

# %%
taus, ad, ns = oadev(noise, dt=dt_sample, taus='octave2', data_type='averaged')

# %% [markdown]
# In the high-frequency limit, the Markov process PSD looks like $1/f^2$ noise, which produces a random-walk like ADEV
# $$
# \sigma_x^2(\tau) \approx \frac{D \tau}{3}, \qquad \tau \Gamma \ll 1
# $$
#
# In the low-frequency limit, the Markov process PSD looks like white-noise with PSD $S_xx(0) = 2 D / \Gamma^2$, which produces an ADEV
# $$
# \sigma_x^2(\tau) \approx \frac{D}{\Gamma^2 \tau}, \qquad \tau \Gamma \gg 1
# $$

# %%
lf_avar = gen.diffusion_rate * gen.correlation_time**2 / taus
plt.loglog(taus, np.sqrt(lf_avar), '--')

hf_avar = gen.diffusion_rate / 3 * taus
plt.loglog(taus, np.sqrt(hf_avar), '--')

gen_avar = gen.avar(taus)
# x = taus / gen.correlation_time
# gen_avar = gen.diffusion_rate * gen.correlation_time / (2 * x**2) * (4*np.exp(-x) - np.exp(-2*x) - 3 + 2*x)
plt.loglog(taus, np.sqrt(gen_avar), '-', label='Theory')

plt.loglog(taus, ad, '.k', label='Simulated')

plt.ylabel(r'$\sigma_x(\tau)$')
plt.xlabel(r'$\tau$ (s)')

# %%
