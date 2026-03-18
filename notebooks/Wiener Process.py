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
from noiselab.generators import WienerProcess

# %%
s = 1
Hz = 1 / s

# %% [markdown]
# # Wiener process (Brownian motion / random-walk)

# %% [markdown]
# A random walk is a first-order Markov process in the limit $\Gamma \rightarrow 0$. It is no longer a stationary process, with an infinite signal correlation time.

# %% [markdown]
# The mean is determined by the initial value
# $$
# \langle x(t) \rangle = \langle x(0) \rangle
# $$
# and the 2nd-moment (variance) grows without bound according to
# $$
# \langle x^2(t) \rangle = \langle x^2(0) \rangle + D t
# $$

# %%
num = int(10e6)
dt_sample = 1 / (400 * Hz)
t_vec = np.arange(num) * dt_sample

diffusion_rate = 1 / np.sqrt(s)

gen = WienerProcess(diffusion_rate=diffusion_rate, mean=1, init_var=1, shape=1)
noise = gen.sample(num=num, dt=dt_sample)

# %%
noise = gen.sample(num=num, dt=dt_sample)

# %%
sl = slice(0, None, 1000)
plt.plot(t_vec[sl], noise[sl, ...], '.')

# %% [markdown]
# ## Stats

# %%
plt.hist(noise, bins=100)

# %% [markdown]
# ## PSD

# %% [markdown]
# The PSD is not finite for an infinite time-series. However, for a finite time-series of length $T$ the (single-sided) PSD can be described by (for $f \gg 1/T$)
# $$
# S_{xx}[f] = \frac{D}{4 \pi f^2}
# $$

# %%
fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(8, 3))

nperseg = 2**18

f_psd, P_noise = signal.welch(noise, fs=1 / dt_sample, nperseg=nperseg, detrend='linear')
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
# The Wiener process PSD looks like $1/f^2$ noise, which produces a random-walk like ADEV
# $$
# \sigma_x^2(\tau) \approx \frac{D \tau}{3}, \qquad \tau \Gamma \ll 1
# $$

# %%
hf_avar = gen.avar(taus)
plt.loglog(taus, np.sqrt(hf_avar), '--')
plt.loglog(taus, ad, '.k')

plt.ylabel(r'$\sigma_x(\tau)$')
plt.xlabel(r'$\tau$ (s)')

# %%
