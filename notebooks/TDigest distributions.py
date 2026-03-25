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
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from noiselab.tdigest import TDigest


# %%
def plot_distribution(dist: stats.rv_continuous, *,
                      title: str | None = None,
                      size: int = 100_000,
                      compression: int = 400) -> plt.Figure:
    td = TDigest(compression=compression)
    noise = dist.rvs(size=size)
    td.extend(noise)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.set_title(title)

    x_vec = np.linspace(td.quantile(0.0001), td.quantile(0.9999), num=10_000)
    ax1.plot(x_vec, td.cdfs(x_vec), label='T-Digest')
    ax1.plot(x_vec, dist.cdf(x_vec), 'r:', lw=2, label='CDF')

    ax1.legend()
    ax1.set_ylabel('CDF')

    ax2.hist(noise, density=True, bins=100, histtype='stepfilled')
    ax2.plot(x_vec, dist.pdf(x_vec), 'r:', lw=2, label='PDF')

    cdf = td.cdfs(x_vec)
    ax2.plot(x_vec[0:-1], np.diff(cdf) / np.diff(x_vec), label='T-Digest')

    ax2.legend()
    ax2.set_ylabel('PDF')
    return fig


# %% [markdown]
# # Normal distribution

# %%
plot_distribution(stats.norm(), title='stats.norm()')

# %% [markdown]
# # Gamma distribution

# %%
plot_distribution(stats.gamma(a=10.0), title='stats.gamma(a=10.0)')

# %%
plot_distribution(stats.gamma(a=1.0), title='stats.gamma(a=1.0)')

# %% [markdown]
# # Chi-sqaured distribution

# %%
plot_distribution(stats.chi2(df=3), title='stats.chi2(df=3)')

# %% [markdown]
# # Student-T distribution

# %%
plot_distribution(stats.t(df=30), title='stats.t(df=30)')

# %% [markdown]
# # Beta distribution

# %%
plot_distribution(stats.beta(a=2.0, b=10.0), title='stats.beta(a=2.0, b=10.0)')

# %% [markdown]
# # Cosine distribution

# %%
plot_distribution(stats.cosine(), title='stats.cosine()')

# %% [markdown]
# # Laplace distribution

# %%
plot_distribution(stats.laplace(), title='stats.laplace()')
