import pytest

import numpy as np
from scipy import stats

from noiselab.tdigest import TDigest

distributions = {
    'gamma': stats.gamma(a=10.0),
    'norm': stats.norm,
}


@pytest.fixture(scope='module', params=list(distributions.values()), ids=list(distributions.keys()))
def noise_sample(request):
    dist = request.param
    return dist.rvs(size=100_000)


def test_min_max(noise_sample):
    td = TDigest(compression=400)
    td.extend(noise_sample)

    assert td.min == np.min(noise_sample)
    assert td.max == np.max(noise_sample)


def test_append_vs_extend(noise_sample):
    td1 = TDigest(compression=400)
    td1.extend(noise_sample)

    td2 = TDigest(compression=400)
    for x in noise_sample:
        td2.add(x)

    assert td1.size == td2.size
    assert td1.centroid_count == td2.centroid_count

    means1, weights1 = td1.centroids()
    means2, weights2 = td2.centroids()

    assert np.all(means1 == means2)
    assert np.all(weights1 == weights2)


def test_quantile_vs_quantiles(noise_sample):
    td = TDigest(compression=400)
    td.extend(noise_sample)

    qs = np.linspace(0, 1, num=1000)
    xs = td.quantiles(qs)
    xs2 = np.array([td.quantile(q) for q in qs])
    assert np.all(xs == xs2)


def test_cdf_vs_cdfs(noise_sample):
    td = TDigest(compression=400)
    td.extend(noise_sample)

    qs = np.linspace(0, 1, num=1000)
    xs = td.quantiles(qs)
    xs2 = np.array([td.quantile(q) for q in qs])
    assert np.all(xs == xs2)
