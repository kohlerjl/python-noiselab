import numpy as np
import pytest

from noiselab.generators import MarkovProcess
from noiselab.adev import oavar


@pytest.fixture(scope="module")
def generator():
    diffusion_rate = 0.1
    correlation_time = 10.0

    return MarkovProcess(diffusion_rate=diffusion_rate, correlation_time=correlation_time)


def test_adev_spacing(generator):
    noise = generator.sample(num=100, dt=1)

    def _test(noise, taus, data_type):  # noqa
        taus, avar, num_terms = oavar(noise, dt=1, taus=taus, data_type=data_type, return_num=True)
        assert all(num_terms > 1)

    for data_type in ('averaged', 'integrated'):
        for taus in ('octave', 'octave2', 'octave3', 'decade', 'decade1', 'decade2', 'decade3', 'decade4'):
            for num in range(len(noise)+1):
                _test(noise[:num], taus=taus, data_type=data_type)


def test_adev_large(generator):
    """
    Test for errors due to numerical overflow in case of integer overflow due to improper type conversions
    """
    noise = generator.sample(num=10_000_000, dt=1)

    def _test(noise, taus, data_type):  # noqa
        taus, avar = oavar(noise, dt=1, taus=taus, data_type=data_type)
        # If data type conversions aren't done properly, avar can end up negative!!
        assert all(np.isfinite(avar))
        assert all(avar > 0)

    for data_type in ('averaged', 'integrated'):
        for taus in ('octave', 'decade'):
            _test(noise, taus=taus, data_type=data_type)


def test_adev_large_mean(generator):
    """
    Test for loss of precision due to large constant offset on data
    """
    noise = generator.sample(num=1_000_000, dt=1)

    def _test_offset(x, offset, taus, data_type):  # noqa
        taus, avar, num_terms = oavar(x, dt=1, taus=taus, data_type=data_type, return_num=True)
        taus2, avar2, num_terms2 = oavar(x + offset, dt=1, taus=taus, data_type=data_type, return_num=True)

        assert np.all(np.isclose(taus, taus2))
        assert np.all(np.isclose(num_terms, num_terms2))
        assert np.all(np.isclose(avar, avar2))

    for data_type in ('averaged', 'integrated'):
        for spacing in ('octave', 'decade'):
            _test_offset(noise, offset=2**32, taus=spacing, data_type=data_type)
