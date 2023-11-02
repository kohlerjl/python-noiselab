import numpy as np
import pytest

from noiselab.generators import MarkovProcess
from noiselab.adev import oavar, RealtimeADEV

num = int(2**16)
dt = 1


@pytest.fixture(scope="module")
def generator():
    diffusion_rate = 0.1
    correlation_time = 10.0

    return MarkovProcess(diffusion_rate=diffusion_rate, correlation_time=correlation_time)


@pytest.fixture(scope="module")
def noise(generator):
    return generator.sample(num, dt=dt)


def test_realtime(noise):
    for data_type in ('averaged', 'integrated'):
        for taus_str in ('octave', 'octave2', 'decade1', 'decade3'):
            taus, avar, ns = oavar(noise, dt=dt, taus=taus_str, data_type=data_type)
            rt = RealtimeADEV(dt=dt, max_tau=taus[-1], taus=taus_str, data_type=data_type)
            rt.extend(noise)

            rec = np.array(list(reversed(rt.record)))
            if data_type == 'averaged':
                rec_truth = np.cumsum(noise)[-len(rec):]
            else:
                rec_truth = noise[-len(rec):]
            assert np.all(np.isclose(rec, rec_truth))

            assert np.all(rt.taus == np.double(taus))
            assert np.all(ns == np.array(rt.counts))
            assert np.all(np.isclose(rt.avar(), avar))


def run_test(noise, chunksize, taus='decade3', data_type='averaged'):
    taus, avar, ns = oavar(noise, dt=dt, taus=taus, data_type=data_type)
    rt = RealtimeADEV(dt=dt, taus=taus, data_type=data_type)
    for chunk in noise.reshape((chunksize, -1)):
        rt.extend(chunk)

    rec = np.array(list(reversed(rt.record)))
    if data_type == 'averaged':
        rec_truth = np.cumsum(noise)[-len(rec):]
    else:
        rec_truth = noise[-len(rec):]
    assert np.all(np.isclose(rec, rec_truth))

    assert np.all(rt.taus == np.double(taus))
    assert np.all(ns == np.array(rt.counts))
    assert np.all(np.isclose(rt.avar(), avar))


def test_realtime_chunks(noise):
    for data_type in ('averaged', 'integrated'):
        for taus in ('octave', 'octave2', 'decade1', 'decade3'):
            for chunksize in [2**0, 2**1, 2**4, 201]:
                n = (len(noise) // chunksize) * chunksize
                run_test(noise[:n], chunksize=chunksize, taus=taus, data_type=data_type)
