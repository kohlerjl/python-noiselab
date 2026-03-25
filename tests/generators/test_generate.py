import numpy as np
import pytest

from noiselab.generators import GeneratorBase, MarkovProcess, RelaxationProcess, WhitenoiseProcess, WienerProcess

rng = np.random.default_rng(seed=42)

GENERATORS_1D = {
    'white': WhitenoiseProcess(psd=1.0, mean=0.0, shape=1, rng=rng),
    'wiener': WienerProcess(diffusion_rate=1.0, mean=0.0, init_var=1.0, shape=1, rng=rng),
    'markov': MarkovProcess(diffusion_rate=1.0, var=1.0, mean=0.0, shape=1, rng=rng),
    'relaxation': RelaxationProcess(alpha=1.0, f_min=1e-3, f_max=1e3, mean=0.0, var=1.0, rng=rng),
}


@pytest.mark.parametrize('generator', GENERATORS_1D.values(), ids=GENERATORS_1D.keys())
def test_generate_1d(generator: GeneratorBase) -> None:
    generator.reset(seed=42)

    data = generator.sample(1000, dt=0.01)

    assert data.shape == (1000,)


GENERATORS_2D = {
    'white': WhitenoiseProcess(psd=1.0, mean=0.0, shape=10, rng=rng),
    # 'wiener': WienerProcess(diffusion_rate=1.0, mean=0.0, init_var=1.0, shape=10, rng=rng),
    # 'markov': MarkovProcess(diffusion_rate=1.0, var=1.0, mean=0.0, shape=10, rng=rng),
    # 'relaxation': RelaxationProcess(alpha=1.0, f_min=1e-3, f_max=1e3, mean=0.0, var=1.0, shape=10, rng=rng),
}


@pytest.mark.parametrize('generator', GENERATORS_2D.values(), ids=GENERATORS_2D.keys())
def test_generate_2d(generator: GeneratorBase) -> None:
    generator.reset(seed=42)

    data = generator.sample(1000, dt=0.01)

    assert data.shape == (1000, 10)
