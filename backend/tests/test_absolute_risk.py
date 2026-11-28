import numpy as np

from ml.absolute_risk import to_real


def test_intercept_correction_recovers_the_population_rate():
    rates = {"north-mid": {"8": 0.002}}
    # a model trained at a 50% fire share that says 50% means "average", i.e. the base rate
    p = to_real(np.array([0.5]), ["north-mid"], [8], rates, rho=0.5)
    assert abs(p[0] - 0.002) < 1e-6
    # ranking is preserved: higher model probability -> higher real probability
    hi = to_real(np.array([0.9]), ["north-mid"], [8], rates, rho=0.5)
    assert hi[0] > p[0]
