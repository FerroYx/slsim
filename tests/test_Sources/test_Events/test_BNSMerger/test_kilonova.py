import numpy as np  
from slsim.Sources.Events.BNSMerger.kilonova import Kilonova
import numpy.testing as npt
import pytest

@pytest.fixture
def Kilonova_class():
    KN = Kilonova(
        redshift=0.1,
        model_name="mosfit_kilonova",
        ejecta_mass=[0.01, 0.02, 0.03],
        ejecta_velocity=[0.1, 0.2, 0.3],
        opacity=[0.5, 3.0, 10.0],
        temperature_floor=[5000, 4000, 3000],
        mag_zpsys="AB",
        dense_resolution=50,
    )

    return KN

def test_kilonova_mag(Kilonova_class):
    time = np.array([0.5, 1.0, 2.0])
    mag = Kilonova_class.get_apparent_magnitude(time=time, band="lsstr")

    npt.assert_equal(np.shape(mag), np.shape(time))
    npt.assert_(np.all(np.isfinite(mag)))
    npt.assert_(np.all(mag > 0))


def test_kilonova_missing_parameters():
    with pytest.raises(ValueError):
        Kilonova(
            redshift=0.1,
            ejecta_velocity=[0.1, 0.2, 0.3],
            opacity=[0.5, 3.0, 10.0],
            temperature_floor=[5000, 4000, 3000],
        )


def test_kilonova_parameter_length():
    with pytest.raises(ValueError):
        Kilonova(
            redshift=0.1,
            ejecta_mass=[0.01, 0.02],
            ejecta_velocity=[0.1, 0.2, 0.3],
            opacity=[0.5, 3.0, 10.0],
            temperature_floor=[5000, 4000, 3000],
        )


def test_kilonova_invalid_model_name():
    with pytest.raises(ValueError):
        Kilonova(
            redshift=0.1,
            model_name="not_a_kilonova_model",
            ejecta_mass=[0.01, 0.02, 0.03],
            ejecta_velocity=[0.1, 0.2, 0.3],
            opacity=[0.5, 3.0, 10.0],
            temperature_floor=[5000, 4000, 3000],
        )


def test_kilonova_external_modeldir_not_supported():
    with pytest.raises(NotImplementedError):
        Kilonova(
            redshift=0.1,
            ejecta_mass=[0.01, 0.02, 0.03],
            ejecta_velocity=[0.1, 0.2, 0.3],
            opacity=[0.5, 3.0, 10.0],
            temperature_floor=[5000, 4000, 3000],
            modeldir="some/path",
        )


if __name__ == "__main__":
    pytest.main()