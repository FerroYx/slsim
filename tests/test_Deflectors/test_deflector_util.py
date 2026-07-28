from slsim.Deflectors import deflector_util
from slsim.Deflectors.deflector import Deflector
from astropy.cosmology import FlatLambdaCDM
import numpy.testing as npt


def test_deflector_from_table():

    mass_type = "EPL"
    extended_source_type = "single_sersic"
    table = {
        "z": 0.2,
        "angular_size": 1,
        "n_sersic": 1,
        "e1_light": 0.1,
        "e2_light": 0.1,
        "stellar_mass": 1e11,
        "vel_disp": 250,
        "e1_mass": -0.1,
        "e2_mass": 0.1,
    }
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    deflector = deflector_util.deflector_from_table(
        table, mass_type, extended_source_type, cosmo=cosmo
    )
    assert isinstance(deflector, Deflector)


def test_light2mass():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    deflector_util.set_colossus_cosmo(cosmo=cosmo)
    kwargs_source = {"z": 0.1, "e1": 0, "e2": 0}
    halo_dict = {"richness": 100, "e1_mass": 0, "e2_mass": 0}
    with npt.assert_raises(ValueError):
        # test that NFW halo needs halo dict
        kwargs_mass = deflector_util.light2mass(
            kwargs_source,
            mass_type="NFW",
            light2mass_e_scaling=1,
            light2mass_e_scatter=0.1,
            halo_dict=None,
            m_star_v_disp_scaling=False,
            richness_fn="Abdullah2022",
        )

    with npt.assert_raises(ValueError):
        # test that NFW halo needs either halo_mass or richness as keys
        kwargs_mass = deflector_util.light2mass(
            kwargs_source,
            mass_type="NFW",
            light2mass_e_scaling=1,
            light2mass_e_scatter=0.1,
            halo_dict={},
            m_star_v_disp_scaling=False,
            richness_fn="Abdullah2022",
        )
    # get NFW halo with mass-richness relation
    kwargs_mass = deflector_util.light2mass(
        kwargs_source,
        mass_type="NFW",
        light2mass_e_scaling=1,
        light2mass_e_scatter=0.1,
        halo_dict=halo_dict,
        m_star_v_disp_scaling=False,
        richness_fn="Abdullah2022",
    )
    assert kwargs_mass["halo_mass"] > 1e14

    kwargs_mass_epl = deflector_util.light2mass(
        kwargs_source, mass_type="EPL", halo_dict={"theta_E": 1, "gamma_pl": 2}
    )
    npt.assert_almost_equal(kwargs_mass_epl["theta_E"], 1, decimal=5)
