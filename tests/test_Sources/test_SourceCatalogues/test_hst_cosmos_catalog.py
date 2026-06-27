import os
import pathlib

from astropy.cosmology import FlatLambdaCDM

from slsim.Sources.SourceCatalogues.HSTCosmosCatalog import galaxy_match


HST_COSMOS_PATH = os.path.join(
    str(pathlib.Path(__file__).parent.parent.parent),
    "TestData",
    "test_COSMOS_23.5_training_sample",
)


def test_hst_cosmos_process_catalog_keeps_noise_metadata_and_loads_image():
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    catalog = galaxy_match.process_catalog(cosmo=cosmo, catalog_path=HST_COSMOS_PATH)

    assert "NOISE_MEAN" in catalog.colnames
    assert "NOISE_VARIANCE" in catalog.colnames

    image_list, scale, phi, matched_source = galaxy_match.load_source(
        angular_size=0.3,
        physical_size=2.3,
        axis_ratio=0.7,
        sersic_angle=0.0,
        n_sersic=0.8,
        processed_catalog=catalog,
        catalog_path=HST_COSMOS_PATH,
        max_scale=3,
    )

    assert len(image_list) == 1
    assert image_list[0].ndim == 2
    assert scale > 0
    assert isinstance(phi, float)
    assert "NOISE_MEAN" in matched_source.colnames
