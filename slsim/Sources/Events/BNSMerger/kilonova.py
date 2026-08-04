import numpy as np

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

from astropy import cosmology
from redback.transient_models import kilonova_models


class Kilonova:
    """Class for initializing a kilonova light curve model.

    If modeldir is provided, external kilonova model files are expected.
    This option is currently not supported. If modeldir is not provided,
    the model is retrieved from Redback's built-in kilonova models. By
    default, the MOSFiT-based kilonova model is used. Information about
    Redback can be found at
    https://redback.readthedocs.io/en/latest/.
    """

    def __init__(
        self,
        redshift,
        mej_1,
        mej_2,
        mej_3,
        vej_1,
        vej_2,
        vej_3,
        kappa_1,
        kappa_2,
        kappa_3,
        temperature_floor_1,
        temperature_floor_2,
        temperature_floor_3,
        model_name="mosfit_kilonova",
        kappa_gamma=10,
        mag_zpsys="AB",
        cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
        modeldir=None,
        **kwargs,
    ):
        """
        :param redshift: The redshift of the kilonova source.
        :type redshift: float

        :param mej_1: Ejecta mass of the first kilonova component in [M_sun].
        :type mej_1: float
        :param mej_2: Ejecta mass of the second kilonova component in [M_sun].
        :type mej_2: float
        :param mej_3: Ejecta mass of the third kilonova component in [M_sun].
        :type mej_3: float
        :param vej_1: Ejecta velocity of the first component in units of the
            speed of light [c].
        :type vej_1: float
        :param vej_2: Ejecta velocity of the second component in units of the
            speed of light [c].
        :type vej_2: float
        :param vej_3: Ejecta velocity of the third component in units of the
            speed of light [c].
        :type vej_3: float
        :param kappa_1: Opacity of the first component in [cm^2 g^-1].
        :type kappa_1: float
        :param kappa_2: Opacity of the second component in [cm^2 g^-1].
        :type kappa_2: float
        :param kappa_3: Opacity of the third component in [cm^2 g^-1].
        :type kappa_3: float
        :param temperature_floor_1: Temperature floor of the first component in [K].
        :type temperature_floor_1: float
        :param temperature_floor_2: Temperature floor of the second component in [K].
        :type temperature_floor_2: float
        :param temperature_floor_3: Temperature floor of the third component in [K].
        :type temperature_floor_3: float

        :param model_name: The kilonova light curve model to be used. If not provided,
            the default model is the MOSFiT-based kilonova model.
        :type model_name: str
        :param mag_zpsys: Optional, AB or Vega (AB default).
        :type mag_zpsys: str
        :param cosmo: Cosmology for luminosity distance calculation.
        :type cosmo: `~astropy.cosmology`
        :param modeldir: Directory including files for external kilonova models.
        :type modeldir: str or None
        :param kwargs: Additional keyword arguments passed to the Redback kilonova model.
        :type kwargs: dict
        """

        if modeldir is not None:
            # external kilonova model
            raise NotImplementedError(
                "External kilonova model files are not supported yet."
            )
        else:
            # use Redback built-in kilonova model, e.g. mosfit_kilonova
            if not hasattr(kilonova_models, model_name):
                raise ValueError(
                    f"{model_name} is not available in "
                    "redback.transient_models.kilonova_models."
                )
            else:
                self._model = getattr(kilonova_models, model_name)

        self._model_name = model_name
        self._redshift = redshift
        self._mag_zpsys = mag_zpsys
        self._cosmo = cosmo
        self._kwargs = kwargs

        self._model_parameters = {
            "mej_1": mej_1,
            "mej_2": mej_2,
            "mej_3": mej_3,
            "vej_1": vej_1,
            "vej_2": vej_2,
            "vej_3": vej_3,
            "kappa_1": kappa_1,
            "kappa_2": kappa_2,
            "kappa_3": kappa_3,
            "temperature_floor_1": temperature_floor_1,
            "temperature_floor_2": temperature_floor_2,
            "temperature_floor_3": temperature_floor_3,
            "kappa_gamma": kappa_gamma,
        }

    def get_apparent_magnitude(self, time, band, zpsys="AB"):
        """Function to return apparent magnitude of a kilonova for a given band
        and time.

        :param time: The observer-frame time array to evaluate the model
            (in days)
        :type time: array-like
        :param band: The band to evaluate the model over.
        :type band: str or list
        :param zpsys: Optional, AB or Vega (AB default)
        :type zpsys: str
        :return: magnitude of source
        """

        return self._model(
            time=time,
            redshift=self._redshift,
            bands=band,
            output_format="magnitude",
            cosmology=self._cosmo,
            **self._model_parameters,
            **self._kwargs,
        )
