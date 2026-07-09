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
        model_name="mosfit_kilonova",
        ejecta_mass=None,
        ejecta_velocity=None,
        opacity=None,
        temperature_floor=None,
        kappa_gamma=10,
        mag_zpsys="AB",
        cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
        modeldir=None,
        **kwargs,
    ):
        """
        :param redshift: The redshift of the kilonova source.
        :type redshift: float
        :param model_name: The kilonova light curve model to be used. If not provided,
            the default model is the MOSFiT-based kilonova model.
        :type model_name: str
        :param ejecta_mass: Ejecta masses for the kilonova components.
        :type ejecta_mass: array-like or None
        :param ejecta_velocity: Ejecta velocities for the kilonova components.
        :type ejecta_velocity: array-like or None
        :param opacity: Opacities for the kilonova components.
        :type opacity: array-like or None
        :param temperature_floor: Temperature floors for the kilonova components.
        :type temperature_floor: array-like or None
        :param kappa_gamma: Gamma-ray opacity.
        :type kappa_gamma: float
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

        parameter_groups = {
            "ejecta_mass": ejecta_mass,
            "ejecta_velocity": ejecta_velocity,
            "opacity": opacity,
            "temperature_floor": temperature_floor,
        }

        for name, values in parameter_groups.items():
            if values is None:
                raise ValueError(f"{name} must be provided.")
            if len(values) != 3:
                raise ValueError(f"{name} must have three components.")

        self._model_parameters = {
            "mej_1": ejecta_mass[0],
            "mej_2": ejecta_mass[1],
            "mej_3": ejecta_mass[2],
            "vej_1": ejecta_velocity[0],
            "vej_2": ejecta_velocity[1],
            "vej_3": ejecta_velocity[2],
            "kappa_1": opacity[0],
            "kappa_2": opacity[1],
            "kappa_3": opacity[2],
            "temperature_floor_1": temperature_floor[0],
            "temperature_floor_2": temperature_floor[1],
            "temperature_floor_3": temperature_floor[2],
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
