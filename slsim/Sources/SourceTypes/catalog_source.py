import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label

from slsim.Sources.SourceTypes.double_sersic import DoubleSersic
from slsim.Sources.SourceTypes.single_sersic import SingleSersic
from slsim.Sources.SourceTypes.source_base import SourceBase
from slsim.Sources.SourceCatalogues.CosmosWebCatalog import galaxy_match as CosmosWeb
from slsim.Sources.SourceCatalogues.HSTCosmosCatalog import galaxy_match as HSTCosmos
from slsim.ImageSimulation.image_quality_lenstronomy import get_band_normalized_position
from lenstronomy.Util.param_util import ellipticity2phi_q

CATALOG_TYPES = ["HST_COSMOS, COSMOS_WEB"]


class CatalogSource(SourceBase):
    """Class to match sersic parameters to a real source in a given catalog.

    The sources in the catalog must have parameters that have been
    obtained by performing a sersic fit.
    """

    def __init__(
        self,
        angular_size,
        e1,
        e2,
        n_sersic,
        cosmo,
        catalog_type,
        catalog_path,
        max_scale=1,
        match_n_sersic=False,
        sersic_fallback=False,
        band_dependent_color_gradient=False,
        color_gradient=None,
        fallback_double_sersic_kwargs=None,
        **source_dict,
    ):
        """
        :param angular_size: half light radius of object [arcseconds]
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param n_sersic: Sersic index
        :param source_dict: Source properties. May be a dictionary or an Astropy table.
            This dict or table should contain atleast redshift, a magnitude in any band,
            sersic index, angular size in arcsec, and ellipticities e1 and e2.
            eg: {"z": 0.8, "mag_i": 22, "n_sersic": 1, "angular_size": 0.10,
            "e1": 0.002, "e2": 0.001}. One can provide magnitudes in multiple bands.
        :type source_dict: dict or astropy.table.Table
        :param cosmo: instance of astropy cosmology
        :param catalog_type: specifies which catalog to use. Curently the options are:
            1. "HST_COSMOS" - https://zenodo.org/records/3242143
            2. "COSMOS_WEB" - https://zenodo.org/records/19188494
        :type catalog_type: string
        :param catalog_path: path to the directory containing the source catalog
        :type catalog_path: string
        :param max_scale: The matched image will be scaled to have the desired angular size. Scaling up
            results in a more pixelated image. This input determines what the maximum up-scale factor is.
        :type max_scale: int or float
        :param match_n_sersic: determines whether to match based off of the sersic index as well.
            Since n_sersic is usually undefined and set to 1 in SLSim, this is set to False by default.
        :type match_n_sersic: bool
        :param sersic_fallback: If the matching process returns no matches, then fall back on a single sersic profile.
        :type sersic_fallback: bool
        :param band_dependent_color_gradient: If True, apply an opt-in radial
         colour-gradient transfer to matched HST_COSMOS images. Failed matches
         fall back to a DoubleSersic model with the same ``color_gradient``.
        :type band_dependent_color_gradient: bool
        :param color_gradient: Dictionary containing the DoubleSersic
         colour-gradient settings. ``strength`` controls the gradient and
         ``reference_band`` defaults to ``F814W`` for HST_COSMOS images.
        :type color_gradient: dict or None
        :param fallback_double_sersic_kwargs: Optional overrides for the
         DoubleSersic parameters used after a failed HST_COSMOS match.
        :type fallback_double_sersic_kwargs: dict or None
        """
        super().__init__(extended_source=True, point_source=False, **source_dict)
        self.name = "GAL"
        self._angular_size = angular_size
        self._e1, self._e2 = e1, e2
        self._phi, self._q = ellipticity2phi_q(e1=e1, e2=e2)
        self._n_sersic = n_sersic
        self._cosmo = cosmo
        self._max_scale = max_scale
        self._match_n_sersic = match_n_sersic
        self._sersic_fallback = sersic_fallback
        self._band_dependent_color_gradient = band_dependent_color_gradient
        self._color_gradient = color_gradient
        self._fallback_double_sersic_kwargs = fallback_double_sersic_kwargs
        self.source_dict = source_dict

        # Process catalog and store as class attribute
        # If multiple instances of the class are created with the same catalog type, the catalog is only processed once
        if catalog_type == "HST_COSMOS":

            self._match_source = HSTCosmos.load_source

            if not hasattr(CatalogSource, "processed_hst_cosmos_catalog"):
                CatalogSource.processed_hst_cosmos_catalog = HSTCosmos.process_catalog(
                    cosmo=cosmo, catalog_path=catalog_path
                )
            self.final_catalog = CatalogSource.processed_hst_cosmos_catalog

        elif catalog_type == "COSMOS_WEB":

            self._match_source = CosmosWeb.load_source

            if not hasattr(CatalogSource, "processed_cosmos_web_catalog"):
                CatalogSource.processed_cosmos_web_catalog = CosmosWeb.process_catalog(
                    cosmo=cosmo, catalog_path=catalog_path
                )
            self.final_catalog = CatalogSource.processed_cosmos_web_catalog
        else:
            raise ValueError(
                f"Catalog_type {catalog_type} not supported. Currently only {CATALOG_TYPES} are supported."
            )

        if self._band_dependent_color_gradient:
            if catalog_type != "HST_COSMOS":
                raise ValueError(
                    "band_dependent_color_gradient is currently supported only "
                    "for HST_COSMOS."
                )
            if not isinstance(self._color_gradient, dict):
                raise ValueError(
                    "color_gradient must be a dictionary when "
                    "band_dependent_color_gradient is enabled."
                )
            if self._fallback_double_sersic_kwargs is not None and not isinstance(
                self._fallback_double_sersic_kwargs, dict
            ):
                raise ValueError(
                    "fallback_double_sersic_kwargs must be a dictionary or None."
                )

        self._catalog_type = catalog_type
        self._catalog_path = catalog_path

    @property
    def catalog_type(self):
        """The catalog being used in this instance of the class."""
        return self._catalog_type

    @property
    def matched_source(self):
        """Row of astropy table from the catalog describing the matched source.

        The source is only matched after having called
        kwargs_extended_light() once.
        """
        if hasattr(self, "_matched_source"):
            return self._matched_source
        else:
            return None

    @property
    def matched_source_id(self):
        """ID of the matched galaxy from the corresponding catalog."""
        if hasattr(self, "_matched_source"):
            if self._catalog_type == "HST_COSMOS":
                return self._matched_source["IDENT"]

            elif self._catalog_type == "COSMOS_WEB":
                return self._matched_source["id"]
        else:
            return None

    def kwargs_extended_light(self, band=None):
        """Provides dictionary of keywords for the source light model(s).
        Keywords used are in lenstronomy conventions.

        :param band: Imaging band
        :return: dictionary of keywords for the source light model(s)
        """
        if not hasattr(self, "_image_list"):
            self._image_list, self._scale, self._phi, self._matched_source = (
                self._match_source(
                    angular_size=self.angular_size,
                    physical_size=self.physical_size(cosmo=self._cosmo),
                    axis_ratio=self._q,
                    sersic_angle=self._phi,
                    n_sersic=self._n_sersic,
                    processed_catalog=self.final_catalog,
                    catalog_path=self._catalog_path,
                    max_scale=self._max_scale,
                    match_n_sersic=self._match_n_sersic,
                )
            )
        # If matching fails, the optional chromatic mode uses DoubleSersic.
        if self._image_list is None:
            if self._band_dependent_color_gradient:
                return self._double_sersic_fallback().kwargs_extended_light(band=band)
            if self._sersic_fallback:
                if not hasattr(self, "single_sersic"):
                    self.single_sersic = SingleSersic(
                        angular_size=self.angular_size,
                        n_sersic=self._n_sersic,
                        e1=self._e1,
                        e2=self._e2,
                        **self.source_dict,
                    )
                return self.single_sersic.kwargs_extended_light(band=band)
            raise ValueError(
                "No valid matches found! Try reducing the desired angular size or increasing max_scale."
                "Alternatively, enable sersic_fallback to use a single sersic whenever the matching fails."
            )

        if self._band_dependent_color_gradient and self._image_list is not None:
            if not hasattr(self, "_chromatic_template"):
                try:
                    self._chromatic_template = self._clean_hst_template(
                        self._image_list[0]
                    )
                except ValueError:
                    self._chromatic_template = None

            if self._chromatic_template is None:
                return self._double_sersic_fallback().kwargs_extended_light(band=band)

        if band is None:
            mag_source = 1
        else:
            mag_source = self.extended_source_magnitude(band=band)
        center_source = self.extended_source_position

        image = self._image_for_band(band)

        light_model_list = ["INTERPOL"]
        kwargs_extended_source = [
            {
                "magnitude": mag_source,
                "image": image,
                "center_x": center_source[0],
                "center_y": center_source[1],
                "phi_G": self._phi,
                "scale": self._scale,
            }
        ]
        return light_model_list, kwargs_extended_source

    def _image_for_band(self, band):
        """Return the catalog image, optionally with HST chromatic morphology."""
        if self._band_dependent_color_gradient:
            image = self._chromatic_template
        else:
            image = self._select_image_from_band(band)

        if not self._band_dependent_color_gradient or band is None:
            return image

        reference_band = self._color_gradient.get("reference_band") or "F814W"
        strength = float(self._color_gradient.get("strength", 0.0))
        band_offset = get_band_normalized_position(
            band=band, reference_band=reference_band
        )
        if strength == 0 or band_offset == 0:
            return image

        y_grid, x_grid = np.indices(image.shape, dtype=float)
        center_x = (image.shape[1] - 1) / 2
        center_y = (image.shape[0] - 1) / 2
        radius = np.hypot(x_grid - center_x, y_grid - center_y)
        half_light_radius_pixels = self.angular_size / self._scale
        radius = np.maximum(radius, 0.5)
        radius_ratio = radius / max(half_light_radius_pixels, 0.5)

        # Bound the radial response so small HST pixels cannot dominate a band.
        radial_coordinate = np.tanh(np.log(radius_ratio))
        exponent = -0.5 * strength * band_offset * radial_coordinate
        chromatic_image = image * np.exp(exponent)
        original_flux = np.sum(image)
        chromatic_flux = np.sum(chromatic_image)
        if chromatic_flux != 0:
            chromatic_image *= original_flux / chromatic_flux
        return chromatic_image

    def _clean_hst_template(self, image):
        """Return a clean, non-negative HST morphology template.

        The COSMOS postage stamps include background noise and may include
        neighbouring objects. Chromatic reweighting of those pixels produces
        artificial lensed features, so this method retains only the central
        detected object with an apodized mask.
        """
        image = np.asarray(image, dtype=float)
        background, noise_rms = self._hst_background_statistics(image)
        background_subtracted = image - background
        smoothed_image = gaussian_filter(background_subtracted, sigma=1.0)
        threshold = 2.5 * noise_rms
        labeled_image, number_of_labels = label(smoothed_image > threshold)
        if number_of_labels == 0:
            raise ValueError("HST template has no detected central source.")

        source_label = self._central_source_label(labeled_image)
        if source_label is None:
            raise ValueError("HST template has no source close to its centre.")

        source_mask = labeled_image == source_label
        source_mask = binary_dilation(source_mask, iterations=2)
        if not self._has_template_margin(source_mask):
            raise ValueError("HST template source is too close to the cutout edge.")

        soft_mask = gaussian_filter(source_mask.astype(float), sigma=1.0)
        soft_mask /= np.max(soft_mask)
        clean_image = np.clip(background_subtracted, 0, None) * soft_mask
        if not np.any(clean_image > 0):
            raise ValueError("HST template has no positive source flux after cleaning.")
        return clean_image

    def _hst_background_statistics(self, image):
        """Return catalog noise statistics, with an edge-pixel fallback."""
        noise_mean = self._matched_source["NOISE_MEAN"]
        noise_variance = self._matched_source["NOISE_VARIANCE"]
        if np.isfinite(noise_mean) and np.isfinite(noise_variance) and noise_variance > 0:
            return float(noise_mean), float(np.sqrt(noise_variance))

        edge_width = max(2, min(image.shape) // 10)
        edge_pixels = np.concatenate(
            (
                image[:edge_width, :].ravel(),
                image[-edge_width:, :].ravel(),
                image[:, :edge_width].ravel(),
                image[:, -edge_width:].ravel(),
            )
        )
        background = np.median(edge_pixels)
        noise_rms = 1.4826 * np.median(np.abs(edge_pixels - background))
        return float(background), float(max(noise_rms, np.finfo(float).eps))

    def _central_source_label(self, labeled_image):
        """Select the detected segment nearest the expected cutout centre."""
        y_grid, x_grid = np.indices(labeled_image.shape)
        center_x = (labeled_image.shape[1] - 1) / 2
        center_y = (labeled_image.shape[0] - 1) / 2
        center_distance = np.hypot(x_grid - center_x, y_grid - center_y)
        valid_labels = np.unique(labeled_image[labeled_image > 0])
        if len(valid_labels) == 0:
            return None

        closest_label = min(
            valid_labels,
            key=lambda current_label: np.min(center_distance[labeled_image == current_label]),
        )
        maximum_distance = 1.5 * self.angular_size / self._scale
        if np.min(center_distance[labeled_image == closest_label]) > maximum_distance:
            return None
        return closest_label

    def _has_template_margin(self, source_mask):
        """Require one effective radius of clean cutout around the source mask."""
        y_indices, x_indices = np.nonzero(source_mask)
        margin = min(
            np.min(y_indices),
            source_mask.shape[0] - 1 - np.max(y_indices),
            np.min(x_indices),
            source_mask.shape[1] - 1 - np.max(x_indices),
        )
        return margin >= max(2, self.angular_size / self._scale)

    def _double_sersic_fallback(self):
        """Build the chromatic fallback model after a failed HST match."""
        if hasattr(self, "double_sersic"):
            return self.double_sersic

        fallback_kwargs = {
            "angular_size_0": 0.5 * self.angular_size,
            "angular_size_1": self.angular_size,
            "n_sersic_0": 4.0,
            "n_sersic_1": 1.0,
            "w0": 0.4,
            "w1": 0.6,
            "e1_1": self._e1,
            "e2_1": self._e2,
            "e1_2": self._e1,
            "e2_2": self._e2,
        }
        fallback_kwargs.update(self._fallback_double_sersic_kwargs or {})
        fallback_kwargs["color_gradient"] = self._color_gradient
        self.double_sersic = DoubleSersic(
            **fallback_kwargs,
            **self.source_dict,
        )
        return self.double_sersic

    def _select_image_from_band(self, band):
        """Selects an image based off of the input band. Only relevant for
        source catalogs that provide images for multiple bands.

        :param band: imaging band
        :type band: string
        :return: image from source catalog corresponding to specific
            band
        """

        if len(self._image_list) == 1 or band is None:
            return self._image_list[0]

        # Image_list contains images for bands [F115W, F150W, F277W, F444W]
        if self._catalog_type == "COSMOS_WEB":
            return CosmosWeb._select_image_from_band(band, self._image_list)
