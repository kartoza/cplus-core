# -*- coding: utf-8 -*-
"""
    Handles storage and retrieval of the plugin QgsSettings.
"""

import enum


class Settings(enum.Enum):
    """Plugin settings names"""

    DOWNLOAD_FOLDER = "download_folder"
    REFRESH_FREQUENCY = "refresh/period"
    REFRESH_FREQUENCY_UNIT = "refresh/unit"
    REFRESH_LAST_UPDATE = "refresh/last_update"
    REFRESH_STATE = "refresh/state"

    # Report settings
    REPORT_ORGANIZATION = "report/organization"
    REPORT_CONTACT_EMAIL = "report/email"
    REPORT_WEBSITE = "report/website"
    REPORT_CUSTOM_LOGO = "report/custom_logo"
    REPORT_CPLUS_LOGO = "report/cplus_logo"
    REPORT_CI_LOGO = "report/ci_logo"
    REPORT_LOGO_DIR = "report/logo_dir"
    REPORT_FOOTER = "report/footer"
    REPORT_DISCLAIMER = "report/disclaimer"
    REPORT_LICENSE = "report/license"
    REPORT_STAKEHOLDERS = "report/stakeholders"
    REPORT_CULTURE_POLICIES = "report/culture_policies"

    # Last selected data directory
    LAST_DATA_DIR = "last_data_dir"
    LAST_MASK_DIR = "last_mask_dir"

    # Advanced settings
    BASE_DIR = "advanced/base_dir"

    # Scenario basic details
    SCENARIO_NAME = "scenario_name"
    SCENARIO_DESCRIPTION = "scenario_description"
    SCENARIO_EXTENT = "scenario_extent"

    # Coefficient for carbon layers
    CARBON_COEFFICIENT = "carbon_coefficient"

    # Pathway suitability index value
    PATHWAY_SUITABILITY_INDEX = "pathway_suitability_index"

    # Snapping values
    SNAPPING_ENABLED = "snapping_enabled"
    SNAP_LAYER = "snap_layer"
    ALLOW_RESAMPLING = "snap_resampling"
    RESCALE_VALUES = "snap_rescale"
    RESAMPLING_METHOD = "snap_method"
    SNAP_PIXEL_VALUE = "snap_pixel_value"

    # Sieve function parameters
    SIEVE_ENABLED = "sieve_enabled"
    SIEVE_THRESHOLD = "sieve_threshold"
    SIEVE_MASK_PATH = "mask_path"

    # Mask layer
    MASK_LAYERS_PATHS = "mask_layers_paths"

    # Outputs options
    NCS_WITH_CARBON = "ncs_with_carbon"
    LANDUSE_PROJECT = "landuse_project"
    LANDUSE_NORMALIZED = "landuse_normalized"
    LANDUSE_WEIGHTED = "landuse_weighted"
    HIGHEST_POSITION = "highest_position"

    # Processing option
    PROCESSING_TYPE = "processing_type"

    # DEBUG
    DEBUG = "debug"
    DEV_MODE = "dev_mode"
    BASE_API_URL = "base_api_url"
