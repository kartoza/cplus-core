# -*- coding: utf-8 -*-
"""
    Handles retrieval of the task config.
"""

import enum


class Settings(enum.Enum):
    """Plugin settings names"""

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
