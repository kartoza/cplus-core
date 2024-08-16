# -*- coding: utf-8 -*-
"""
    Definitions for all defaults settings
"""

DEFAULT_CRS_ID = 4326

SCENARIO_OUTPUT_FILE_NAME = "cplus_scenario_output"
SCENARIO_OUTPUT_LAYER_NAME = "scenario_result"

QGIS_GDAL_PROVIDER = "gdal"


class DEFAULT_VALUES(object):
    """Default values for analysis."""

    snapping_enabled = False
    pathway_suitability_index = 0
    carbon_coefficient = 0.0
    snap_rescale = False
    snap_method = 0
    sieve_enabled = False
    sieve_threshold = 10.0
    ncs_with_carbon = False
    landuse_project = True
    landuse_normalized = True
    landuse_weighted = True
    highest_position = True
