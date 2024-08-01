# -*- coding: utf-8 -*-
"""
    Definitions for all defaults settings
"""

import os
import json

from pathlib import Path

PILOT_AREA_EXTENT = {
    "type": "Polygon",
    "coordinates": [30.743498637, 32.069186664, -25.201606226, -23.960197335],
}

DEFAULT_CRS_ID = 4326

SCENARIO_OUTPUT_FILE_NAME = "cplus_scenario_output"
SCENARIO_OUTPUT_LAYER_NAME = "scenario_result"

QGIS_GDAL_PROVIDER = "gdal"

# Initiliazing the plugin default data as found in the data directory
priority_layer_path = (
    Path(__file__).parent.parent.resolve()
    / "data"
    / "default"
    / "priority_weighting_layers.json"
)

with priority_layer_path.open("r") as fh:
    priority_layers_dict = json.load(fh)
PRIORITY_LAYERS = priority_layers_dict["layers"]


pathways_path = (
    Path(__file__).parent.parent.resolve() / "data" / "default" / "ncs_pathways.json"
)

with pathways_path.open("r") as fh:
    pathways_dict = json.load(fh)
# Path just contains the file name and is relative to {download_folder}/ncs_pathways
DEFAULT_NCS_PATHWAYS = pathways_dict["pathways"]


activities_path = (
    Path(__file__).parent.parent.resolve() / "data" / "default" / "activities.json"
)

with activities_path.open("r") as fh:
    models_dict = json.load(fh)

DEFAULT_ACTIVITIES = models_dict["activities"]


PRIORITY_GROUPS = [
    {
        "uuid": "dcfb3214-4877-441c-b3ef-8228ab6dfad3",
        "name": "Biodiversity",
        "description": "Placeholder text for bio diversity",
    },
    {
        "uuid": "8b9fb419-b6b8-40e8-9438-c82901d18cd9",
        "name": "Livelihood",
        "description": "Placeholder text for livelihood",
    },
    {
        "uuid": "21a30a80-eb49-4c5e-aff6-558123688e09",
        "name": "Climate Resilience",
        "description": "Placeholder text for climate resilience ",
    },
    {
        "uuid": "ae1791c3-93fd-4e8a-8bdf-8f5fced11ade",
        "name": "Ecological infrastructure",
        "description": "Placeholder text for ecological infrastructure",
    },
    {
        "uuid": "8cac9e25-98a8-4eae-a257-14a4ef8995d0",
        "name": "Policy",
        "description": "Placeholder text for policy",
    },
    {
        "uuid": "3a66c845-2f9b-482c-b9a9-bcfca8395ad5",
        "name": "Finance - Years Experience",
        "description": "Placeholder text for years of experience",
    },
    {
        "uuid": "c6dbfe09-b05c-4cfc-8fc0-fb63cfe0ceee",
        "name": "Finance - Market Trends",
        "description": "Placeholder text for market trends",
    },
    {
        "uuid": "3038cce0-3470-4b09-bb2a-f82071fe57fd",
        "name": "Finance - Net Present value",
        "description": "Placeholder text for net present value",
    },
    {
        "uuid": "3b2c7421-f879-48ef-a973-2aa3b1390694",
        "name": "Finance - Carbon",
        "description": "Placeholder text for finance carbon",
    },
]


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

