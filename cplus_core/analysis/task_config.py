# -*- coding: utf-8 -*-
"""
    TaskConfig
"""
import typing
import enum

from ..models.base import Scenario, Activity
from ..definitions.defaults import DEFAULT_VALUES
from ..utils.conf import Settings


class TaskConfig(object):
    """Config class for Scenario Analysis."""

    # scenario data
    scenario: Scenario = None
    priority_layers: typing.List = []
    priority_layer_groups: typing.List = []
    analysis_activities: typing.List[Activity] = []
    all_activities: typing.List[Activity] = []

    # config
    snapping_enabled: bool = DEFAULT_VALUES.snapping_enabled
    snap_layer = ""
    snap_rescale: bool = DEFAULT_VALUES.snap_rescale
    snap_method = DEFAULT_VALUES.snap_method
    pathway_suitability_index = DEFAULT_VALUES.pathway_suitability_index
    carbon_coefficient = DEFAULT_VALUES.carbon_coefficient
    sieve_enabled = DEFAULT_VALUES.sieve_enabled
    sieve_threshold = DEFAULT_VALUES.sieve_threshold
    mask_path = ""
    mask_layers_paths = ""

    # output selections
    ncs_with_carbon = DEFAULT_VALUES.ncs_with_carbon
    landuse_project = DEFAULT_VALUES.landuse_project
    landuse_normalized = DEFAULT_VALUES.landuse_normalized
    landuse_weighted = DEFAULT_VALUES.landuse_weighted
    highest_position = DEFAULT_VALUES.highest_position
    base_dir = ""

    def __init__(
        self,
        scenario,
        priority_layers,
        priority_layer_groups,
        analysis_activities,
        all_activities,
        snapping_enabled=False,
        snap_layer=None,
        mask_layers_paths="",
        snap_rescale: bool = DEFAULT_VALUES.snap_rescale,
        snap_method=DEFAULT_VALUES.snap_method,
        pathway_suitability_index=DEFAULT_VALUES.pathway_suitability_index,  # noqa
        carbon_coefficient=DEFAULT_VALUES.carbon_coefficient,
        sieve_enabled=DEFAULT_VALUES.sieve_enabled,
        sieve_threshold=DEFAULT_VALUES.sieve_threshold,
        ncs_with_carbon=DEFAULT_VALUES.ncs_with_carbon,
        landuse_project=DEFAULT_VALUES.landuse_project,
        landuse_normalized=DEFAULT_VALUES.landuse_normalized,
        landuse_weighted=DEFAULT_VALUES.landuse_weighted,
        highest_position=DEFAULT_VALUES.highest_position,
        base_dir="",
    ) -> None:
        """Initialize analysis task configuration.

        :param scenario: scenario object
        :type scenario: Scenario

        :param priority_layers: list of priority layer dict
        :type priority_layers: List

        :param priority_layer_groups: List of priority layer group dict
        :type priority_layer_groups: List

        :param analysis_activities: scenario activities
        :type analysis_activities: List[Activity]

        :param all_activities: every activities from main config
        :type all_activities: List[Activity]

        :param snapping_enabled: enable snapping, defaults to False
        :type snapping_enabled: bool, optional

        :param snap_rescale: Enable snap rescale,
            defaults to DEFAULT_VALUES.snap_rescale
        :type snap_rescale: bool, optional

        :param snap_method: Snap method,
            defaults to DEFAULT_VALUES.snap_method
        :type snap_method: int, optional

        :param pathway_suitability_index: Pathway suitability index,
            defaults to DEFAULT_VALUES.pathway_suitability_index
        :type pathway_suitability_index: int, optional

        :param sieve_enabled: Enable sieve function,
            defaults to DEFAULT_VALUES.sieve_enabled
        :type sieve_enabled: bool, optional

        :param sieve_threshold: Sieve function threshold,
            defaults to DEFAULT_VALUES.sieve_threshold
        :type sieve_threshold: float, optional

        :param ncs_with_carbon: Enable output ncs with carbon,
            defaults to DEFAULT_VALUES.ncs_with_carbon
        :type ncs_with_carbon: bool, optional

        :param landuse_project: Enable output landuse project,
            defaults to DEFAULT_VALUES.landuse_project
        :type landuse_project: bool, optional

        :param landuse_normalized: Enable output landuse normalized,
            defaults to DEFAULT_VALUES.landuse_normalized
        :type landuse_normalized: bool, optional

        :param landuse_weighted: Enable output landuse weighted,
            defaults to DEFAULT_VALUES.landuse_weighted
        :type landuse_weighted: bool, optional

        :param highest_position: Enable output highest position,
            defaults to DEFAULT_VALUES.highest_position
        :type highest_position: bool, optional

        :param base_dir: base scenario directory, defaults to ""
        :type base_dir: str, optional
        """
        self.scenario = scenario
        self.priority_layers = priority_layers
        self.priority_layer_groups = priority_layer_groups
        self.analysis_activities = analysis_activities
        self.all_activities = all_activities

        self.snapping_enabled = snapping_enabled
        self.pathway_suitability_index = pathway_suitability_index
        self.carbon_coefficient = carbon_coefficient
        self.snap_rescale = snap_rescale
        self.snap_method = snap_method
        self.sieve_enabled = sieve_enabled
        self.sieve_threshold = sieve_threshold
        self.snap_layer = snap_layer

        self.mask_layers_paths = mask_layers_paths

        # output selections
        self.ncs_with_carbon = ncs_with_carbon
        self.landuse_project = landuse_project
        self.landuse_normalized = landuse_normalized
        self.landuse_weighted = landuse_weighted
        self.highest_position = highest_position

        self.base_dir = base_dir

    def get_activity(
            self, activity_uuid: str) -> typing.Union[Activity, None]:
        """Retrieve activity by uuid.

        :param activity_uuid: Activity UUID
        :type activity_uuid: str

        :return: Activity
        :rtype: typing.Union[Activity, None]
        """
        activity = None
        filtered = [
            act for act in self.all_activities if
            str(act.uuid) == activity_uuid
        ]
        if filtered:
            activity = filtered[0]
        return activity

    def get_priority_layers(self) -> typing.List:
        """Retrieve priority layer list.

        :return: Priority Layers
        :rtype: typing.List
        """
        return self.priority_layers

    def get_priority_layer(self, identifier) -> typing.Dict:
        """Retrieve priority layer by identifier.

        :param identifier: Priority layer ID
        :type identifier: str

        :return: Dictionary of priority layer
        :rtype: typing.Dict
        """
        priority_layer = None
        filtered = [f for f in self.priority_layers if f["uuid"] == str(identifier)]
        if filtered:
            priority_layer = filtered[0]
        return priority_layer

    def get_value(self, attr_name: enum.Enum, default=None):
        """Get attribute value by name.

        :param attr_name: Settings enum
        :type attr_name: enum.Enum

        :param default: Default value if not found, defaults to None
        :type default: any, optional

        :return: Attribute value
        :rtype: any
        """
        if attr_name == Settings.BASE_DIR:
            return self.base_dir
        return getattr(self, attr_name.value, default)

    def to_dict(self) -> dict:
        """Generate dictionary of TaskConfig.

        :return: Dictionary of task config
        :rtype: dict
        """
        input_dict = {
            "scenario_name": self.scenario.name,
            "scenario_desc": self.scenario.description,
            "extent": self.scenario.extent.bbox,
            "snapping_enabled": self.snapping_enabled,
            "snap_layer": self.snap_layer,
            "snap_rescale": self.snap_rescale,
            "snap_method": self.snap_method,
            "pathway_suitability_index": self.pathway_suitability_index,
            "carbon_coefficient": self.carbon_coefficient,
            "sieve_enabled": self.sieve_enabled,
            "sieve_threshold": self.sieve_threshold,
            "mask_path": self.mask_path,
            "mask_layers_paths": self.mask_layers_paths,
            "priority_layers": self.priority_layers,
            "priority_layer_groups": self.priority_layer_groups,
            "activities": [],
            "ncs_with_carbon": self.ncs_with_carbon,
            "landuse_project": self.landuse_project,
            "landuse_normalized": self.landuse_normalized,
            "landuse_weighted": self.landuse_weighted,
            "highest_position": self.highest_position,
            "base_dir": self.base_dir,
        }
        for activity in self.scenario.activities:
            activity_dict = {
                "uuid": str(activity.uuid),
                "name": activity.name,
                "description": activity.description,
                "path": activity.path,
                "layer_type": activity.layer_type,
                "user_defined": activity.user_defined,
                "pathways": [],
                "layer_styles": activity.layer_styles,
            }
            for pathway in activity.pathways:
                activity_dict["pathways"].append(
                    {
                        "uuid": str(pathway.uuid),
                        "name": pathway.name,
                        "description": pathway.description,
                        "path": pathway.path,
                        "layer_type": pathway.layer_type,
                        "priority_layers": pathway.priority_layers,
                    }
                )
            input_dict["activities"].append(activity_dict)
        return input_dict
