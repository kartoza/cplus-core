# -*- coding: utf-8 -*-
"""
    TaskConfig
"""
import uuid
import typing
import enum

from ..models.base import (
    Scenario,
    Activity,
    SpatialExtent,
    LayerType,
    NcsPathway
)
from ..definitions.defaults import DEFAULT_VALUES


class TaskConfig(object):

    scenario_name = ''
    scenario_desc = ''
    scenario_uuid = uuid.uuid4()
    analysis_activities: typing.List[Activity] = []
    priority_layers: typing.List = []
    priority_layer_groups: typing.List = []
    analysis_extent: SpatialExtent = None
    snapping_enabled: bool = DEFAULT_VALUES.snapping_enabled
    snap_layer = ''
    snap_layer_uuid = ''
    pathway_suitability_index = DEFAULT_VALUES.pathway_suitability_index
    carbon_coefficient = DEFAULT_VALUES.carbon_coefficient
    snap_rescale = DEFAULT_VALUES.snap_rescale
    snap_method = DEFAULT_VALUES.snap_method
    sieve_enabled = DEFAULT_VALUES.sieve_enabled
    sieve_threshold = DEFAULT_VALUES.sieve_threshold
    sieve_mask_uuid = ''
    mask_path = ''
    mask_layers_paths = ''
    mask_layer_uuids = []
    scenario: Scenario = None
    pathway_uuid_layers = {}
    carbon_uuid_layers = {}
    priority_uuid_layers = {}
    total_input_layers = 0
    # output selections
    ncs_with_carbon = DEFAULT_VALUES.ncs_with_carbon
    landuse_project = DEFAULT_VALUES.landuse_project
    landuse_normalized = DEFAULT_VALUES.landuse_normalized
    landuse_weighted = DEFAULT_VALUES.landuse_weighted
    highest_position = DEFAULT_VALUES.highest_position

    def __init__(self, scenario_name, scenario_desc, extent,
                 analysis_activities, priority_layers,
                 priority_layer_groups,
                 snapping_enabled=False, snap_layer_uuid='',
                 pathway_suitability_index=DEFAULT_VALUES.pathway_suitability_index,  # noqa
                 carbon_coefficient=DEFAULT_VALUES.carbon_coefficient,
                 snap_rescale=DEFAULT_VALUES.snap_rescale,
                 snap_method=DEFAULT_VALUES.snap_method,
                 sieve_enabled=DEFAULT_VALUES.sieve_enabled,
                 sieve_threshold=DEFAULT_VALUES.sieve_threshold,
                 sieve_mask_uuid='',
                 mask_layer_uuids='', scenario_uuid=None,
                 ncs_with_carbon=DEFAULT_VALUES.ncs_with_carbon,
                 landuse_project=DEFAULT_VALUES.landuse_project,
                 landuse_normalized=DEFAULT_VALUES.landuse_normalized,
                 landuse_weighted=DEFAULT_VALUES.landuse_weighted,
                 highest_position=DEFAULT_VALUES.highest_position) -> None:
        self.scenario_name = scenario_name
        self.scenario_desc = scenario_desc
        if scenario_uuid:
            self.scenario_uuid = uuid.UUID(scenario_uuid)
        self.analysis_extent = SpatialExtent(bbox=extent)
        self.analysis_activities = analysis_activities
        self.priority_layers = priority_layers
        self.priority_layer_groups = priority_layer_groups
        self.snapping_enabled = snapping_enabled
        self.snap_layer_uuid = snap_layer_uuid
        self.pathway_suitability_index = pathway_suitability_index
        self.carbon_coefficient = carbon_coefficient
        self.snap_rescale = snap_rescale
        self.snap_method = snap_method
        self.sieve_enabled = sieve_enabled
        self.sieve_threshold = sieve_threshold
        self.sieve_mask_uuid = sieve_mask_uuid
        self.mask_layer_uuids = mask_layer_uuids
        self.scenario = Scenario(
            uuid=self.scenario_uuid,
            name=self.scenario_name,
            description=self.scenario_desc,
            extent=self.analysis_extent,
            activities=self.analysis_activities,
            weighted_activities=[],
            priority_layer_groups=self.priority_layer_groups
        )
        # output selections
        self.ncs_with_carbon = ncs_with_carbon
        self.landuse_project = landuse_project
        self.landuse_normalized = landuse_normalized
        self.landuse_weighted = landuse_weighted
        self.highest_position = highest_position

    def get_activity(
        self, activity_uuid: str
    ) -> typing.Union[Activity, None]:
        activity = None
        filtered = [
            act for act in self.analysis_activities if
            str(act.uuid) == activity_uuid
        ]
        if filtered:
            activity = filtered[0]
        return activity

    def get_priority_layers(self) -> typing.List:
        return self.priority_layers

    def get_priority_layer(self, identifier) -> typing.Dict:
        priority_layer = None
        filtered = [
            f for f in self.priority_layers if f['uuid'] == str(identifier)]
        if filtered:
            priority_layer = filtered[0]
        return priority_layer

    def get_value(self, attr_name: enum.Enum, default=None):
        return getattr(self, attr_name.value, default)

    def to_dict(self):
        input_dict = {
            'scenario_name': self.scenario.name,
            'scenario_desc': self.scenario.description,
            'extent': self.analysis_extent.bbox,
            'snapping_enabled': self.snapping_enabled,
            'snap_layer': self.snap_layer_uuid,
            'pathway_suitability_index': self.pathway_suitability_index,
            'carbon_coefficient': self.carbon_coefficient,
            'snap_rescale': self.snap_rescale,
            'snap_method': self.snap_method,
            'sieve_enabled': self.sieve_enabled,
            'sieve_threshold': self.sieve_threshold,
            'sieve_mask_uuid': self.sieve_mask_uuid,
            'mask_layer_uuids': self.mask_layer_uuids,
            'priority_layers': self.priority_layers,
            'priority_layer_groups': self.priority_layer_groups,
            'activities': [],
            'pathway_uuid_layers': self.pathway_uuid_layers,
            'carbon_uuid_layers': self.carbon_uuid_layers,
            'priority_uuid_layers': self.priority_uuid_layers,
            'total_input_layers': self.total_input_layers,
            'ncs_with_carbon': self.ncs_with_carbon,
            'landuse_project': self.landuse_project,
            'landuse_normalized': self.landuse_normalized,
            'landuse_weighted': self.landuse_weighted,
            'highest_position': self.highest_position
        }
        for activity in self.analysis_activities:
            activity_dict = {
                'uuid': str(activity.uuid),
                'name': activity.name,
                'description': activity.description,
                'path': activity.path,
                'layer_type': activity.layer_type,
                'user_defined': activity.user_defined,
                'pathways': [],
                'priority_layers': activity.priority_layers,
                'layer_styles': activity.layer_styles
            }
            for pathway in activity.pathways:
                activity_dict["pathways"].append({
                    'uuid': str(pathway.uuid),
                    'name': pathway.name,
                    'description': pathway.description,
                    'path': pathway.path,
                    'layer_type': pathway.layer_type,
                    'carbon_paths': pathway.carbon_paths
                })
            input_dict["activities"].append(activity_dict)
        return input_dict

    @classmethod
    def from_dict(cls, data: dict) -> typing.Self:
        config = TaskConfig(
            data.get('scenario_name', ''), data.get('scenario_desc', ''),
            data.get('extent', []), [], [], []
        )
        config.priority_layers = data.get('priority_layers', [])
        config.priority_layer_groups = data.get('priority_layer_groups', [])
        config.snapping_enabled = data.get(
            'snapping_enabled', DEFAULT_VALUES.snapping_enabled)
        config.snap_layer_uuid = data.get('snap_layer_uuid', '')
        config.pathway_suitability_index = data.get(
            'pathway_suitability_index',
            DEFAULT_VALUES.pathway_suitability_index)
        config.carbon_coefficient = data.get(
            'carbon_coefficient', DEFAULT_VALUES.carbon_coefficient)
        config.snap_rescale = data.get(
            'snap_rescale', DEFAULT_VALUES.snap_rescale)
        config.snap_method = data.get(
            'snap_method', DEFAULT_VALUES.snap_method)
        config.sieve_enabled = data.get(
            'sieve_enabled', DEFAULT_VALUES.sieve_enabled)
        config.sieve_threshold = data.get(
            'sieve_threshold', DEFAULT_VALUES.sieve_threshold)
        config.sieve_mask_uuid = data.get('sieve_mask_uuid', '')
        config.mask_layer_uuids = data.get('mask_layer_uuids', '')
        config.ncs_with_carbon = data.get(
            'ncs_with_carbon', DEFAULT_VALUES.ncs_with_carbon)
        config.landuse_project = data.get(
            'landuse_project', DEFAULT_VALUES.landuse_project)
        config.landuse_normalized = data.get(
            'landuse_normalized', DEFAULT_VALUES.landuse_normalized)
        config.landuse_weighted = data.get(
            'landuse_weighted', DEFAULT_VALUES.landuse_weighted)
        config.highest_position = data.get(
            'highest_position', DEFAULT_VALUES.highest_position)
        # store dict of <layer_uuid, list of obj identifier>
        config.priority_uuid_layers = {}
        config.pathway_uuid_layers = {}
        config.carbon_uuid_layers = {}
        for priority_layer in config.priority_layers:
            priority_layer_uuid = priority_layer.get('uuid', None)
            if not priority_layer_uuid:
                continue
            layer_uuid = priority_layer.get('layer_uuid', None)
            if not layer_uuid:
                continue
            if layer_uuid in config.priority_uuid_layers:
                config.priority_uuid_layers[layer_uuid].append(
                    priority_layer_uuid)
            else:
                config.priority_uuid_layers[layer_uuid] = [
                    priority_layer_uuid
                ]
        _activities = data.get('activities', [])
        for activity in _activities:
            uuid_str = activity.get('uuid', None)
            m_priority_layers = activity.get('priority_layers', [])
            filtered_priority_layer = []
            for m_priority_layer in m_priority_layers:
                if not m_priority_layer:
                    continue
                filtered_priority_layer.append(m_priority_layer)
                m_priority_uuid = m_priority_layer.get('uuid', None)
                if not m_priority_uuid:
                    continue
                m_priority_layer_uuid = m_priority_layer.get(
                    'layer_uuid', None)
                if not m_priority_layer_uuid:
                    continue
                if m_priority_layer_uuid in config.priority_uuid_layers:
                    config.priority_uuid_layers[m_priority_layer_uuid].append(
                        m_priority_uuid)
                else:
                    config.priority_uuid_layers[m_priority_layer_uuid] = [
                        m_priority_uuid
                    ]
            activity_obj = Activity(
                uuid=uuid.UUID(uuid_str) if uuid_str else uuid.uuid4(),
                name=activity.get('name', ''),
                description=activity.get('description', ''),
                path='',
                layer_type=LayerType(activity.get('layer_type', -1)),
                user_defined=activity.get('user_defined', False),
                pathways=[],
                priority_layers=filtered_priority_layer,
                layer_styles=activity.get('layer_styles', {})
            )
            pathways = activity.get('pathways', [])
            for pathway in pathways:
                pw_uuid_str = pathway.get('uuid', None)
                pw_uuid = (
                    uuid.UUID(pw_uuid_str) if pw_uuid_str else
                    uuid.uuid4()
                )
                pathway_model = NcsPathway(
                    uuid=pw_uuid,
                    name=pathway.get('name', ''),
                    description=pathway.get('description', ''),
                    path=pathway.get('path', ''),
                    layer_type=LayerType(pathway.get('layer_type', -1)),
                    # store carbon layer uuids instead of the path
                    carbon_paths=pathway.get('carbon_uuids', [])
                )
                activity_obj.pathways.append(pathway_model)
                pw_layer_uuid = pathway.get('layer_uuid', None)
                if pw_layer_uuid:
                    if pw_layer_uuid in config.pathway_uuid_layers:
                        config.pathway_uuid_layers[pw_layer_uuid].append(
                            str(pw_uuid))
                    else:
                        config.pathway_uuid_layers[pw_layer_uuid] = [
                            str(pw_uuid)
                        ]
                carbon_uuids = pathway.get('carbon_uuids', [])
                for carbon_uuid in carbon_uuids:
                    if carbon_uuid in config.carbon_uuid_layers:
                        config.carbon_uuid_layers[carbon_uuid].append(
                            str(pw_uuid))
                    else:
                        config.carbon_uuid_layers[carbon_uuid] = [
                            str(pw_uuid)
                        ]

            config.analysis_activities.append(activity_obj)
        config.scenario = Scenario(
            uuid=config.scenario_uuid,
            name=config.scenario_name,
            description=config.scenario_desc,
            extent=config.analysis_extent,
            activities=config.analysis_activities,
            weighted_activities=[],
            priority_layer_groups=config.priority_layer_groups
        )
        config.total_input_layers = (
            len(config.pathway_uuid_layers) +
            len(config.priority_uuid_layers) +
            len(config.carbon_uuid_layers)
        )
        if config.snap_layer_uuid:
            config.total_input_layers += 1
        if config.sieve_mask_uuid:
            config.total_input_layers += 1
        config.total_input_layers += len(config.mask_layer_uuids)
        return config
