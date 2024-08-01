# -*- coding: utf-8 -*-

"""Helper functions for supporting model management."""

from dataclasses import fields
import typing
import uuid

from .base import (
    Activity,
    LayerModelComponent,
    LayerModelComponentType,
    NcsPathway,
)


def clone_layer_component(
    layer_component: LayerModelComponent,
    model_cls: typing.Callable[[uuid.UUID, str, str], LayerModelComponentType],
) -> typing.Union[LayerModelComponent, None]:
    """Clones a layer-based model component.

    :param layer_component: Layer-based model component to clone.
    :type layer_component: LayerModelComponent

    :param model_cls: Callable class that will be created based on the
    input argument values from the dictionary.
    :type model_cls: LayerModelComponent

    :returns: A new instance of the cloned model component. It
    will return None if the input is not a layer-based model
    component.
    :rtype: LayerModelComponent
    """
    if not isinstance(layer_component, LayerModelComponent):
        return None

    cloned_component = model_cls(
        layer_component.uuid, layer_component.name, layer_component.description
    )

    for f in fields(layer_component):
        attr_val = getattr(layer_component, f.name)
        setattr(cloned_component, f.name, attr_val)

    return cloned_component


def clone_ncs_pathway(ncs: NcsPathway) -> NcsPathway:
    """Creates a deep copy of the given NCS pathway.

    :param ncs: NCS pathway to clone.
    :type ncs: NcsPathway

    :returns: A deep copy of the original NCS pathway object.
    :rtype: NcsPathway
    """
    return clone_layer_component(ncs, NcsPathway)


def clone_activity(
    activity: Activity,
) -> Activity:
    """Creates a deep copy of the given activity.

    :param activity: activity to clone.
    :type activity: Activity

    :returns: A deep copy of the original activity object.
    :rtype: Activity
    """
    activity = clone_layer_component(activity, Activity)
    if activity is None:
        return None

    pathways = activity.pathways
    cloned_pathways = []
    for p in pathways:
        cloned_ncs = clone_ncs_pathway(p)
        if cloned_ncs is not None:
            cloned_pathways.append(cloned_ncs)

    activity.pathways = cloned_pathways

    return activity
