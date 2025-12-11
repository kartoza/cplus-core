# coding=utf-8
"""
 Plugin tasks related to the scenario analysis

"""
import datetime
import os
import traceback
import uuid
from pathlib import Path
import typing

import math
from qgis import processing
from qgis.PyQt import QtCore
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsRasterLayer,
    QgsRectangle,
    QgsVectorLayer,
    QgsWkbTypes,
    QgsTask,
    QgsRasterBandStats,
    QgsProcessingFeatureSourceDefinition,
    QgsFeatureRequest
)

from ..utils.conf import Settings
from ..definitions.defaults import (
    SCENARIO_OUTPUT_FILE_NAME, DEFAULT_VALUES
)
from ..models.base import ScenarioResult, Activity, NcsPathway
from ..utils.helper import (
    align_rasters,
    clean_filename,
    tr,
    BaseFileUtils,
    normalize_raster,
)
from .task_config import TaskConfig


class ScenarioAnalysisTask(QgsTask):
    """Prepares and runs the scenario analysis"""

    status_message_changed = QtCore.pyqtSignal(str)
    info_message_changed = QtCore.pyqtSignal(str, int)

    custom_progress_changed = QtCore.pyqtSignal(float)
    log_received = QtCore.pyqtSignal(str, str, bool, bool)
    task_cancelled = QtCore.pyqtSignal(bool)

    def __init__(self, task_config: TaskConfig):
        super().__init__()
        self.task_config = task_config
        self.analysis_scenario_name = task_config.scenario.name
        self.analysis_scenario_description = task_config.scenario.description

        self.analysis_activities: typing.List[Activity] = task_config.analysis_activities
        self.analysis_priority_layers_groups = (
            task_config.priority_layer_groups
        )
        self.analysis_extent = task_config.scenario.extent
        self.analysis_extent_string = None

        self.analys_crs = task_config.scenario.extent.crs
        self.clip_to_studyarea = task_config.clip_to_studyarea

        self.relative_impact_matrix = task_config.relative_impact_matrix

        self.analysis_weighted_activities = []
        self.scenario_result = None

        self.success = True
        self.output = None
        self.error = None
        self.status_message = None

        self.info_message = None

        self.processing_cancelled = False
        self.feedback = QgsProcessingFeedback()
        self.processing_context = QgsProcessingContext()

        self.scenario = task_config.scenario
        self.scenario_directory = task_config.base_dir

    def get_settings_value(self, name: str, default=None, setting_type=None):
        """Get attribute value by attribute name.

        :param name: Attribute name/config key
        :type name: Settings

        :param default: Default value if not found, defaults to None
        :type default: any, optional

        :param setting_type: type of attribute, defaults to None
        :type setting_type: any, optional

        :return: Attribute value
        :rtype: any
        """
        return self.task_config.get_value(name, default)

    def get_priority_layer(self, identifier):
        """Get priority layer dict by its UUID.

        :param identifier: Priority Layer UUID
        :type identifier: str

        :return: Priority Layer dict
        :rtype: typing.Dict
        """
        return self.task_config.get_priority_layer(identifier)

    def get_activity(self, activity_uuid):
        """Get activity object by its UUID.

        :param activity_uuid: activity UUID
        :type activity_uuid: str

        :return: Activity object or None if not found
        :rtype: typing.Union[Activity, None]
        """
        return self.task_config.get_activity(activity_uuid)

    def get_priority_layers(self):
        """Get all priority layers.

        :return: List of priority layer dictionary
        :rtype: typing.List
        """
        return self.task_config.get_priority_layers()

    def get_masking_layers(self):
        """Get masking layers

        :return: list of mask layer file path
        :rtype: List
        """
        masking_layers_paths = self.get_settings_value(
            Settings.MASK_LAYERS_PATHS, default=None
        )
        masking_layers = masking_layers_paths.split(",") if masking_layers_paths else []

        masking_layers.remove("") if "" in masking_layers else None
        return masking_layers

    def cancel_task(self, exception=None):
        """Cancel current task.

        :param exception: exception if any, defaults to None
        :type exception: Exception, optional
        """
        self.error = exception
        try:
            self.cancel()
        except Exception:
            pass
        finally:
            self.task_cancelled.emit(exception is not None)

    def log_message(
        self,
        message: str,
        name: str = "qgis_cplus",
        info: bool = True,
        notify: bool = True,
    ):
        """Handle when log is received from running task.

        :param message: Message log
        :type message: str

        :param name: log name, defaults to "qgis_cplus"
        :type name: str, optional

        :param info: True if it is information log, defaults to True
        :type info: bool, optional

        :param notify: Not used in API, defaults to True
        :type notify: bool, optional
        """
        self.log_received.emit(message, name, info, notify)

    def on_terminated(self):
        """Called when the task is terminated."""
        message = "Processing has been cancelled by the user."
        if self.error:
            message = f"Problem in running scenario analysis: {self.error}"
        self.set_status_message(tr(message))
        self.log_message(message)

    def get_extent_string(
            self, 
            reference_path: str,
            processing_extent: QgsRectangle,
            align_with_processing_extent: bool = True
        ) -> str:
        """Get the extent string for the analysis based on the reference layer
        and the processing extent.
        :param reference_path: Path to the reference layer
        :type reference_path: str
        :param processing_extent: The extent to be aligned with the reference layer
        :type processing_extent: QgsRectangle
        :return: The extent string in the format "xmin,xmax,ymin,ymax [CRS]"
        :param align_with_processing_extent: Whether to align the extent with the
        processing extent, defaults to True
        :type align_with_processing_extent: bool
        :rtype: str
        """
        target_layer = QgsRasterLayer(reference_path, "target_layer")

        if self.analys_crs is not None:
            # Use the CRS of the analysis if it is provided
            dest_crs = QgsCoordinateReferenceSystem(self.analys_crs)
        else:
            # Use the CRS of the target layer if it is valid, otherwise use EPSG:4326
            dest_crs = (
                target_layer.crs()
                if target_layer.isValid()
                else QgsCoordinateReferenceSystem("EPSG:4326")
            )
            

        if align_with_processing_extent:            
            snapped_extent = self.align_extent(target_layer, processing_extent)
        else:
            snapped_extent = target_layer.extent()

        self.log_message(
            "Snapped area of interest extent " f"{snapped_extent.asWktPolygon()} \n"
        )

        return (
            f"{snapped_extent.xMinimum()},{snapped_extent.xMaximum()},"
            f"{snapped_extent.yMinimum()},{snapped_extent.yMaximum()}"
            f" [{dest_crs.authid()}]"
        )

    def get_reference_layer(self):
        """Get the path of the reference layer

        Returns:
            str|None: Return the path of the reference layer or None is it doesn't exist
        """
        snapping_enabled = self.get_settings_value(
            Settings.SNAPPING_ENABLED, default=False, setting_type=bool
        )
        if snapping_enabled is False:
            self.log_message("Snapping is disabled, no reference layer will be used.")
            return None
        
        reference_layer = self.get_settings_value(Settings.SNAP_LAYER, default="")
        if reference_layer is None or reference_layer == "":
            self.log_message(
                "No reference layer set for snapping, "
                "snapping will not be performed."
            )
            return None
        
        reference_layer_path = Path(reference_layer)
        if (
            snapping_enabled
            and os.path.exists(reference_layer)
            and reference_layer_path.is_file()
        ):
            return reference_layer

    def run(self):
        """Runs the main scenario analysis task operations"""

        BaseFileUtils.create_new_dir(self.scenario_directory)

        selected_pathway = None
        pathway_found = False

        for activity in self.analysis_activities:
            if pathway_found:
                break
            for pathway in activity.pathways:
                if pathway is not None:
                    pathway_found = True
                    selected_pathway = pathway
                    break

        processing_extent = QgsRectangle(
            float(self.analysis_extent.bbox[0]),
            float(self.analysis_extent.bbox[2]),
            float(self.analysis_extent.bbox[1]),
            float(self.analysis_extent.bbox[3]),
        )

        extent_string = self.get_extent_string(
            reference_path=selected_pathway.path,
            processing_extent=processing_extent
        )

        self.log_message(
            "Original area of interest extent: "
            f"{processing_extent.asWktPolygon()} \n"
        )

        # Run pathways layers snapping using a specified reference layer

        snapping_enabled = self.get_settings_value(
            Settings.SNAPPING_ENABLED, default=False, setting_type=bool
        )
        reference_layer = self.get_reference_layer()
        if (
            snapping_enabled
            and reference_layer
        ):
            extent_string = self.get_extent_string(
                reference_path=reference_layer,
                processing_extent=processing_extent,
                align_with_processing_extent=False
            )
            self.snap_analysis_data(extent_string)

        # Clip to StudyArea
        studyarea_path = self.get_settings_value(Settings.STUDYAREA_PATH, default="")
        if self.clip_to_studyarea and studyarea_path and studyarea_path != "":
            self.clip_analysis_data(studyarea_path)

        # Reproject the pathways and priority layers to the
        # scenario CRS if it is not the same as the pathways CRS

        if self.analys_crs is not None:
            self.reproject_pathways(
                target_extent=extent_string,
                target_crs=QgsCoordinateReferenceSystem(self.analys_crs)
            )

        # Replace no data value for the pathways and priority layers
        nodata_value = float(
            self.get_settings_value(
                Settings.NCS_NO_DATA_VALUE,
                default=DEFAULT_VALUES.nodata_value,
                setting_type=float
            )
        )
        self.log_message(
            f"Replacing nodata value for the pathways and priority layers to {nodata_value}"
        )
        self.run_pathways_replace_nodata(
            nodata_value=nodata_value
        )

        # Calculate total carbon mitigation values for the Naturebase pathways
        self.run_pathways_carbon_summation()
        
        # Normalize the pathways
        self.run_pathways_normalization()

        # Weight the pathways using the pathway suitability index
        # and priority group coefficients for the PWLs

        save_output = self.get_settings_value(
            Settings.NCS_WEIGHTED, default=True, setting_type=bool
        )

        self.run_pathways_weighting(
            self.analysis_priority_layers_groups,
            extent_string,
            temporary_output=not save_output,
        )

        # Creating activities from the weigghted pathways
        save_output = self.get_settings_value(
            Settings.LANDUSE_PROJECT, default=True, setting_type=bool
        )

        self.run_activities_analysis(
            extent_string,
            temporary_output=not save_output,
        )

        # Normalize the activities. 
        # This is useful when weighting pathways with relative impact matrix
        # Activities created in previous step may have values greater than 1
        self.run_activity_normalization()

        # Run masking of the activities layers
        masking_layers = self.get_masking_layers()
        self.log_message(f"Masking layers: {masking_layers}")

        if masking_layers:
            self.run_activities_masking(extent_string)

        # Run internal masking of the activities layers
        self.run_internal_activities_masking(extent_string)

        # TODO enable the sieve functionality
        sieve_enabled = self.get_settings_value(
            Settings.SIEVE_ENABLED, default=False, setting_type=bool
        )

        if sieve_enabled:
            self.run_activities_sieve(
                self.analysis_activities,
            )

        # Clean up activities
        save_output = self.get_settings_value(
            Settings.LANDUSE_NORMALIZED, default=True, setting_type=bool
        )

        self.run_activities_cleaning(
            extent_string,
            temporary_output=not save_output
        )

        # Investability
        self.run_investability_analysis()
        

        # The highest position tool analysis
        save_output = self.get_settings_value(
            Settings.HIGHEST_POSITION, default=True, setting_type=bool
        )
        self.run_highest_position_analysis(temporary_output=not save_output)

        return True

    def finished(self, result: bool):
        """Calls the handler responsible for doing post analysis workflow.

        :param result: Whether the run() operation finished successfully
        :type result: bool
        """
        if result:
            self.log_message("Finished from the main task \n")
        else:
            self.log_message(f"Error from task scenario task {self.error}")

    def set_status_message(self, message):
        """Handle when status message is updated.

        :param message: status message
        :type message: str
        """
        self.status_message = message
        self.status_message_changed.emit(self.status_message)

    def set_info_message(self, message, level=Qgis.Info):
        """Handle when info message is updated.

        :param message: message
        :type message: str
        :param level: severity level, defaults to Qgis.Info
        :type level: int, optional
        """
        self.info_message = message
        self.info_message_changed.emit(self.info_message, level)

    def set_custom_progress(self, value):
        """Handle when progress value is updated.

        :param value: progress value
        :type value: float
        """
        self.custom_progress = value
        self.custom_progress_changed.emit(self.custom_progress)

    def update_progress(self, value):
        """Sets the value of the task progress

        :param value: Value to be set on the progress bar
        :type value: float
        """
        if not self.processing_cancelled:
            self.set_custom_progress(value)
        else:
            self.feedback = QgsProcessingFeedback()
            self.processing_context = QgsProcessingContext()

    def align_extent(self, raster_layer, target_extent):
        """Snaps the passed extent to the activities pathway layer pixel bounds

        :param raster_layer: The target layer that the passed extent will be
        aligned with
        :type raster_layer: QgsRasterLayer

        :param target_extent: Spatial extent that will be used a target extent when
        doing alignment.
        :type target_extent: QgsRectangle
        """

        try:
            raster_extent = raster_layer.extent()

            x_res = raster_layer.rasterUnitsPerPixelX()
            y_res = raster_layer.rasterUnitsPerPixelY()

            left = raster_extent.xMinimum() + x_res * math.floor(
                (target_extent.xMinimum() - raster_extent.xMinimum()) / x_res
            )
            right = raster_extent.xMinimum() + x_res * math.ceil(
                (target_extent.xMaximum() - raster_extent.xMinimum()) / x_res
            )
            bottom = raster_extent.yMinimum() + y_res * math.floor(
                (target_extent.yMinimum() - raster_extent.yMinimum()) / y_res
            )
            top = raster_extent.yMaximum() - y_res * math.floor(
                (raster_extent.yMaximum() - target_extent.yMaximum()) / y_res
            )

            return QgsRectangle(left, bottom, right, top)

        except Exception as e:
            self.log_message(
                tr(
                    f"Problem snapping area of "
                    f"interest extent, using the original extent,"
                    f"{str(e)}"
                )
            )

        return target_extent

    def replace_nodata(self, layer_path, output_path, nodata_value: float = DEFAULT_VALUES.nodata_value):
        """Adds nodata value info into the layer available
        in the passed layer_path and save the layer in the passed output_path
        path.

        The addition will replace any current nodata value available in
        the input layer.

        :param layer_path: Input layer path
        :type layer_path: str

        :param output_path: Output layer path
        :type output_path: str

        :param nodata_value: Nodata value to be used
        :type output_path: int

        :returns: Whether the task operations was successful
        :rtype: bool

        """
        self.feedback = QgsProcessingFeedback()
        self.feedback.progressChanged.connect(self.update_progress)

        try:
            alg_params = {
                "COPY_SUBDATASETS": False,
                "DATA_TYPE": 6,  # Float32
                "EXTRA": "",
                "INPUT": layer_path,
                "NODATA": None,
                "OPTIONS": "",
                "TARGET_CRS": None,
                "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
            }
            translate_output = processing.run(
                "gdal:translate",
                alg_params,
                context=self.processing_context,
                feedback=self.feedback,
                is_child_algorithm=True,
            )

            alg_params = {
                "DATA_TYPE": 0,  # Use Input Layer Data Type
                "EXTRA": "",
                "INPUT": translate_output["OUTPUT"],
                "MULTITHREADING": False,
                "NODATA": nodata_value,
                "OPTIONS": "",
                "RESAMPLING": 0,  # Nearest Neighbour
                "SOURCE_CRS": None,
                "TARGET_CRS": None,
                "TARGET_EXTENT": None,
                "TARGET_EXTENT_CRS": None,
                "TARGET_RESOLUTION": None,
                "OUTPUT": output_path,
            }
            outputs = processing.run(
                "gdal:warpreproject",
                alg_params,
                context=self.processing_context,
                feedback=self.feedback,
                is_child_algorithm=True,
            )

            return outputs is not None
        except Exception as e:
            self.log_message(
                f"Problem replacing no data value from a snapping output, {e}"
            )

        return False
    
    def run_pathways_normalization(
        self, 
    ) -> bool:
        """Runs normalization analysis on the pathways in the activities.
        The formula is: (pathway - min) / (max - min)

        :returns: True if the task operation was successfully completed else False.
        :rtype: bool
    """
        if self.processing_cancelled:
            return False

        self.set_status_message(tr("Normalization of pathways"))
        pathways :typing.List[NcsPathway] = []

        normalized_pathways_directory = os.path.join(
            self.scenario_directory, "normalized_pathways"
        )
        BaseFileUtils.create_new_dir(normalized_pathways_directory)

        try:
            for activity in self.analysis_activities:
                if not activity.pathways and (
                    activity.path is None or activity.path == ""
                ):
                    self.set_info_message(
                        tr(
                            f"No defined activity pathways or a"
                            f" activity layer for the activity {activity.name}"
                        ),
                        level=Qgis.Critical,
                    )
                    self.log_message(
                        f"No defined activity pathways or a "
                        f"activity layer for the activity {activity.name}"
                    )
                    return False

                for pathway in activity.pathways:
                    if not (pathway in pathways):
                        pathways.append(pathway)

            if len(pathways) > 0:
                for pathway in pathways:
                    if self.processing_cancelled:
                        return False
                    
                    pathway_layer = QgsRasterLayer(pathway.path, pathway.name)

                    if not pathway_layer.isValid():
                        self.log_message(
                            f"Pathway layer {pathway.name} is not valid, "
                            f"skipping the layer from snapping."
                        )
                        continue

                    provider = pathway_layer.dataProvider()
                    band_statistics = provider.bandStatistics(1)
                    min_value = band_statistics.minimumValue
                    max_value = band_statistics.maximumValue

                    if min_value is None or max_value is None:
                        self.log_message(
                            f"Pathway layer {pathway.name} has no valid "
                            f"statistics, skipping the layer from normalization."
                        )
                        continue
                    
                    if min_value == 0 and max_value == 1:                        
                        new_path = BaseFileUtils.copy_file(pathway.path, normalized_pathways_directory)
                        if new_path and os.path.exists(new_path):
                            pathway.path = new_path
                            self.log_message(
                                f"Pathway layer {pathway.name} is already normalized (min={min_value}, max={max_value}), "
                                f"skipping the layer from normalization for pathway {pathway.path}."
                            )
                        continue             
                    
                    self.log_message(f"Normalizing {pathway.name} pathway layer \n")

                    expression = f"(A - {min_value}) / ({max_value} - {min_value})"
                    output_path = os.path.join(
                        f"{normalized_pathways_directory}",
                        f"{Path(pathway.path).stem}_norm_{str(uuid.uuid4())[:4]}.tif"
                    )
                    alg_params = {
                        'INPUT_A': pathway.path,
                        "BAND_A": 1,
                        'FORMULA': expression,
                        'OPTIONS':'COMPRESS=DEFLATE|ZLEVEL=6|TILED=YES', # Compress the layer
                        'OUTPUT': output_path,
                    }

                    self.feedback = QgsProcessingFeedback()
                    self.feedback.progressChanged.connect(self.update_progress)

                    if self.processing_cancelled:
                        return False

                    result = processing.run(
                        "gdal:rastercalculator",
                        alg_params,
                        context=self.processing_context,
                        feedback=self.feedback,
                    )
                    if result.get("OUTPUT"):
                        pathway.path = result.get("OUTPUT")
                    else:
                        self.log_message(
                            f"Problem normalizing pathway layer {pathway.name}, "
                            f"skipping the layer from normalization."
                        )
            return True
        except Exception as e:
            self.log_message(f"Problem normalizing pathways, {e} \n")
            self.log_message(traceback.format_exc())
            self.cancel_task(e)
            return False
    
    def run_activity_normalization(
        self, 
    ) -> bool:
        """Runs normalization analysis on the activities.
        The formula is: (activity - min) / (max - min)

        :returns: True if the task operation was successfully completed else False.
        :rtype: bool
    """
        if self.processing_cancelled:
            return False

        self.set_status_message(tr("Normalization of activities"))
        pathways :typing.List[NcsPathway] = []

        normalized_activities_directory = os.path.join(
            self.scenario_directory, "normalized_activities"
        )
        BaseFileUtils.create_new_dir(normalized_activities_directory)

        try:
            for activity in self.analysis_activities:
                if not activity.path:
                    msg = f"No defined activity layer for the activity {activity.name}"
                    self.set_info_message(
                        tr(msg),
                        level=Qgis.Critical,
                    )
                    self.log_message(msg)
                    return False

                if self.processing_cancelled:
                    return False
                    
                activity_layer = QgsRasterLayer(activity.path, activity.name)

                if not activity_layer.isValid():
                    self.log_message(
                        f"Activity layer {activity.name} is not valid, "
                        f"skipping the layer from normalization."
                    )
                    continue

                provider = activity_layer.dataProvider()
                band_statistics = provider.bandStatistics(1)
                min_value = band_statistics.minimumValue
                max_value = band_statistics.maximumValue

                if min_value is None or max_value is None:
                    self.log_message(
                        f"Activity layer {activity.name} has no valid "
                        f"statistics, skipping the layer from normalization."
                    )
                    continue
                    
                if min_value == 0 and max_value == 1:
                    new_path = BaseFileUtils.copy_file(activity.path, normalized_activities_directory)
                    if new_path and os.path.exists(new_path):
                        activity.path = new_path
                        self.log_message(
                            f"Activity layer {activity.name} is already normalized (min={min_value}, max={max_value}), "
                            f"skipping the layer from normalization."
                        )
                    continue             
                    
                self.log_message(f"Normalizing {activity.name} activity layer \n")

                expression = f"(A - {min_value}) / ({max_value} - {min_value})"
                output_path = os.path.join(
                    f"{normalized_activities_directory}",
                    f"{Path(activity.path).stem}_norm_{str(uuid.uuid4())[:4]}.tif"
                )
                alg_params = {
                    'INPUT_A': activity.path,
                    'BAND_A': 1,
                    'FORMULA': expression,
                    'OPTIONS':'COMPRESS=DEFLATE|ZLEVEL=6|TILED=YES', # Compress the layer
                    'OUTPUT': output_path,
                }

                self.feedback = QgsProcessingFeedback()
                self.feedback.progressChanged.connect(self.update_progress)

                if self.processing_cancelled:
                    return False

                result = processing.run(
                    "gdal:rastercalculator",
                    alg_params,
                    context=self.processing_context,
                    feedback=self.feedback,
                )
                if result.get("OUTPUT"):
                    activity.path = result.get("OUTPUT")
                else:
                    self.log_message(
                        f"Problem normalizing activity layer {activity.name}, "
                        f"skipping the layer from normalization."
                    )
            return True
        except Exception as e:
            self.log_message(f"Problem normalizing activities, {e} \n")
            self.log_message(traceback.format_exc())
            self.cancel_task(e)
            return False
    
    def _can_clip_raster_by_mask(
        self, raster_layer: QgsRasterLayer, mask_layer: QgsVectorLayer
    ) -> bool:
        """
        Check if the raster layer can be clipped by the mask layer.

        Conditions:
            1. The difference in area of the extent of the raster and
                vector layer should be greater than 15%
            2. The mask layer should be intersecting the raster layer

        :param raster_layer: Raster layer
        :type raster_layer: QgsRasterLayer

        :param mask_layer: Masking layer
        :type mask_layer: QgsVectorLayer

        :returns: True if the check is successful else False.
        :rtype: bool
        """
        # Check if raster and mask layer is valid

        if not mask_layer.isValid() or not raster_layer.isValid():
            self.log_message(
                "Cancelling clipping raster, the mask layer or"
                " the raster layer is invalid."
            )
            self.cancel_task()
            return False

        # Reproject the masklayer if its CRS is different from the raster CRS
        if mask_layer.crs() != raster_layer.crs:
            mask_path_reprojected = self.reproject_layer(
                mask_layer.source(), raster_layer.crs(), is_raster=False
            )
            if mask_path_reprojected and os.path.exists(mask_path_reprojected):
                mask_layer = QgsVectorLayer(
                    mask_path_reprojected, "mask_layer_reprojected"
                )

        # If mask layer intersects the raster layer
        if not mask_layer.extent().intersects(raster_layer.extent()):
            self.log_message(
                "Cancelling clipping raster, the mask layer extent"
                " and the raster extent do not overlap."
            )
            self.cancel_task()
            return False

        mask_extent_area = mask_layer.extent().area()
        raster_extent_area = raster_layer.extent().area()

        area_ratio = mask_extent_area / raster_extent_area

        # If mask layer extent is 15 % greater than raster layer extent
        if abs(area_ratio - 1) < 0.15:
            self.log_message(
                "Skipping clipping raster layer, "
                "the mask layer extent is within 15 percent of the raster layer extent"
            )
            return False

        return True

    def clip_raster_by_mask(self, 
        input_raster_path: str,
        mask_layer_path: str,
        output_path: str
        ) -> bool:
        """Clip a given raster by the specified mask layer.
        :param input_raster_path: Input raster path
        :type input_raster_path: str

        :param mask_layer_path: Path to the masking layer
        :type mask_layer_path: str

        :param output_path: Output layer path
        :type output_path: str

        :returns: True if the task operation was successfully completed else False.
        :rtype: bool
        """
        raster_layer = QgsRasterLayer(input_raster_path, "raster_layer")
        mask_layer = QgsVectorLayer(mask_layer_path, "mask_layer")

        if not raster_layer.isValid():
            self.log_message(
                f"Invalid raster layer: {input_raster_path}\n"
            )
            return False

        if not mask_layer.isValid():
            self.log_message(
                f"Invalid mask layer: {mask_layer_path}\n"
            )
            return False
        
        try:
            alg_params = {
                "INPUT": input_raster_path,
                "MASK": mask_layer,
                "SOURCE_CRS": raster_layer.crs(),
                "DESTINATION_CRS": raster_layer.crs(),
                "OUTPUT": output_path,
                "NO_DATA": self.get_settings_value(
                    Settings.NCS_NO_DATA_VALUE, DEFAULT_VALUES.nodata_value
                ),
                "CROP_TO_CUTLINE": True
            }

            self.log_message(
                f"Used parameters for clipping the raster: {input_raster_path} "
                f" using mask layer: {alg_params} \n"
            )

            self.feedback = QgsProcessingFeedback()
            self.feedback.progressChanged.connect(self.update_progress)

            if self.processing_cancelled:
                return False

            result = processing.run(
                "gdal:cliprasterbymasklayer",
                alg_params,
                context=self.processing_context,
                feedback=self.feedback,
            )
            if result.get("OUTPUT"):
                return True
        
        except Exception as e:
            self.log_message(f"Problem clipping the layer {e} \n")
        return False
    
    def clip_analysis_data(self, studyarea_path: str) -> bool:
        """Clips the activity pathways and priority layers by the given study area.
        :param studyarea_path: The path to the study area layer
        :type studyarea_path: str
        :returns: True if the task operation was successfully completed else False.
        :rtype: bool
        """
        if self.processing_cancelled:
            return False
        
        mask_layer = QgsVectorLayer(studyarea_path, "mask_layer")

        if not mask_layer.isValid():
            self.log_message(
                f"Invalid mask layer: {studyarea_path} "
                f"Skipping clipping of activity pathways and priority layers\n"
            )
            return False

        self.set_status_message(
            tr(
                "Clipping the activity pathways and priority layers by the study area layer"
            )
        )

        clipped_pathways_directory = os.path.join(
            self.scenario_directory, "pathways", "clipped"
        )
        BaseFileUtils.create_new_dir(clipped_pathways_directory)

        clipped_priority_directory = os.path.join(
            self.scenario_directory, "priority_layer", "clipped"
        )
        BaseFileUtils.create_new_dir(clipped_priority_directory)

        pathways: typing.List[NcsPathway] = []

        try:
            for activity in self.analysis_activities:
                if not activity.pathways and (
                    activity.path is None or activity.path == ""
                ):
                    self.set_info_message(
                        tr(
                            f"No defined activity pathways or "
                            f" activity layers for the activity {activity.name}"
                        ),
                        level=Qgis.Critical,
                    )
                    self.log_message(
                        f"No defined activity pathways or "
                        f"activity layers for the activity {activity.name}"
                    )
                    return False

                for pathway in activity.pathways:
                    if not (pathway in pathways):
                        pathways.append(pathway)

            if pathways is not None and len(pathways) > 0:                
                for pathway in pathways:
                    pathway_layer = QgsRasterLayer(pathway.path, pathway.name)

                    if self.processing_cancelled:
                        return False
                    if not pathway_layer.isValid():
                        self.log_message(
                            f"Pathway layer {pathway.name} is not valid, "
                            f"skipping clipping the layer."
                        )
                        continue
                    self.log_message(
                        f"Clipping the {pathway.name} pathway layer by "
                        f"the study area layer\n"
                    )

                    if not self._can_clip_raster_by_mask(pathway_layer, aoi_layer):
                        continue

                    if self.processing_cancelled:
                        return False

                    output_file = os.path.join(
                        clipped_pathways_directory,
                        f"{Path(pathway.path).stem}_{str(self.scenario.uuid)[:4]}.tif",
                    )

                    result = self.clip_raster_by_mask(
                        pathway.path, studyarea_path, output_file
                    )
                    if result:
                        pathway.path = output_file

                    self.log_message(
                        f"Clipping the {pathway.name} pathway's {len(pathway.priority_layers)} "
                        f"priority weighting layers by study area\n"
                    )

                    if (
                        pathway.priority_layers is not None
                        and len(pathway.priority_layers) > 0
                    ):
                        priority_layers = []
                        for priority_layer in pathway.priority_layers:
                            if priority_layer is None:
                                continue

                            if self.processing_cancelled:
                                return False

                            priority_layer_settings = self.get_priority_layer(
                                priority_layer.get("uuid")
                            )
                            if priority_layer_settings is None:
                                continue

                            priority_layer_path = priority_layer_settings.get("path")

                            if not Path(priority_layer_path).exists():
                                priority_layers.append(priority_layer)
                                continue

                            layer = QgsRasterLayer(
                                priority_layer_path, f"{str(uuid.uuid4())[:4]}"
                            )
                            if not layer.isValid():
                                self.log_message(
                                    f"Priority layer {priority_layer.get('name')} "
                                    f"from pathway {pathway.name} is not valid, "
                                    f"skipping clipping the layer."
                                )
                                continue

                            if not self._can_clip_raster_by_mask(layer, aoi_layer):
                                continue

                            if self.processing_cancelled:
                                return False

                            self.log_message(
                                f"Clipping the {priority_layer.get('name')} priority layer "
                                f"by study area layer\n"
                            )

                            output_file = os.path.join(
                                clipped_priority_directory,
                                f"{Path(pathway.path).stem}_{str(self.scenario.uuid)[:4]}.tif",
                            )

                            result = self.clip_raster_by_mask(
                                priority_layer_path, studyarea_path, output_file
                            )
                            if result:
                                priority_layer["path"] = output_file

                            priority_layers.append(priority_layer)

                        pathway.priority_layers = priority_layers

        except Exception as e:
            self.log_message(f"Problem clipping the layers, {e} \n")
            self.cancel_task(e)
            return False

        return True

    def run_pathways_weighting(
        self,
        priority_layers_groups: dict,
        extent: str,
        temporary_output: bool = False
    ) -> bool:
        """Runs weighting analysis on the pathways in the activities using
        the corresponding NCS PWLs.

        The formula is: (suitability_index * pathway) *
        ((priority group coefficient 1 * impact weight * PWL 1) +
        (priority group coefficient 2 * impact weight * PWL 2) ...)

        :param priority_layers_groups: Used priority layers groups and their values
        :type priority_layers_groups: dict

        :param extent: selected extent from user
        :type extent: str

        :param temporary_output: Whether to save the processing outputs as temporary
        files
        :type temporary_output: bool

        :returns: True if the task operation was successfully completed else False.
        :rtype: bool
        """
        if self.processing_cancelled:
            return False

        self.set_status_message(tr("Weighting of pathways"))

        if len(self.analysis_activities) == 0:
            msg = tr("No defined activities for running pathways weighting.")
            self.set_info_message(
                msg,
                level=Qgis.Critical,
            )
            self.log_message(msg)
            return False

        # Get relative impact matrix items
        pathway_uuids = self.relative_impact_matrix.get("pathway_uuids", [])
        priority_layer_uuids = self.relative_impact_matrix.get("priority_layer_uuids", [])
        relative_impact_values = self.relative_impact_matrix.get("values", [])
        
        # Get valid pathways
        pathways: typing.List[NcsPathway] = []
        activities_paths = []

        try:
            # Validate activities and corresponding pathways
            for activity in self.analysis_activities:
                if not activity.pathways and (
                    activity.path is None or activity.path == ""
                ):
                    self.set_info_message(
                        tr(
                            f"No defined activity pathways or an"
                            f" activity layer for the activity {activity.name}"
                        ),
                        level=Qgis.Critical,
                    )
                    self.log_message(
                        f"No defined activity pathways or an "
                        f"activity layer for the activity {activity.name}"
                    )
                    return False

                for pathway in activity.pathways:
                    if pathway not in pathways:
                        pathways.append(pathway)

                if activity.path is not None and activity.path != "":
                    activities_paths.append(activity.path)

            if not pathways and len(activities_paths) > 0:
                self.run_activities_analysis(self.analysis_activities, extent)
                return False

            suitability_index = float(
                self.get_settings_value(
                    Settings.PATHWAY_SUITABILITY_INDEX,
                    default=0
                )
            )

            settings_priority_layers = self.get_priority_layers()

            weighted_pathways_directory = os.path.join(
                self.scenario_directory, "weighted_pathways"
            )
            BaseFileUtils.create_new_dir(weighted_pathways_directory)

            for pathway in pathways:
                # Skip processing if cancelled
                if self.processing_cancelled:
                    return False

                base_names = []
                layers = [pathway.path]
                run_calculation = False

                # Include suitability index if not zero
                pathway_basename = Path(pathway.path).stem
                if suitability_index > 0:
                    base_names.append(
                        f'({suitability_index}*"{pathway_basename}@1")'
                    )
                    run_calculation = True
                else:
                    base_names.append(f'("{pathway_basename}@1")')

                for layer in pathway.priority_layers:
                    if not any(priority_layers_groups):
                        self.log_message(
                            "There are no defined priority layers in groups,"
                            " skipping the inclusion of PWLs in pathways "
                            "weighting."
                        )
                        break

                    if layer is None:
                        continue

                    if layer.get("uuid") is None:
                        continue

                    pwl = layer.get("path")

                    missing_pwl_message = (
                        f"Path {pwl} for priority "
                        f"weighting layer {layer.get('name')} "
                        f"doesn't exist, skipping the layer "
                        f"from the pathway {pathway.name} weighting."
                    )
                    if pwl is None or pwl == "":
                        self.log_message(missing_pwl_message)
                        continue

                    pwl_path = Path(pwl)

                    if not pwl_path.exists():
                        self.log_message(missing_pwl_message)
                        continue

                    pwl_path_basename = pwl_path.stem

                    for priority_layer in settings_priority_layers:
                        if priority_layer.get("name") == layer.get("name"):
                            for group in priority_layer.get("groups", []):
                                try:
                                    row = pathway_uuids.index(str(pathway.uuid))
                                    col = priority_layer_uuids.index(layer["uuid"])
                                except ValueError:
                                    self.log_message(
                                        f"Could not find pathway uuid {pathway.uuid} or "
                                        f"priority layer uuid {layer['uuid']} in the relative impact matrix."
                                    )
                                    impact_value = None
                                else:
                                    if row < len(relative_impact_values) and col < len(relative_impact_values[row]):
                                        impact_value = relative_impact_values[row][col]
                                    else:
                                        self.log_message(
                                            f"Index out of range for relative impact "
                                            f"matrix: row={row}, col={col}."
                                        )
                                        impact_value = None

                                value = group.get("value")
                                priority_group_coefficient = float(value)
                                if priority_group_coefficient > 0:
                                    if pwl not in layers:
                                        layers.append(pwl)

                                    pwl_expression = (
                                        f'({priority_group_coefficient}*'
                                        f'"{pwl_path_basename}@1")'
                                    )

                                    if impact_value is not None:                                        
                                        pwl_expression = (
                                            f'({priority_group_coefficient * int(impact_value)}*'
                                            f'"{pwl_path_basename}@1")'
                                        )
                                        # Inverse the PWL
                                        if impact_value < 0:
                                            pwl_expression = (
                                                f'({priority_group_coefficient * abs(int(impact_value))}*'
                                                f'("{pwl_path_basename}@1" - 1) * -1)'
                                        )
                                    base_names.append(pwl_expression)
                                    if not run_calculation:
                                        run_calculation = True

                # No need to run the calculation if suitability index is
                # zero or there are no PWLs in the activity.
                if not run_calculation:
                    continue

                file_name = clean_filename(pathway.name.replace(" ", "_"))
                output_file = os.path.join(
                    weighted_pathways_directory,
                    f"{file_name}_{str(uuid.uuid4())[:4]}.tif",
                )

                expression = f"{base_names[0]}"                
                if len(base_names) > 1:
                    pwl_calc_expression = " + ".join(base_names[1:])
                    expression += f" * ({pwl_calc_expression})"

                output = (
                    QgsProcessing.TEMPORARY_OUTPUT
                    if temporary_output
                    else output_file
                )

                # Actual processing calculation
                alg_params = {
                    "CELLSIZE": 0,
                    "CRS": None,
                    "EXPRESSION": expression,
                    "EXTENT": extent,
                    "LAYERS": layers,
                    "OUTPUT": output,
                }

                self.log_message(
                    f" Used parameters for calculating weighting pathways "
                    f"{alg_params} \n"
                )

                self.feedback = QgsProcessingFeedback()
                self.feedback.progressChanged.connect(self.update_progress)

                if self.processing_cancelled:
                    return False

                results = processing.run(
                    "qgis:rastercalculator",
                    alg_params,
                    context=self.processing_context,
                    feedback=self.feedback,
                )
                pathway.path = results["OUTPUT"]

        except Exception as e:
            self.log_message(f"Problem weighting pathways, {e}\n")
            self.cancel_task(e)
            self.log_message(traceback.format_exc())
            return False

        return True

    def snap_analysis_data(self, extent: str):
        """Snaps the passed activities pathways, carbon layers and priority
        layers to align with the reference layer set on the settings
        manager.

        :param extent: The selected extent from user
        :type extent: list
        """
        if self.processing_cancelled:
            # Will not proceed if processing has been cancelled by the user
            return False

        self.set_status_message(
            tr(
                "Snapping the selected activity pathways "
                "and priority layers"
            )
        )

        pathways :typing.List[NcsPathway] = []

        try:
            for activity in self.analysis_activities:
                if not activity.pathways and (
                    activity.path is None or activity.path == ""
                ):
                    self.set_info_message(
                        tr(
                            f"No defined activity pathways or a"
                            f" activity layer for the activity {activity.name}"
                        ),
                        level=Qgis.Critical,
                    )
                    self.log_message(
                        f"No defined activity pathways or a "
                        f"activity layer for the activity {activity.name}"
                    )
                    return False

                for pathway in activity.pathways:
                    if not (pathway in pathways):
                        pathways.append(pathway)

            reference_layer_path = self.get_settings_value(Settings.SNAP_LAYER)
            rescale_values = self.get_settings_value(
                Settings.RESCALE_VALUES, default=False, setting_type=bool
            )

            resampling_method = self.get_settings_value(
                Settings.RESAMPLING_METHOD, default=0
            )

            if pathways is not None and len(pathways) > 0:
                snapped_pathways_directory = os.path.join(
                    self.scenario_directory, "pathways"
                )

                BaseFileUtils.create_new_dir(snapped_pathways_directory)

                for pathway in pathways:
                    pathway_layer = QgsRasterLayer(pathway.path, pathway.name)
                    nodata_value = pathway_layer.dataProvider().sourceNoDataValue(1)

                    if self.processing_cancelled:
                        return False
                    if not pathway_layer.isValid():
                        self.log_message(
                            f"Pathway layer {pathway.name} is not valid, "
                            f"skipping the layer from snapping."
                        )
                        continue
                    self.log_message(f"Snapping {pathway.name} pathway layer \n")

                    # Pathway snapping

                    output_path = self.snap_layer(
                        input_path=pathway.path,
                        reference_path=reference_layer_path,
                        extent=extent,
                        directory=snapped_pathways_directory,
                        rescale_values=rescale_values,
                        resampling_method=resampling_method,
                        nodata_value=nodata_value,
                        name=pathway.name
                    )
                    if output_path:
                        pathway.path = output_path

                    self.log_message(
                        f"Snapping {len(pathway.priority_layers)} "
                        f"priority weighting layers from pathway {pathway.name} with layers\n"
                    )

                    if (
                        pathway.priority_layers is not None
                        and len(pathway.priority_layers) > 0
                    ):
                        snapped_priority_directory = os.path.join(
                            self.scenario_directory, "priority_layer"
                        )

                        BaseFileUtils.create_new_dir(snapped_priority_directory)

                        priority_layers = []
                        for priority_layer in pathway.priority_layers:
                            if priority_layer is None:
                                continue

                            priority_layer_settings = self.get_priority_layer(
                                priority_layer.get("uuid")
                            )
                            if priority_layer_settings is None:
                                continue

                            priority_layer_path = priority_layer_settings.get("path")

                            if not Path(priority_layer_path).exists():
                                priority_layers.append(priority_layer)
                                continue

                            layer = QgsRasterLayer(
                                priority_layer_path, f"{str(uuid.uuid4())[:4]}"
                            )
                            if not layer.isValid():
                                self.log_message(
                                    f"Priority layer {priority_layer.get('name')} "
                                    f"from pathway {pathway.name} is not valid, "
                                    f"skipping the layer from snapping."
                                )
                                continue

                            nodata_value_priority = layer.dataProvider().sourceNoDataValue(
                                1
                            )

                            priority_output_path = self.snap_layer(
                                input_path=priority_layer_path,
                                reference_path=reference_layer_path,
                                extent=extent,
                                directory=snapped_priority_directory,
                                rescale_values=rescale_values,
                                resampling_method=resampling_method,
                                nodata_value=nodata_value_priority,
                                name=priority_layer.get("name", "priority layer")
                            )

                            if priority_output_path:
                                priority_layer["path"] = priority_output_path

                            priority_layers.append(priority_layer)

                        pathway.priority_layers = priority_layers

        except Exception as e:
            self.log_message(f"Problem snapping layers, {e} \n")
            self.log_message(traceback.format_exc())
            self.cancel_task(e)
            return False

        return True

    def snap_layer(
        self,
        input_path: str,
        reference_path: str,
        extent: str,
        directory: str,
        rescale_values: bool,
        resampling_method: int,
        nodata_value: float = DEFAULT_VALUES.nodata_value,
        name: str = "layer"
    ):
        """Snaps the passed input layer using the reference layer and updates
        the snap output no data value to be the same as the original input layer
        no data value.

        :param input_path: Input layer source
        :type input_path: str

        :param reference_path: Reference layer source
        :type reference_path: str

        :param extent: Clip extent
        :type extent: list

        :param directory: Absolute path of the output directory for the snapped
        layers
        :type directory: str

        :param rescale_values: Whether to rescale pixel values
        :type rescale_values: bool

        :param resample_method: Method to use when resampling
        :type resample_method: QgsAlignRaster.ResampleAlg

        :param nodata_value: No data value to be set in the output layer. Defaults to -9999.0
        :type nodata_value: float

        :param name: Name of the layer
        :type name: str

        """
        if not os.path.exists(input_path):
            self.log_message(
                f"Input layer {input_path} does not exist, "
                f"skipping the layer from snapping."
            )
            return None
        if not os.path.exists(reference_path):
            self.log_message(
                f"Reference layer {reference_path} does not exist, "
                f"skipping the layer from snapping."
            )
            return None
        if not os.path.exists(directory):
            self.log_message(
                f"Output directory {directory} does not exist, "
                f"skipping the layer from snapping."
            )
            return None
        

        input_result_path, logs = align_rasters(
            input_path,
            reference_path,
            extent,
            directory,
            rescale_values,
            resampling_method,
            name=name,
        )
        for log in logs:
            self.log_message(log, info=("Problem" not in log))

        if input_result_path is not None:
            directory = os.path.join(self.scenario_directory, "snapped_layers")
            input_result_path = BaseFileUtils.copy_file(input_result_path, directory)

            result_path = Path(input_result_path)
            name = result_path.stem

            output_path = os.path.join(directory, f"{name}_final.tif")

            ok = self.replace_nodata(
                layer_path=input_result_path,
                output_path=reference_path, 
                nodata_value=nodata_value,
            )

            if ok:
                return output_path
            else:
                for log in logs:
                    self.log_message(log, info=("Problem" not in log))
                self.log_message(
                    f"Problem replacing nodata value in the snapped layer {input_result_path}, "
                    f"skipping the layer from replacing nodata value."
                )
                return input_result_path
        else:
            self.log_message(
                f"Problem snapping layer {input_path}, "
                f"skipping the layer from replacing nodata value."
            )
            return None
        
    def reproject_layer(
            self, 
            input_path: str,
            target_crs: QgsCoordinateReferenceSystem,
            output_directory: str = None,
            target_extent: str = None
            ) -> str:
        
        """Reprojects the input layer to the target CRS and saves it in the
        specified output directory.
        :param input_path: Input layer path
        :type input_path: str
        :param target_crs: Target CRS to reproject the layer to
        :type target_crs: QgsCoordinateReferenceSystem
        :param output_directory: Directory to save the reprojected layer, defaults to None
        :type output_directory: str, optional
        :param target_extent: Target extent, defaults to None
        :type target_extent: str, optional
        :returns: Path to the reprojected layer
        :rtype: str
        """
        if not os.path.exists(input_path):
            self.log_message(
                f"Input layer {input_path} does not exist, "
                f"skipping the layer reprojection."
            )
            return None
        
        if output_directory is None:
            output_directory = Path(input_path).parent
        
        output_file = os.path.join(
            output_directory,
            f"{Path(input_path).stem}_{str(self.scenario.uuid)[:4]}.tif",
        )

        alg_params = {
            "INPUT": input_path,
            "TARGET_CRS": target_crs,
            "OUTPUT": output_file,
        }

        if target_extent is not None and target_extent != "":
            alg_params["TARGET_EXTENT"] = target_extent

        self.log_message(
            f"Used parameters for layer reprojection: "
            f"{alg_params} \n"
        )

        self.feedback = QgsProcessingFeedback()
        self.feedback.progressChanged.connect(self.update_progress)

        if self.processing_cancelled:
            return None

        results = processing.run(
            "gdal:warpreproject",
            alg_params,
            context=self.processing_context,
            feedback=self.feedback,
        )
        return results["OUTPUT"]

    def reproject_pathways(
            self,            
            target_crs: QgsCoordinateReferenceSystem,
            target_extent: str = None,):
        """Reprojects the activity pathways and priority layers to the target CRS. 
        :param target_crs: Target CRS to reproject the layers to
        :type target_crs: QgsCoordinateReferenceSystem
        :param target_extent: Target extent, defaults to None
        :type target_extent: str, optional
        :returns: True if the task operation was successfully completed else False.
        :rtype: bool
        """
        if self.processing_cancelled:
            return False
        
        if target_crs is None or not target_crs.isValid():
            self.set_info_message(
                tr("Invalid target CRS for reprojecting pathways."),
                level=Qgis.Critical,
            )
            self.log_message("Invalid target CRS for reprojecting pathways.")
            return False
        
        self.set_status_message(
            tr(
                "Reprojecting the activity pathways and priority layers"
            )
        )

        pathways :typing.List[NcsPathway] = []

        try:
            for activity in self.analysis_activities:
                if not activity.pathways and (
                    activity.path is None or activity.path == ""
                ):
                    self.set_info_message(
                        tr(
                            f"No defined activity pathways or "
                            f" activity layers for the activity {activity.name}"
                        ),
                        level=Qgis.Critical,
                    )
                    self.log_message(
                        f"No defined activity pathways or "
                        f"activity layers for the activity {activity.name}"
                    )
                    return False

                for pathway in activity.pathways:
                    if not (pathway in pathways):
                        pathways.append(pathway)

            if pathways is not None and len(pathways) > 0:
                reprojected_pathways_directory = os.path.join(
                    self.scenario_directory, "pathways", "reprojected"
                )

                BaseFileUtils.create_new_dir(reprojected_pathways_directory)

                for pathway in pathways:
                    pathway_layer = QgsRasterLayer(pathway.path, pathway.name)

                    if self.processing_cancelled:
                        return False
                    if not pathway_layer.isValid():
                        self.log_message(
                            f"Pathway layer {pathway.name} is not valid, "
                            f"skipping layer reprojection."
                        )
                        continue

                    if pathway_layer.crs() == target_crs:
                        self.log_message(
                            f"Pathway layer {pathway.name} is already in the target CRS "
                            f"{target_crs.authid()}, skipping layer reprojection."
                        )
                    else:
                        self.log_message(f"Reprojecting {pathway.name} pathway layer to {target_crs.authid()}\n")
                        
                        output_path = self.reproject_layer(
                            pathway.path,
                            target_crs,
                            reprojected_pathways_directory,
                            target_extent
                        )
                        if output_path:
                            pathway.path = output_path

                    self.log_message(
                        f"Reprojecting {len(pathway.priority_layers)} "
                        f"priority weighting layers from pathway {pathway.name}\n"
                    )

                    if (
                        pathway.priority_layers is not None
                        and len(pathway.priority_layers) > 0
                    ):
                        reprojected_priority_directory = os.path.join(
                            self.scenario_directory, "priority_layer", "reprojected"
                        )

                        BaseFileUtils.create_new_dir(reprojected_priority_directory)

                        priority_layers = []
                        for priority_layer in pathway.priority_layers:
                            if priority_layer is None:
                                continue

                            priority_layer_settings = self.get_priority_layer(
                                priority_layer.get("uuid")
                            )
                            if priority_layer_settings is None:
                                continue

                            priority_layer_path = priority_layer_settings.get("path")

                            if not Path(priority_layer_path).exists():
                                priority_layers.append(priority_layer)
                                continue

                            layer = QgsRasterLayer(
                                priority_layer_path, f"{str(uuid.uuid4())[:4]}"
                            )
                            if not layer.isValid():
                                self.log_message(
                                    f"Priority layer {priority_layer.get('name')} "
                                    f"from pathway {pathway.name} is not valid, "
                                    f"skipping layer reprojection."
                                )
                                continue

                            if layer.crs() == target_crs:
                                self.log_message(
                                    f"Priority layer {priority_layer.get('name')} "
                                    f"from pathway {pathway.name} is already in the target CRS "
                                    f"{target_crs.authid()}, skipping layer reprojection."
                                )
                            else:
                                output_path = self.reproject_layer(
                                    priority_layer_path,
                                    target_crs,
                                    reprojected_priority_directory,
                                    target_extent
                                )
                                if output_path:
                                    priority_layer["path"] = output_path

                            priority_layers.append(priority_layer)

                        pathway.priority_layers = priority_layers

        except Exception as e:
            self.log_message(f"Problem reprojecting layers, {e} \n")
            self.log_message(traceback.format_exc())
            self.cancel_task(e)
            return False

        return True

    def run_pathways_replace_nodata(
            self,            
            nodata_value: float = DEFAULT_VALUES.nodata_value
            ) -> bool:
        """Replace the nodata value for activity pathways and priority layers. 
        :param nodata_value: The nodata value to replace in the pathways and priority layers
        :type nodata_value: float        
        :returns: True if the task operation was successfully completed else False.
        :rtype: bool
        """
        if self.processing_cancelled:
            return False        
        
        self.set_status_message(
            tr(
                "Replacing the nodata value for the activity pathways and priority layers"
            )
        )

        pathways :typing.List[NcsPathway] = []

        try:
            # Create directories for replaced nodata pathways and priority layers
            replaced_nodata_pathways_directory = os.path.join(
                self.scenario_directory, "pathways", "replaced_nodata"
            )
            BaseFileUtils.create_new_dir(replaced_nodata_pathways_directory)
            
            replaced_nodata_priority_directory = os.path.join(
                self.scenario_directory, "priority_layer", "replaced_nodata"
            )
            BaseFileUtils.create_new_dir(replaced_nodata_priority_directory)

            for activity in self.analysis_activities:
                if not activity.pathways and (
                    activity.path is None or activity.path == ""
                ):
                    self.set_info_message(
                        tr(
                            f"No defined activity pathways or "
                            f" activity layers for the activity {activity.name}"
                        ),
                        level=Qgis.Critical,
                    )
                    self.log_message(
                        f"No defined activity pathways or "
                        f"activity layers for the activity {activity.name}"
                    )
                    return False

                for pathway in activity.pathways:
                    if not (pathway in pathways):
                        pathways.append(pathway)

            if pathways is not None and len(pathways) > 0:
                for pathway in pathways:
                    pathway_layer = QgsRasterLayer(pathway.path, pathway.name)

                    if self.processing_cancelled:
                        return False
                    if not pathway_layer.isValid():
                        self.log_message(
                            f"Pathway layer {pathway.name} is not valid, "
                            f"skipping replacing nodata value for layer."
                        )
                        continue
                    raster_provider = pathway_layer.dataProvider()
                    raster_no_data_value = raster_provider.sourceNoDataValue(1)
                    if raster_no_data_value == nodata_value:
                        self.log_message(
                            f"Pathway layer {pathway.name} already has the nodata value "
                            f"{nodata_value}, skipping replacing nodata value for layer."
                        )
                    else:
                        self.log_message(
                            f"Replacing nodata value for {pathway.name} pathway layer "
                            f"to {nodata_value}\n"
                        )

                        output_file = os.path.join(
                            replaced_nodata_pathways_directory,
                            f"{Path(pathway.path).stem}_{str(self.scenario.uuid)[:4]}.tif",
                        )
                        
                        result = self.replace_nodata(
                            pathway.path,
                            output_file,
                            nodata_value
                        )
                        if result:
                            pathway.path = output_file

                    self.log_message(
                        f"Replacing nodata value for {len(pathway.priority_layers)} "
                        f"priority weighting layers from pathway {pathway.name}\n"
                    )

                    if (
                        pathway.priority_layers is not None
                        and len(pathway.priority_layers) > 0
                    ):
                        priority_layers = []
                        for priority_layer in pathway.priority_layers:
                            if priority_layer is None:
                                continue

                            priority_layer_settings = self.get_priority_layer(
                                priority_layer.get("uuid")
                            )
                            if priority_layer_settings is None:
                                continue

                            priority_layer_path = priority_layer_settings.get("path")

                            if not Path(priority_layer_path).exists():
                                priority_layers.append(priority_layer)
                                continue

                            layer = QgsRasterLayer(
                                priority_layer_path, f"{str(uuid.uuid4())[:4]}"
                            )
                            if not layer.isValid():
                                self.log_message(
                                    f"Priority layer {priority_layer.get('name')} "
                                    f"from pathway {pathway.name} is not valid, "
                                    f"skipping replacing nodata value for layer."
                                )
                                continue

                            raster_provider = layer.dataProvider()
                            raster_no_data_value = raster_provider.sourceNoDataValue(1)
                            if raster_no_data_value == nodata_value:
                                self.log_message(
                                    f"Priority layer {priority_layer.get('name')} already has the nodata value "
                                    f"{nodata_value}, skipping replacing nodata value for layer."
                                )
                            else:
                                self.log_message(
                                    f"Replacing nodata value for {priority_layer.get('name')} priority layer "
                                    f"to {nodata_value}\n"
                                )

                                output_file = os.path.join(
                                    replaced_nodata_priority_directory,
                                    f"{Path(pathway.path).stem}_{str(self.scenario.uuid)[:4]}.tif",
                                )

                                result = self.replace_nodata(
                                    priority_layer_path,
                                    output_file,
                                    nodata_value
                                )
                                if result:
                                    priority_layer["path"] = output_file

                            priority_layers.append(priority_layer)

                        pathway.priority_layers = priority_layers

        except Exception as e:
            self.log_message(f"Problem replacing nodata value for layers, {e} \n")
            self.log_message(traceback.format_exc())
            self.cancel_task(e)
            return False

        return True
    
    def run_pathways_carbon_summation(self) -> bool:
        """Calculates total carbon mitigation values for the Naturebase pathways.

        :returns: True if the task operation was successfully completed else False.
        :rtype: bool
        """
        if self.processing_cancelled:
            return False

        self.set_status_message(
            tr("Calculating total carbon mitigation values for the Naturebase pathways")
        )

        pathways: typing.List[NcsPathway] = []

        try:
            for activity in self.analysis_activities:
                if not activity.pathways and not activity.path:
                    msg = f"""No defined activity pathways or
                     activity layers for the activity {activity.name}
                    """
                    self.set_info_message(tr(msg), level=Qgis.Critical)
                    self.log_message(msg)
                    return False

                for pathway in activity.pathways:
                    if not (pathway in pathways):
                        pathways.append(pathway)

            if len(pathways) > 0:
                for pathway in pathways:
                    if pathway.name.startswith("Naturebase:") is False:
                        continue

                    pathway_layer = QgsRasterLayer(pathway.path, pathway.name)

                    if self.processing_cancelled:
                        return False
                    if not pathway_layer.isValid():
                        self.log_message(
                            f"Pathway layer {pathway.name} is not valid, "
                            f"skipping calculating total carbon value for layer."
                        )
                        continue
                    
                    raster_provider = pathway_layer.dataProvider()
                    stats = raster_provider.bandStatistics(
                        1, QgsRasterBandStats.Stats.Sum
                    )
                    if stats is None or stats.sum is None:
                        self.log_message(
                            f"Could not calculate statistics for {pathway.name}, skipping."
                        )
                        continue
                    pathway.carbon_impact_value = stats.sum

        except Exception as e:
            self.log_message(f"Problem calculating total carbon values for layers, {e} \n")
            self.cancel_task(e)
            return False
        
        return True
    
    def run_activities_analysis(
        self,
        extent: str,
        temporary_output: bool = False,
    ):
        """Runs the required activity analysis on the passed
        activities pathways. The analysis is responsible for creating
        activities layers from their respective pathways layers.

        :param extent: selected extent from user
        :type extent: SpatialExtent

        :param temporary_output: Whether to save the processing outputs as
        temporary
        files
        :type temporary_output: bool

        :returns: Whether the task operations was successful
        :rtype: bool
        """
        if self.processing_cancelled:
            # Will not proceed if processing has been cancelled by the user
            return False

        self.set_status_message(tr("Creating activity layers from pathways"))

        try:
            for activity in self.analysis_activities:
                activities_directory = os.path.join(
                    self.scenario_directory, "activities"
                )
                BaseFileUtils.create_new_dir(activities_directory)
                file_name = clean_filename(activity.name.replace(" ", "_"))

                layers = []
                if not activity.pathways and (
                    activity.path is None or activity.path == ""
                ):
                    self.set_info_message(
                        tr(
                            f"No defined activity pathways or a"
                            f" activity layer for the activity {activity.name}"
                        ),
                        level=Qgis.Critical,
                    )
                    self.log_message(
                        f"No defined activity pathways or an "
                        f"activity layer for the activity {activity.name}"
                    )

                    return False

                output_file = os.path.join(
                    activities_directory,
                    f"{file_name}_{str(uuid.uuid4())[:4]}.tif"
                )

                # Due to the activities base class
                # activity only one of the following blocks will be executed,
                # the activity either contain a path or
                # pathways

                if activity.path is not None and activity.path != "":
                    layers = [activity.path]

                for pathway in activity.pathways:
                    layers.append(pathway.path)

                output = (
                    QgsProcessing.TEMPORARY_OUTPUT
                    if temporary_output
                    else output_file
                )

                # Actual processing calculation
                reference_layer = self.get_reference_layer()
                if (reference_layer is None or reference_layer == "") and len(layers) > 0:
                    reference_layer = layers[0]                    
                alg_params = {
                    "IGNORE_NODATA": True,
                    "INPUT": layers,
                    "EXTENT": extent,
                    "OUTPUT_NODATA_VALUE": self.get_settings_value(
                        Settings.NCS_NO_DATA_VALUE,
                        default=DEFAULT_VALUES.nodata_value
                    ),
                    "REFERENCE_LAYER": reference_layer  ,
                    "STATISTIC": 0,  # Sum
                    "OUTPUT": output,
                }

                self.log_message(
                    f"Used parameters for activities generation: "
                    f"{alg_params} \n"
                )

                self.feedback = QgsProcessingFeedback()
                self.feedback.progressChanged.connect(self.update_progress)

                if self.processing_cancelled:
                    return False

                results = processing.run(
                    "native:cellstatistics",
                    alg_params,
                    context=self.processing_context,
                    feedback=self.feedback,
                )
                activity.path = results["OUTPUT"]
        except Exception as e:
            self.log_message(f"Problem creating activity layers, {e}")
            self.cancel_task(e)
            return False

        return True

    def run_activities_masking(
        self,
        extent: str, 
        temporary_output: bool = False
    ):
        """Applies the mask layers into the passed activities

        :param extent: selected extent from user
        :type extent: str

        :param temporary_output: Whether to save the processing outputs as temporary files
        :type temporary_output: bool

        :returns: Whether the task operations was successful
        :rtype: bool
        """
        if self.processing_cancelled:
            # Will not proceed if processing has been cancelled by the user
            return False

        self.set_status_message(tr("Masking activities using the saved masked layers"))

        try:
            masked_activities_directory = os.path.join(
                self.scenario_directory, "masked_activities"
            )
            BaseFileUtils.create_new_dir(masked_activities_directory)
            for activity in self.analysis_activities:
                if activity.path is None or activity.path == "":
                    if not self.processing_cancelled:
                        self.set_info_message(
                            tr(
                                f"Problem when masking activities, "
                                f"there is no map layer for the activity {activity.name}"
                            ),
                            level=Qgis.Critical,
                        )
                        self.log_message(
                            f"Problem when masking activities, "
                            f"there is no map layer for the activity {activity.name}"
                        )
                    else:
                        # If the user cancelled the processing
                        self.set_info_message(
                            tr(f"Processing has been cancelled by the user."),
                            level=Qgis.Critical,
                        )
                        self.log_message(f"Processing has been cancelled by the user.")

                    return False
                
                masking_layers = activity.mask_paths

                if len(masking_layers) < 1:
                    continue

                if len(masking_layers) > 1:
                    mask_layer_path = self.merge_vector_layers(masking_layers)
                else:
                    mask_layer_path = masking_layers[0]

                initial_mask_layer = QgsVectorLayer(mask_layer_path, "mask", "ogr")

                if not initial_mask_layer.isValid():
                    self.log_message(
                        f"Skipping activities masking "
                        f"using layer {mask_layer_path}, not a valid layer."
                    )
                    continue

                # see https://qgis.org/pyqgis/master/core/Qgis.html#qgis.core.Qgis.GeometryType
                if Qgis.versionInt() < 33000:
                    layer_check = initial_mask_layer.geometryType() == QgsWkbTypes.Polygon
                    layer_check = (
                        initial_mask_layer.geometryType() == QgsWkbTypes.PolygonGeometry
                    )
                else:
                    layer_check = (
                        initial_mask_layer.geometryType() == Qgis.GeometryType.Polygon
                    )

                if not layer_check:
                    self.log_message(
                        f"Skipping activities masking "
                        f"using layer {mask_layer_path}, not a polygon layer."
                    )
                    return False

                extent_layer = self.layer_extent(extent)

                if extent_layer.crs() != initial_mask_layer.crs():                    
                    reprojected_mask_path = self.reproject_mask_layer(
                        input_path=mask_layer_path,
                        output_directory=masked_activities_directory,
                        target_crs=extent_layer.crs()
                    )
                    if reprojected_mask_path is not None:
                        initial_mask_layer = QgsVectorLayer(reprojected_mask_path, "ogr")
                    else:
                        self.log_message(
                            f"Skipping masking, the mask layers crs ({mask_layer.crs().authid()})"
                            f" do not match the scenario crs ({extent_layer.crs().authid()})."
                        )
                        return False
                    
                mask_layer = self.mask_layer_difference(initial_mask_layer, extent_layer)

                if isinstance(mask_layer, str):
                    mask_layer = QgsVectorLayer(mask_layer, "ogr")

                if not mask_layer.isValid():
                    self.log_message(
                        f"Skipping activities masking "
                        f"the created difference mask layer {mask_layer.source()},"
                        f" not a valid layer."
                    )
                    return False
                

                if not extent_layer.extent().intersects(mask_layer.extent()):
                    self.log_message(
                        "Skipping masking, the mask layers extent"
                        " and the scenario extent do not overlap."
                    )
                    return False

                
                file_name = clean_filename(activity.name.replace(" ", "_"))

                output_file = os.path.join(
                    masked_activities_directory,
                    f"{file_name}_{str(uuid.uuid4())[:4]}.tif",
                )

                output = (
                    QgsProcessing.TEMPORARY_OUTPUT if temporary_output else output_file
                )

                activity_layer = QgsRasterLayer(activity.path, "activity_layer")

                # Actual processing calculation
                alg_params = {
                    "INPUT": activity.path,
                    "MASK": mask_layer,
                    "SOURCE_CRS": activity_layer.crs(),
                    "DESTINATION_CRS": activity_layer.crs(),
                    "TARGET_EXTENT": extent,
                    "OUTPUT": output,
                    "NO_DATA": self.get_settings_value(
                        Settings.NCS_NO_DATA_VALUE,
                        default=DEFAULT_VALUES.nodata_value
                    ),
                }

                self.log_message(
                    f"Used parameters for masking the activities: {alg_params} \n"
                )

                self.feedback = QgsProcessingFeedback()
                self.feedback.progressChanged.connect(self.update_progress)

                if self.processing_cancelled:
                    return False

                results = processing.run(
                    "gdal:cliprasterbymasklayer",
                    alg_params,
                    context=self.processing_context,
                    feedback=self.feedback,
                )
                activity.path = results["OUTPUT"]

        except Exception as e:
            self.log_message(f"Problem masking activities layers, {e} \n")
            self.cancel_task(e)
            return False

        return True

    def run_internal_activities_masking(
        self,
        extent: str,
        temporary_output: bool = False
    ):
        """Applies the mask layers into the passed activities

        :param extent: selected extent from user
        :type extent: str

        :param temporary_output: Whether to save the processing outputs as temporary
        files
        :type temporary_output: bool

        :returns: Whether the task operations was successful
        :rtype: bool
        """
        if self.processing_cancelled:
            # Will not proceed if processing has been cancelled by the user
            return False

        self.set_status_message(
            tr("Masking activities using their respective mask layers.")
        )

        try:
            masked_activities_directory = os.path.join(
                self.scenario_directory, "final_masked_activities"
            )
            BaseFileUtils.create_new_dir(masked_activities_directory)
            
            for activity in self.analysis_activities:
                masking_layers = activity.mask_paths

                if len(masking_layers) < 1:
                    self.log_message(
                        f"Skipping internal activity masking "
                        f"No mask layer(s) for activity {activity.name}"
                    )
                    continue
                if len(masking_layers) > 1:
                    mask_layer_path = self.merge_vector_layers(masking_layers)
                else:
                    mask_layer_path = masking_layers[0]
                initial_mask_layer = QgsVectorLayer(mask_layer_path, "mask", "ogr")

                if not initial_mask_layer.isValid():
                    self.log_message(
                        f"Skipping activity masking "
                        f"using layer {mask_layer_path}, not a valid layer."
                    )
                    continue

                # see https://qgis.org/pyqgis/master/core/Qgis.html#qgis.core.Qgis.GeometryType
                if Qgis.versionInt() < 33000:
                    layer_check = (
                        initial_mask_layer.geometryType() == QgsWkbTypes.PolygonGeometry
                    )
                else:
                    layer_check = (
                        initial_mask_layer.geometryType() == Qgis.GeometryType.Polygon
                    )

                if not layer_check:
                    self.log_message(
                        f"Skipping activity masking "
                        f"using layer {mask_layer_path}, not a polygon layer."
                    )
                    continue

                extent_layer = self.layer_extent(extent)

                if extent_layer.crs() != initial_mask_layer.crs():
                    reprojected_mask = self.reproject_mask_layer(
                        input_path=mask_layer_path,
                        output_directory=masked_activities_directory,
                        target_crs=extent_layer.crs()
                    )
                    if reprojected_mask is not None:
                        initial_mask_layer = QgsVectorLayer(reprojected_mask, "ogr")
                    else:                        
                        self.log_message(
                            f"Skipping masking, the mask layers crs ({initial_mask_layer.crs().authid()})"
                            f" do not match the scenario crs ({extent_layer.crs().authid()})."
                        )
                        continue

                if not extent_layer.extent().intersects(initial_mask_layer.extent()):
                    self.log_message(
                        "Skipping masking, the mask layers extent"
                        " and the scenario extent do not overlap."
                    )
                    continue

                mask_layer = self.mask_layer_difference(
                    initial_mask_layer, extent_layer
                )

                if isinstance(mask_layer, str):
                    mask_layer = QgsVectorLayer(mask_layer, "ogr")

                if not mask_layer.isValid():
                    self.log_message(
                        f"Skipping activity masking "
                        f"the created difference mask layer {mask_layer.source()},"
                        f"is not a valid layer."
                    )
                    continue
                if activity.path is None or activity.path == "":
                    if not self.processing_cancelled:
                        self.set_info_message(
                            tr(
                                f"Problem when masking activity, "
                                f"there is no map layer for the activity {activity.name}"
                            ),
                            level=Qgis.Critical,
                        )
                        self.log_message(
                            f"Problem when masking activity, "
                            f"there is no map layer for the activity {activity.name}"
                        )
                    else:
                        # If the user cancelled the processing
                        self.set_info_message(
                            tr(f"Processing has been cancelled by the user."),
                            level=Qgis.Critical,
                        )
                        self.log_message(f"Processing has been cancelled by the user.")

                    continue

                file_name = clean_filename(activity.name.replace(" ", "_"))

                output_file = os.path.join(
                    masked_activities_directory,
                    f"{file_name}_{str(uuid.uuid4())[:4]}.tif",
                )

                output = (
                    QgsProcessing.TEMPORARY_OUTPUT if temporary_output else output_file
                )

                activity_layer = QgsRasterLayer(activity.path, "activity_layer")

                if activity_layer.crs() != mask_layer.crs():
                    self.log_message(
                        f"Skipping masking, activity layer and"
                        f" mask layer(s) have different CRS"
                    )
                    continue

                if not activity_layer.extent().intersects(mask_layer.extent()):
                    self.log_message(
                        "Skipping masking, the extents of the activity layer "
                        "and mask layers do not overlap."
                    )
                    continue

                # Actual processing calculation
                alg_params = {
                    "INPUT": activity.path,
                    "MASK": mask_layer,
                    "SOURCE_CRS": activity_layer.crs(),
                    "DESTINATION_CRS": activity_layer.crs(),
                    "TARGET_EXTENT": extent,
                    "OUTPUT": output,
                    "NO_DATA": self.get_settings_value(
                        Settings.NCS_NO_DATA_VALUE,
                        default=DEFAULT_VALUES.nodata_value
                    ),
                }

                self.log_message(
                    f"Used parameters for masking the activity {activity.name}: {alg_params} \n"
                )

                self.feedback = QgsProcessingFeedback()
                self.feedback.progressChanged.connect(self.update_progress)

                if self.processing_cancelled:
                    return False

                results = processing.run(
                    "gdal:cliprasterbymasklayer",
                    alg_params,
                    context=self.processing_context,
                    feedback=self.feedback,
                )
                activity.path = results["OUTPUT"]

        except Exception as e:
            self.log_message(f"Problem masking activities layers, {e} \n")
            self.cancel_task(e)

            return False

        return True

    def merge_vector_layers(self, layers):
        """Merges the passed vector layers into a single layer

        :param layers: List of the vector layers paths
        :type layers: typing.List[str]

        :return: Merged vector layer
        :rtype: QgsMapLayer
        """

        input_map_layers = []

        for layer_path in layers:
            layer = QgsVectorLayer(layer_path, "mask", "ogr")
            if layer.isValid():
                input_map_layers.append(layer)
            else:
                self.log_message(
                    f"Skipping invalid mask layer {layer_path} from masking."
                )
        if len(input_map_layers) == 0:
            return None
        if len(input_map_layers) == 1:
            return input_map_layers[0].source()

        self.set_status_message(tr("Merging mask layers"))

        # Actual processing calculation
        alg_params = {
            "LAYERS": input_map_layers,
            "CRS": None,
            "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
        }

        self.log_message(f"Used parameters for merging mask layers: {alg_params} \n")

        results = processing.run(
            "native:mergevectorlayers",
            alg_params,
            context=self.processing_context,
            feedback=self.feedback,
        )

        return results["OUTPUT"]

    def layer_extent(self, extent):
        """Creates a new vector layer contains has a
        feature with geometry matching an extent parameter.
        :param extent: Extent parameter
        :type extent: str
        :returns: Vector layer
        :rtype: QgsVectorLayer
        """

        alg_params = {
            "INPUT": extent,
            "CRS": None,
            "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
        }

        results = processing.run(
            "native:extenttolayer",
            alg_params,
            context=self.processing_context,
            feedback=self.feedback,
        )

        return results["OUTPUT"]

    def mask_layer_difference(self, input_layer, overlay_layer):
        """Creates a new vector layer that contains
         difference of features between the two passed layers.
        :param input_layer: Input layer
        :type input_layer: QgsVectorLayer
        :param overlay_layer: Target overlay layer
        :type overlay_layer: QgsVectorLayer
        :returns: Vector layer
        :rtype: QgsVectorLayer
        """

        alg_params = {
            "INPUT": input_layer,
            "OVERLAY": overlay_layer,
            "OVERLAY_FIELDS_PREFIX": "",
            "GRID_SIZE": None,
            "OUTPUT": QgsProcessing.TEMPORARY_OUTPUT,
        }

        results = processing.run(
            "native:symmetricaldifference",
            alg_params,
            context=self.processing_context,
            feedback=self.feedback,
        )

        return results["OUTPUT"]
    
    def reproject_mask_layer(
            self, 
            input_path: str, 
            output_directory: str, 
            target_crs: QgsCoordinateReferenceSystem
        ):
        """Reprojects the input layer to the target CRS and saves it to the output path.
        :param input_path: Input layer path
        :type input_path: str
        :param output_directory: Output directory path
        :type output_directory: str
        :param target_crs: Target CRS to reproject to
        :type target_crs: QgsCoordinateReferenceSystem
        :returns: Reprojected layer path
        :rtype: str
        """
        if output_directory is None:
            output_directory = Path(input_path).parent
        output_path = os.path.join(
            f"{output_directory}",
            f"{Path(input_path).stem}_reprojected_{str(uuid.uuid4())[:4]}.tif"
        )

        alg_params = {
            'INPUT': input_path,
            'TARGET_CRS': target_crs,
            'CONVERT_CURVED_GEOMETRIES': False,
            'OPERATION': None,
            'OUTPUT': output_path
        }
        results = processing.run(
            "native:reprojectlayer", 
            alg_params,
            context=self.processing_context,
            feedback=self.feedback,
        )
        return results["OUTPUT"]

    def run_activities_sieve(self, models, temporary_output=False):
        """Runs the sieve functionality analysis on the passed models layers,
        removing the models layer polygons that are smaller than the provided
        threshold size (in pixels) and replaces them with the pixel value of
        the largest neighbour polygon.

        :param models: List of the analyzed activities
        :type models: typing.List[ImplementationModel]

        :param extent: Selected area of interest extent
        :type extent: str

        :param temporary_output: Whether to save the processing outputs as temporary
        files
        :type temporary_output: bool

        :returns: Whether the task operations was successful
        :rtype: bool
        """
        if self.processing_cancelled:
            # Will not proceed if processing has been cancelled by the user
            return False

        self.set_status_message(tr("Applying sieve function to the activities"))

        try:
            for model in models:
                if model.path is None or model.path == "":
                    if not self.processing_cancelled:
                        self.set_info_message(
                            tr(
                                f"Problem when running sieve function on models, "
                                f"there is no map layer for the model {model.name}"
                            ),
                            level=Qgis.Critical,
                        )
                        self.log_message(
                            f"Problem when running sieve function on models, "
                            f"there is no map layer for the model {model.name}"
                        )
                    else:
                        # If the user cancelled the processing
                        self.set_info_message(
                            tr(f"Processing has been cancelled by the user."),
                            level=Qgis.Critical,
                        )
                        self.log_message(f"Processing has been cancelled by the user.")

                    return False

                sieved_ims_directory = os.path.join(
                    self.scenario_directory, "sieved_ims"
                )
                BaseFileUtils.create_new_dir(sieved_ims_directory)
                file_name = clean_filename(model.name.replace(" ", "_"))

                output_file = os.path.join(
                    sieved_ims_directory, f"{file_name}_{str(uuid.uuid4())[:4]}.tif"
                )

                threshold_value = float(
                    self.get_settings_value(Settings.SIEVE_THRESHOLD, default=10.0)
                )

                mask_layer = self.get_settings_value(
                    Settings.SIEVE_MASK_PATH, default=""
                )

                output = (
                    QgsProcessing.TEMPORARY_OUTPUT if temporary_output else output_file
                )

                # Actual processing calculation
                alg_params = {
                    "INPUT": model.path,
                    "THRESHOLD": threshold_value,
                    "MASK_LAYER": mask_layer,
                    "OUTPUT": output,
                }

                self.log_message(f"Used parameters for sieving: {alg_params} \n")

                input_name = os.path.splitext(os.path.basename(model.path))[0]

                # Step 1: Create a binary mask from the original raster
                binary_mask = processing.run(
                    "qgis:rastercalculator",
                    {
                        "CELLSIZE": 0,
                        "LAYERS": [model.path],
                        "CRS": None,
                        "EXPRESSION": f"{input_name}@1 > 0",
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                )["OUTPUT"]

                # feedback.pushInfo(f"binary mask {binary_mask}")

                # binary_mask_layer = QgsRasterLayer(binary_mask, 'binary')

                # QgsProject.instance().addMapLayer(binary_mask_layer)

                # Step 2: Run sieve analysis from on the binary mask
                sieved_mask = processing.run(
                    "gdal:sieve",
                    {
                        "INPUT": binary_mask,
                        "THRESHOLD": threshold_value,
                        "EIGHT_CONNECTEDNESS": True,
                        "NO_MASK": True,
                        "MASK_LAYER": None,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context=self.processing_context,
                    feedback=self.feedback,
                )["OUTPUT"]

                # feedback.pushInfo(f"sieved mask {sieved_mask}")

                # sieved_mask_layer = QgsRasterLayer(sieved_mask, 'sieved_mask')

                # QgsProject.instance().addMapLayer(sieved_mask_layer)

                expr = f"({os.path.splitext(os.path.basename(sieved_mask))[0]}@1 > 0) * {os.path.splitext(os.path.basename(sieved_mask))[0]}@1"
                # feedback.pushInfo(f"used expression {expr}")

                # Step 3: Remove and convert any no data value to 0
                sieved_mask_clean = processing.run(
                    "qgis:rastercalculator",
                    {
                        "CELLSIZE": 0,
                        "LAYERS": [sieved_mask],
                        "CRS": None,
                        "EXPRESSION": expr,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context=self.processing_context,
                    feedback=self.feedback,
                )["OUTPUT"]

                # feedback.pushInfo(f"sieved mask clean {sieved_mask_clean}")

                # sieved_mask_clean_layer = QgsRasterLayer(sieved_mask_clean, 'sieved_mask_clean')

                # QgsProject.instance().addMapLayer(sieved_mask_clean_layer)

                expr_2 = f"{input_name}@1 * {os.path.splitext(os.path.basename(sieved_mask_clean))[0]}@1"

                # feedback.pushInfo(f"Used expression 2 {expr_2}")

                # Step 4: Join the sieved mask with the original input layer to filter out the small areas
                sieve_output = processing.run(
                    "qgis:rastercalculator",
                    {
                        "CELLSIZE": 0,
                        "LAYERS": [model.path, sieved_mask_clean],
                        "CRS": None,
                        "EXPRESSION": expr_2,
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context=self.processing_context,
                    feedback=self.feedback,
                )["OUTPUT"]

                # feedback.pushInfo(f"sieved output joined {sieve_output}")

                # sieve_output_layer = QgsRasterLayer(sieve_output, 'sieve_output')

                # QgsProject.instance().addMapLayer(sieve_output_layer)

                # expr_3 = f'if ( {os.path.splitext(os.path.basename(sieve_output))[0]}@1 <= 0, -9999, {os.path.splitext(os.path.basename(sieve_output))[0]}@1 )'

                # feedback.pushInfo(f"used expression 3 {expr_3}")

                # Step 5. Replace all 0 with -9999 using if ("combined@1" <= 0, -9999, "combined@1")

                self.feedback = QgsProcessingFeedback()
                self.feedback.progressChanged.connect(self.update_progress)

                if self.processing_cancelled:
                    return False
                
                sieve_output_updated = processing.run(
                    "gdal:rastercalculator",
                    {
                        "INPUT_A": f"{sieve_output}",
                        "BAND_A": 1,
                        "FORMULA": "9999*(A<=0)*(-1)+A*(A>0)",
                        "NO_DATA": None,
                        "EXTENT_OPT": 0,
                        "PROJWIN": None,
                        "RTYPE": 5,
                        "OPTIONS": "",
                        "EXTRA": "",
                        "OUTPUT": "TEMPORARY_OUTPUT",
                    },
                    context=self.processing_context,
                    feedback=self.feedback,
                )["OUTPUT"]

                # feedback.pushInfo(f"sieved output updated {sieve_output_updated}")

                # sieve_output_updated_layer = QgsRasterLayer(sieve_output_updated, 'sieve_output_updated')

                # QgsProject.instance().addMapLayer(sieve_output_updated_layer)

                # Step 6. Run sum statistics with ignore no data values set to false and no data value of -9999
                
                results = processing.run(
                    "native:cellstatistics",
                    {
                        "INPUT": [sieve_output_updated],
                        "STATISTIC": 0,
                        "IGNORE_NODATA": False,
                        "REFERENCE_LAYER": sieve_output_updated,
                        "OUTPUT_NODATA_VALUE": self.get_settings_value(
                            Settings.NCS_NO_DATA_VALUE,
                            default=DEFAULT_VALUES.nodata_value
                        ),
                        "OUTPUT": output,
                    },
                    context=self.processing_context,
                    feedback=self.feedback,
                )

                # self.log_message(
                #     f"Used parameters for running sieve function to the models: {alg_params} \n"
                # )                

                model.path = results["OUTPUT"]

        except Exception as e:
            self.log_message(f"Problem running sieve function on models layers, {e} \n")
            self.cancel_task(e)
            return False

        return True

    def run_activities_cleaning(
        self,
        extent: str,
        temporary_output: bool = False
    ):
        """Cleans the weighted activities replacing
        zero values with no-data as they are not statistical meaningful for the
        scenario analysis.

        :param extent: Selected extent from user
        :type extent: str

        :param temporary_output: Whether to save the processing outputs as temporary
        files
        :type temporary_output: bool

        :returns: Whether the task operations was successful
        :rtype: bool
        """

        if self.processing_cancelled:
            return False

        self.set_status_message(tr("Updating weighted activity values"))


        try:
            for activity in self.analysis_activities:
                if activity.path is None or activity.path == "":
                    self.log_message(
                        f"Problem when running activity updates, "
                        f"there is no map layer for the activity {activity.name}"
                    )

                    return False
                
                if not os.path.exists(activity.path):
                    self.log_message(
                        f"Problem when running activity updates, "
                        f"the map layer for the activity {activity.name} does not exist"
                    )
                    return False

                layer = QgsRasterLayer(activity.path, "activity_layer")
                if not layer.isValid():                    
                    self.log_message(
                        f"Problem when running activity updates, "
                        f"the map layer for the activity {activity.name} is not valid"
                    )
                    return False
                file_name = clean_filename(activity.name.replace(" ", "_"))

                weighted_pathways_dir = os.path.join(
                    self.scenario_directory, "weighted_activities"
                )

                BaseFileUtils.create_new_dir(weighted_pathways_dir)
                
                output_file = os.path.join(
                    weighted_pathways_dir, f"{file_name}_{str(uuid.uuid4())[:4]}_cleaned.tif"
                )

                # Actual processing calculation
                # The aim is to convert pixels values to no data, that is why we are
                # using the sum operation with only one layer.

                output = (
                    QgsProcessing.TEMPORARY_OUTPUT if temporary_output else output_file
                )
                reference_layer = self.get_reference_layer()
                if (reference_layer is None or reference_layer == ""):
                    reference_layer = activity.path  

                alg_params = {
                    "IGNORE_NODATA": True,
                    "INPUT": [activity.path],
                    "EXTENT": extent,
                    "OUTPUT_NODATA_VALUE": 0,
                    "REFERENCE_LAYER": reference_layer,
                    "STATISTIC": 0,  # Sum
                    "OUTPUT": output,
                }

                self.log_message(
                    f"Used parameters for "
                    f"updates on the weighted activities: {alg_params} \n"
                )

                self.feedback = QgsProcessingFeedback()
                self.feedback.progressChanged.connect(self.update_progress)

                if self.processing_cancelled:
                    return False

                results = processing.run(
                    "native:cellstatistics",
                    alg_params,
                    context=self.processing_context,
                    feedback=self.feedback,
                )
                activity.path = results["OUTPUT"]

        except Exception as e:
            self.log_message(f"Problem cleaning activities, {e}")
            self.log_message(traceback.format_exc())
            self.cancel_task(e)
            return False

        return True
    
    def create_activity_connectivity_layer(self, activity: Activity):
        """Create an activity connectivity layer for investability analysis

        :param activity: Activity
        :type activity: Activity

        :returns: The path to the connectivity layer or None if the process failed
        :rtype: str | None
        """
        if not activity.path:
            self.log_message(
                f"Problem when creating the connectivity layer, "
                f"there is no map layer for the activity {activity.name}"
            )
            return None
                
        if not os.path.exists(activity.path):
            self.log_message(
                f"Problem when creating the connectivity layer, "
                f"the map layer for the activity {activity.name} does not exist"
            )
            return None
        
        self.set_status_message(
            tr(f"Creating connectivity layer for the activity: {activity.name}")
        )

        output_directory = os.path.join(
            self.scenario_directory, "investable_activities"
        )
        BaseFileUtils.create_new_dir(output_directory)

        output_path = os.path.join(
                output_directory,
                f"{Path(activity.path).stem}_connectivity_{str(uuid.uuid4())[:4]}.tif"
            )
        
        try:                        
            if self.processing_cancelled:
                return None
            
            self.feedback = QgsProcessingFeedback()
            self.feedback.progressChanged.connect(self.update_progress)
            
            # 1. Creating a binary raster
            binary = processing.run(
                "qgis:rastercalculator",
                {
                    "CELLSIZE": 0,
                    "LAYERS": [activity.path],
                    "CRS": None,
                    "EXPRESSION": f"{Path(activity.path).stem}@1 > 0",
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context=self.processing_context,
                feedback=self.feedback,
            )["OUTPUT"]

            # 2. Polygonize the binary to get polygon clusters
            binary_polygonize = processing.run(
                "gdal:polygonize",
                {
                    "INPUT": binary,
                    "BAND": 1,
                    "FIELD": "DN",
                    "EIGHT_CONNECTEDNESS": True,
                    "EXTRA": "",
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context=self.processing_context,
                feedback=self.feedback,
            )["OUTPUT"]

            if self.processing_cancelled:
                return None

            # 3. Create a zonal statistics to find the number of pixels in each cluster
            zonal_statistics = processing.run(
                "native:zonalstatisticsfb",
                {
                    "INPUT": QgsProcessingFeatureSourceDefinition(
                        binary_polygonize,
                        selectedFeaturesOnly=False,
                        featureLimit=-1,
                        flags=QgsProcessingFeatureSourceDefinition.FlagOverrideDefaultGeometryCheck
                        | QgsProcessingFeatureSourceDefinition.FlagCreateIndividualOutputPerInputFeature,
                        geometryCheck=QgsFeatureRequest.GeometryNoCheck,
                    ),
                    "INPUT_RASTER": binary,
                    "RASTER_BAND": 1,
                    "COLUMN_PREFIX": "_",
                    "STATISTICS": [1],  # sum
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context=self.processing_context,
                feedback=self.feedback,
            )["OUTPUT"]

            # 4. Caculate connectivity score = count of pixels in cluster * compactness
            score = processing.run(
                "native:fieldcalculator",
                {
                    "INPUT": zonal_statistics,
                    "FIELD_NAME": "score",
                    "FIELD_TYPE": 0,
                    "FIELD_LENGTH": 0,
                    "FIELD_PRECISION": 0,
                    "FORMULA": '"_sum"  * 4* pi() *  $area  /($perimeter * $perimeter)',
                    "OUTPUT": "TEMPORARY_OUTPUT",
                },
                context=self.processing_context,
                feedback=self.feedback,
            )["OUTPUT"]

            if self.processing_cancelled:
                return None

            # 5. Rasterize the score vector
            processing.run(
                "gdal:rasterize_over",
                {
                    "INPUT": score,
                    "INPUT_RASTER": binary,
                    "FIELD": "score",
                    "ADD": True,
                    "EXTRA": "",
                },
                context=self.processing_context,
                feedback=self.feedback,
            )

            # 6. Normalize the raster
            ok, message = normalize_raster(
                binary, output_path, self.processing_context, self.feedback
            )

            self.log_message(message)

            if ok and os.path.exists(output_path):
                return output_path
            
            self.log_message(
                f" Error creating the connectivity layer for activity {activity.name}, "
            )
            
            return None

        except Exception as e:
            self.log_message(f"Problem creating connectivity layer for activity, {e} \n")
            self.log_message(traceback.format_exc())
            self.cancel_task(e)      
        return None

    def run_investability_analysis(self) -> bool:
        """Run activity investability analysis

        :returns: True if the task operation was successfully completed else False.
        :rtype: bool
        """
        if self.processing_cancelled:
            return False

        self.set_status_message(
            tr("Calculating investability of the activities")
        )

        investable_activities = os.path.join(
            self.scenario_directory, "investable_activities"
        )
        BaseFileUtils.create_new_dir(investable_activities)

        self.feedback = QgsProcessingFeedback()
        self.feedback.progressChanged.connect(self.update_progress)

        try:
            for activity in self.analysis_activities:
                if activity.path is None or activity.path == "":
                    self.log_message(
                        f"Problem when running activity investability, "
                        f"there is no map layer for the activity {activity.name}"
                    )

                    return False
                
                if not os.path.exists(activity.path):
                    self.log_message(
                        f"Problem when running activity investability, "
                        f"the map layer for the activity {activity.name} does not exist"
                    )
                    return False
                
                layers = [activity.path]

                activity_basename = Path(activity.path).stem
                expression_items = [f'("{activity_basename}@1")']
                
                constant_rasters = activity.constant_rasters
                if constant_rasters is None:
                    constant_rasters = []
                
                if self.processing_cancelled:
                    return False
                
                if self.task_config.pixel_connectivity_enabled:
                    # Add connectivity layer
                    connectivity_path = self.create_activity_connectivity_layer(activity=activity)
                    if connectivity_path and os.path.exists(connectivity_path):
                        constant_rasters.append(
                            {
                                "path": connectivity_path,
                                "name": "Connectivity layer",
                                "skip_raster": False
                            }
                        )
                    else:
                        self.log_message(f"Invalid path for connectivity layer of activity {activity.name}")
                
                nr_constant_rasters = len(constant_rasters)
                if nr_constant_rasters == 0:
                    self.log_message(
                        f"No defined constant rasters, "
                        f"Skipping investability analysis for the activity {activity.name}"
                    )
                    continue                

                for constant_raster in constant_rasters:
                    if "normalized" in constant_raster:
                        expression_items.append(
                            str(constant_raster.get('normalized') / nr_constant_rasters)
                        )
                    else:
                        path = constant_raster.get("path", "")
                        if not os.path.exists(path):
                            self.log_message(
                                f"Invalid constant raster path {path},"
                                f"Skipping from the investability analysis for the activity {activity.name}"
                            )
                            continue

                        normalized_path = os.path.join(
                            f"{investable_activities}",
                            f"{Path(path).stem}_norm_{str(uuid.uuid4())[:4]}.tif"
                        )

                        if self.processing_cancelled:
                            return False

                        ok, log = normalize_raster(
                            input_raster_path=path,
                            output_raster_path=normalized_path,
                            processing_context=self.processing_context,
                            feedback=self.feedback,
                        )
                        self.log_message(log)
                        if not ok:                            
                            self.log_message(
                                f"Skipping {path} from the investability analysis for the activity {activity.name}"
                            )
                            continue

                        if os.path.exists(normalized_path):
                            path = normalized_path

                        layers.append(path)
                        expression_items.append(f'("{Path(path).stem}@1" / {nr_constant_rasters})')                       
                
                output_path = os.path.join(
                    investable_activities,
                    f"{Path(activity.path).stem}_invest_{str(uuid.uuid4())[:4]}.tif"
                )

                alg_params = {
                    "CELLSIZE": 0,
                    "CRS": None,
                    "EXPRESSION": " + ".join(expression_items),
                    "LAYERS": layers,
                    "OUTPUT": output_path,
                }

                self.log_message(
                    f" Used parameters for calculating investability for activity {activity.name}, "
                    f"{alg_params} \n"
                )

                if self.processing_cancelled:
                    return False

                result = processing.run(
                    "qgis:rastercalculator",
                    alg_params,
                    context=self.processing_context,
                    feedback=self.feedback,
                )

                if result.get("OUTPUT"):
                    activity.path = result.get("OUTPUT")
                else:
                    self.log_message(
                        f"Problem calculating investability for activity {activity.name}"
                    )                
        except Exception as e:
            self.log_message(f"Problem calculating activity investability, {e} \n")
            self.log_message(traceback.format_exc())
            self.cancel_task(e)
            return False
        
        return True

    def run_highest_position_analysis(self, temporary_output: bool =False):
        """Runs the highest position analysis which is last step
        in scenario analysis. Uses the activities set by the current ongoing
        analysis.

        :param temporary_output: Whether to save the processing outputs as temporary
        files
        :type temporary_output: bool

        :returns: Whether the task operations was successful
        :rtype: bool

        """
        if self.processing_cancelled:
            # Will not proceed if processing has been cancelled by the user
            return False

        passed_extent_box = self.analysis_extent.bbox
        passed_extent = QgsRectangle(
            passed_extent_box[0],
            passed_extent_box[2],
            passed_extent_box[1],
            passed_extent_box[3],
        )

        # We explicitly set the created_date since the current implementation
        # of the data model means that the attribute value is set only once when
        # the class is loaded hence subsequent instances will have the same value.
        self.scenario_result = ScenarioResult(
            scenario=self.scenario,
            scenario_directory=self.scenario_directory,
            created_date=datetime.datetime.now(),
        )

        try:
            layers = {}

            self.set_status_message(tr("Calculating the highest position"))

            for activity in self.analysis_activities:
                if activity.path is not None and activity.path != "":
                    raster_layer = QgsRasterLayer(activity.path, activity.name)
                    layers[activity.name] = (
                        raster_layer if raster_layer is not None else None
                    )
                else:
                    for pathway in activity.pathways:
                        layers[activity.name] = QgsRasterLayer(pathway.path)

            source_crs = QgsCoordinateReferenceSystem("EPSG:4326")
            dest_crs = list(layers.values())[0].crs() if len(layers) > 0 else source_crs

            extent_string = (
                f"{passed_extent.xMinimum()},{passed_extent.xMaximum()},"
                f"{passed_extent.yMinimum()},{passed_extent.yMaximum()}"
                f" [{dest_crs.authid()}]"
            )

            output_file = os.path.join(
                self.scenario_directory,
                f"{SCENARIO_OUTPUT_FILE_NAME}_{str(self.scenario.uuid)[:4]}.tif",
            )

            # Preparing the input rasters for the highest position
            # analysis in a correct order
            activity_names = [activity.name for activity in self.analysis_activities]
            all_activities = sorted(
                self.analysis_activities,
                key=lambda activity_instance: activity_instance.style_pixel_value,
            )
            for index, activity in enumerate(all_activities):
                activity.style_pixel_value = index + 1

            all_activity_names = [activity.name for activity in all_activities]
            sources = []

            for activity_name in all_activity_names:
                if activity_name in activity_names:
                    sources.append(layers[activity_name].source())

            self.log_message(
                f"Layers sources {[Path(source).stem for source in sources]}"
            )

            output_file = (
                QgsProcessing.TEMPORARY_OUTPUT if temporary_output else output_file
            )

            reference_layer = self.get_reference_layer()
            if (reference_layer is None or reference_layer == "") and len(layers) > 0:
                reference_layer = list(layers.values())[0] 

            alg_params = {
                "IGNORE_NODATA": True,
                "INPUT_RASTERS": sources,
                "EXTENT": extent_string,
                "OUTPUT_NODATA_VALUE": self.get_settings_value(
                    Settings.NCS_NO_DATA_VALUE,
                    default=DEFAULT_VALUES.nodata_value
                ),
                "REFERENCE_LAYER": reference_layer,
                "OUTPUT": output_file,
            }

            self.log_message(
                f"Used parameters for highest position analysis {alg_params} \n"
            )

            self.feedback = QgsProcessingFeedback()
            self.feedback.progressChanged.connect(self.update_progress)

            if self.processing_cancelled:
                return False

            self.output = processing.run(
                "native:highestpositioninrasterstack",
                alg_params,
                context=self.processing_context,
                feedback=self.feedback,
            )

        except Exception as err:
            self.log_message(
                tr(
                    "An error occurred when running task for "
                    'scenario analysis, error message "{}"'.format(str(err))
                )
            )
            self.log_message(traceback.format_exc())
            self.cancel_task(err)
            return False

        return True
