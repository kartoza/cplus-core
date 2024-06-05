# -*- coding: utf-8 -*-
"""
    Plugin utilities
"""


import os
import json
import uuid
import datetime
from pathlib import Path
from uuid import UUID

from qgis.PyQt import QtCore
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsDistanceArea,
    QgsMessageLog,
    QgsProcessingFeedback,
    QgsProject,
    QgsProcessing,
    QgsRasterLayer,
    QgsUnitTypes,
)

from qgis.analysis import QgsAlignRaster

from qgis import processing


def tr(message):
    """Get the translation for a string using Qt translation API.
    We implement this ourselves since we do not inherit QObject.

    :param message: String for translation.
    :type message: str, QString

    :returns: Translated version of message.
    :rtype: QString
    """
    # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
    return QtCore.QCoreApplication.translate("QgisCplus", message)


def log(
    message: str,
    name: str = "qgis_cplus",
    info: bool = True,
    notify: bool = True,
):
    """Logs the message into QGIS logs using qgis_cplus as the default
    log instance.
    If notify_user is True, user will be notified about the log.

    :param message: The log message
    :type message: str

    :param name: Name of te log instance, qgis_cplus is the default
    :type message: str

    :param info: Whether the message is about info or a
    warning
    :type info: bool

    :param notify: Whether to notify user about the log
    :type notify: bool
    """
    level = Qgis.Info if info else Qgis.Warning
    QgsMessageLog.logMessage(
        message,
        name,
        level=level,
        notifyUser=notify,
    )


def clean_filename(filename):
    """Creates a safe filename by removing operating system
    invalid filename characters.

    :param filename: File name
    :type filename: str

    :returns A clean file name
    :rtype str
    """
    characters = " %:/,\[]<>*?"

    for character in characters:
        if character in filename:
            filename = filename.replace(character, "_")

    return filename


def calculate_raster_value_area(
    layer: QgsRasterLayer, band_number: int = 1, feedback: QgsProcessingFeedback = None
) -> dict:
    """Calculates the area of value pixels for the given band in a raster layer.

    Please note that this function will run in the main application thread hence
    for best results, it is recommended to execute it in a background process
    if part of a bigger workflow.

    :param layer: Input layer whose area for value pixels is to be calculated.
    :type layer: QgsRasterLayer

    :param band_number: Band number to compute area, default is band one.
    :type band_number: int

    :param feedback: Feedback object for progress during area calculation.
    :type feedback: QgsProcessingFeedback

    :returns: A dictionary containing the pixel value as
    the key and the corresponding area in hectares as the value for all the pixels
    in the raster otherwise returns a empty dictionary if the raster is invalid
    or if it is empty.
    :rtype: float
    """
    if not layer.isValid():
        log("Invalid layer for raster area calculation.", info=False)
        return {}

    algorithm_name = "native:rasterlayeruniquevaluesreport"
    params = {
        "INPUT": layer,
        "BAND": band_number,
        "OUTPUT_TABLE": "TEMPORARY_OUTPUT",
        "OUTPUT_HTML_FILE": QgsProcessing.TEMPORARY_OUTPUT,
    }

    algorithm_result = processing.run(algorithm_name, params, feedback=feedback)

    # Get number of pixels with values
    total_pixel_count = algorithm_result["TOTAL_PIXEL_COUNT"]
    if total_pixel_count == 0:
        log("Input layer for raster area calculation is empty.", info=False)
        return {}

    output_table = algorithm_result["OUTPUT_TABLE"]
    if output_table is None:
        log("Unique values raster table could not be retrieved.", info=False)
        return {}

    area_calc = QgsDistanceArea()
    crs = layer.crs()
    area_calc.setSourceCrs(crs, QgsCoordinateTransformContext())
    if crs is not None:
        # Use ellipsoid calculation if available
        area_calc.setEllipsoid(crs.ellipsoidAcronym())

    version = Qgis.versionInt()
    if version < 33000:
        unit_type = QgsUnitTypes.AreaUnit.AreaHectares
    else:
        unit_type = Qgis.AreaUnit.Hectares

    pixel_areas = {}
    features = output_table.getFeatures()
    for f in features:
        pixel_value = f.attribute(0)
        area = f.attribute(2)
        pixel_value_area = area_calc.convertAreaMeasurement(area, unit_type)
        pixel_areas[pixel_value] = pixel_value_area

    return pixel_areas


def transform_extent(extent, source_crs, dest_crs):
    """Transforms the passed extent into the destination crs

     :param extent: Target extent
    :type extent: QgsRectangle

    :param source_crs: Source CRS of the passed extent
    :type source_crs: QgsCoordinateReferenceSystem

    :param dest_crs: Destination CRS
    :type dest_crs: QgsCoordinateReferenceSystem
    """

    transform = QgsCoordinateTransform(source_crs, dest_crs, QgsProject.instance())
    transformed_extent = transform.transformBoundingBox(extent)

    return transformed_extent


def align_rasters(
    input_raster_source,
    reference_raster_source,
    extent=None,
    output_dir=None,
    rescale_values=False,
    resample_method=0,
):
    """
    Based from work on https://github.com/inasafe/inasafe/pull/2070
    Aligns the passed raster files source and save the results into new files.

    :param input_raster_source: Input layer source
    :type input_raster_source: str

    :param reference_raster_source: Reference layer source
    :type reference_raster_source: str

    :param extent: Clip extent
    :type extent: list

    :param output_dir: Absolute path of the output directory for the snapped
    layers
    :type output_dir: str

    :param rescale_values: Whether to rescale pixel values
    :type rescale_values: bool

    :param resample_method: Method to use when resampling
    :type resample_method: QgsAlignRaster.ResampleAlg

    """

    try:
        snap_directory = os.path.join(output_dir, "snap_layers")

        BaseFileUtils.create_new_dir(snap_directory)

        input_path = Path(input_raster_source)

        input_layer_output = os.path.join(
            f"{snap_directory}", f"{input_path.stem}_{str(uuid.uuid4())[:4]}.tif"
        )

        BaseFileUtils.create_new_file(input_layer_output)

        align = QgsAlignRaster()
        lst = [
            QgsAlignRaster.Item(input_raster_source, input_layer_output),
        ]

        resample_method_value = QgsAlignRaster.ResampleAlg.RA_NearestNeighbour

        try:
            resample_method_value = QgsAlignRaster.ResampleAlg(int(resample_method))
        except Exception as e:
            log(f"Problem creating a resample value when snapping, {e}")

        if rescale_values:
            lst[0].rescaleValues = rescale_values

        lst[0].resample_method = resample_method_value

        align.setRasters(lst)
        align.setParametersFromRaster(reference_raster_source)

        layer = QgsRasterLayer(input_raster_source, "input_layer")

        extent = transform_extent(
            layer.extent(),
            QgsCoordinateReferenceSystem(layer.crs()),
            QgsCoordinateReferenceSystem(align.destinationCrs()),
        )

        align.setClipExtent(extent)

        log(f"Snapping clip extent {layer.extent().asWktPolygon()} \n")

        if not align.run():
            log(
                f"Problem during snapping for {input_raster_source} and "
                f"{reference_raster_source}, {align.errorMessage()}"
            )
            raise Exception(align.errorMessage())
    except Exception as e:
        log(
            f"Problem occured when snapping, {str(e)}."
            f" Update snap settings and re-run the analysis"
        )

        return None, None

    log(
        f"Finished snapping"
        f" original layer - {input_raster_source},"
        f"snapped output - {input_layer_output} \n"
    )

    return input_layer_output, None


class BaseFileUtils:
    """Provides functionality for commonly used file-related operations."""

    @staticmethod
    def create_new_dir(directory: str, log_message: str = ""):
        """Creates new file directory if it doesn't exist."""
        p = Path(directory)
        if not p.exists():
            try:
                p.mkdir()
            except (FileNotFoundError, OSError):
                log(log_message)

    @staticmethod
    def create_new_file(file_path: str, log_message: str = ""):
        """Creates new file."""
        p = Path(file_path)

        if not p.exists():
            try:
                p.touch(exist_ok=True)
            except FileNotFoundError:
                log(log_message)


def align_rasters(
    input_raster_source,
    reference_raster_source,
    extent=None,
    output_dir=None,
    rescale_values=False,
    resample_method=0,
):
    """
    Based from work on https://github.com/inasafe/inasafe/pull/2070
    Aligns the passed raster files source and save the results into new files.

    :param input_raster_source: Input layer source
    :type input_raster_source: str

    :param reference_raster_source: Reference layer source
    :type reference_raster_source: str

    :param extent: Clip extent
    :type extent: list

    :param output_dir: Absolute path of the output directory for the snapped
    layers
    :type output_dir: str

    :param rescale_values: Whether to rescale pixel values
    :type rescale_values: bool

    :param resample_method: Method to use when resampling
    :type resample_method: QgsAlignRaster.ResampleAlg

    """

    try:
        snap_directory = os.path.join(output_dir, "snap_layers")

        BaseFileUtils.create_new_dir(snap_directory)

        input_path = Path(input_raster_source)

        input_layer_output = os.path.join(
            f"{snap_directory}", f"{input_path.stem}_{str(uuid.uuid4())[:4]}.tif"
        )

        BaseFileUtils.create_new_file(input_layer_output)

        align = QgsAlignRaster()
        lst = [
            QgsAlignRaster.Item(input_raster_source, input_layer_output),
        ]

        resample_method_value = QgsAlignRaster.ResampleAlg.RA_NearestNeighbour

        try:
            resample_method_value = QgsAlignRaster.ResampleAlg(int(resample_method))
        except Exception as e:
            log(f"Problem creating a resample value when snapping, {e}")

        if rescale_values:
            lst[0].rescaleValues = rescale_values

        lst[0].resample_method = resample_method_value

        align.setRasters(lst)
        align.setParametersFromRaster(reference_raster_source)

        layer = QgsRasterLayer(input_raster_source, "input_layer")

        extent = transform_extent(
            layer.extent(),
            QgsCoordinateReferenceSystem(layer.crs()),
            QgsCoordinateReferenceSystem(align.destinationCrs()),
        )

        align.setClipExtent(extent)

        log(f"Snapping clip extent {layer.extent().asWktPolygon()} \n")

        if not align.run():
            log(
                f"Problem during snapping for {input_raster_source} and "
                f"{reference_raster_source}, {align.errorMessage()}"
            )
            raise Exception(align.errorMessage())
    except Exception as e:
        log(
            f"Problem occured when snapping, {str(e)}."
            f" Update snap settings and re-run the analysis"
        )

        return None, None

    log(
        f"Finished snapping"
        f" original layer - {input_raster_source},"
        f"snapped output - {input_layer_output} \n"
    )

    return input_layer_output, None


def get_layer_type(file_path: str):
    """Get layer type code from file path."""
    file_name, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in [".tif", ".tiff"]:
        return 0
    elif file_extension.lower() in [".geojson", ".zip", ".shp"]:
        return 1
    else:
        return -1


class CustomJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder which handles UUID and datetime."""

    def default(self, obj):
        """Return value based on object type.

        :param obj: Object to be encoded
        :type obj: Any
        :return: Encoded value
        :rtype: Any
        """
        if isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return obj.hex
        if isinstance(obj, datetime.datetime):
            # if the obj is uuid, we simply return the value of uuid
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def todict(obj, classkey=None):
    """Convert any object to dictionary."""

    if isinstance(obj, dict):
        data = {}
        for k, v in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict(
            [
                (key, todict(value, classkey))
                for key, value in obj.__dict__.items()
                if not callable(value) and not key.startswith("_")
            ]
        )
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj
