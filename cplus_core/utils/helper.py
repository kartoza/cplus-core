# -*- coding: utf-8 -*-
"""
    Plugin utilities
"""


import os
import json
import uuid
import traceback
import datetime
from pathlib import Path
from uuid import UUID
from enum import Enum
import shutil

from qgis.PyQt import QtCore
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer,
    QgsProcessingContext,
    QgsProcessingFeedback,
)
from qgis import processing

from qgis.analysis import QgsAlignRaster


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


def transform_extent(extent, source_crs, dest_crs):
    """Transforms the passed extent into the destination crs

     :param extent: Target extent
    :type extent: QgsRectangle

    :param source_crs: Source CRS of the passed extent
    :type source_crs: QgsCoordinateReferenceSystem

    :param dest_crs: Destination CRS
    :type dest_crs: QgsCoordinateReferenceSystem
    """

    transform = QgsCoordinateTransform(
        source_crs, dest_crs, QgsProject.instance())
    transformed_extent = transform.transformBoundingBox(extent)

    return transformed_extent


class BaseFileUtils:
    """
    Provides functionality for commonly used file-related operations.
    """

    @staticmethod
    def create_new_dir(directory: str, log_message: str = ""):
        """Creates new file directory if it doesn't exist"""
        p = Path(directory)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_new_file(file_path: str, log_message: str = ""):
        """Creates new file"""
        p = Path(file_path)

        if not p.exists():
            p.touch(exist_ok=True)

    @staticmethod
    def copy_file(file_path: str, target_dir: str, log_message: str = ""):
        """Copies file to the target directory"""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        target_path = Path(target_dir) / p.name
        if not target_path.parent.exists():
            target_path.parent.mkdir(parents=True)

        shutil.copy(p, target_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Failed to copy file to {target_dir}")
        return str(target_path)
    
    @staticmethod
    def remove_dir(directory: str):
        """Removes the directory and all its contents"""
        p = Path(directory)
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            return True
        return False
    
    def remove_file(file_path: str):
        """Removes the file if it exists"""
        p = Path(file_path)
        if p.exists() and p.is_file():
            p.unlink()
            return True
        return False
        

def align_rasters(
    input_raster_source,
    reference_raster_source,
    extent=None,
    output_dir=None,
    rescale_values=False,
    resample_method=0,
    name="layer"
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
    logs = []
    try:
        snap_directory = os.path.join(output_dir, "snap_layers")

        BaseFileUtils.create_new_dir(snap_directory)

        input_path = Path(input_raster_source)

        input_layer_output = os.path.join(
            f"{snap_directory}",
            f"{input_path.stem}_{str(uuid.uuid4())[:4]}.tif"
        )

        BaseFileUtils.create_new_file(input_layer_output)

        align = QgsAlignRaster()
        lst = [
            QgsAlignRaster.Item(input_raster_source, input_layer_output),
        ]

        resample_method_value = QgsAlignRaster.ResampleAlg.RA_NearestNeighbour

        try:
            resample_method_value = QgsAlignRaster.ResampleAlg(
                int(resample_method))
        except Exception as e:
            logs.append(
                f"Problem creating a resample value when snapping {name}, {e}")

        if rescale_values:
            lst[0].rescaleValues = rescale_values

        lst[0].resample_method = resample_method_value

        align.setRasters(lst)
        align.setParametersFromRaster(reference_raster_source)

        layer = QgsRasterLayer(reference_raster_source, "reference_raster_source")

        extent = transform_extent(
            layer.extent(),
            QgsCoordinateReferenceSystem(layer.crs()),
            QgsCoordinateReferenceSystem(align.destinationCrs()),
        )

        align.setClipExtent(extent)

        logs.append(f"Snapping clip extent {layer.extent().asWktPolygon()} \n")

        if not align.run():
            logs.append(
                f"Problem during snapping for {name} {input_raster_source} and "
                f"{reference_raster_source}, {align.errorMessage()}"
            )
            raise Exception(align.errorMessage())
    except Exception as e:
        logs.append(
            f"Problem occured when snapping {name}, {str(e)}."
            f" Update snap settings and re-run the analysis"
        )
        logs.append(traceback.format_exc())

        return None, logs

    logs.append(
        f"Finished snapping {name}"
        f" original layer - {input_raster_source},"
        f"snapped output - {input_layer_output} \n"
    )

    return input_layer_output, logs


def get_layer_type(file_path: str):
    """
    Get layer type code from file path
    """
    file_name, file_extension = os.path.splitext(file_path)
    if file_extension.lower() in [".tif", ".tiff"]:
        return 0
    elif file_extension.lower() in [".geojson", ".zip", ".shp"]:
        return 1
    else:
        return -1


class CustomJsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder which handles UUID and datetime
    """

    def default(self, obj):
        if isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return obj.hex
        if isinstance(obj, datetime.datetime):
            # if the obj is uuid, we simply return the value of uuid
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def todict(obj, classkey=None):
    """
    Convert any object to dictionary
    """

    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
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


def unique_path_from_reference(reference_path: str) -> str:
    """
    Generate a new file path by appending a UUID4 to the base name of the reference path.

    :param reference_path: Original file path (e.g., '/data/country.tif')
    :return: New file path (e.g., '/data/country_<uuid4>.tif')
    """
    directory, filename = os.path.split(reference_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{uuid.uuid4().hex}{ext}"
    return os.path.join(directory, new_filename)


def normalize_raster(
    input_raster_path: str,
    output_raster_path: str,
    processing_context: QgsProcessingContext = None,
    feedback: QgsProcessingFeedback = None
):
    """
    Create a normalized input raster

    :param input_raster_path: Input layer path
    :type input_raster_path: str

    :param output_raster_path: Output layer path
    :type output_raster_path: str

    :param processing_context: Qgis processing context
    :type processing_context: QgsProcessingContext, default None

    :param feedback: Qgis processing feedback
    :type feedback: QgsProcessingFeedback
    """
    try:
        input_raster_layer = QgsRasterLayer(input_raster_path, "Input Raster")

        if not input_raster_layer.isValid():
            return False, f"Invalid raster layer {input_raster_path}"
        
        provider = input_raster_layer.dataProvider()
        band_statistics = provider.bandStatistics(1)
        min_value = band_statistics.minimumValue
        max_value = band_statistics.maximumValue

        if min_value is None or max_value is None:
            return False, f"Raster layer has no valid statistics, {input_raster_path}"
        
        if max_value < min_value:
            return (
                False, 
                f"Layer cannot be normalized, min value {min_value} is greater than max value {max_value}"
            )
        
        if min_value >= 0 and max_value <= 1:
            return True, f"Layer {input_raster_path} is already normalized (min={min_value}, max={max_value})"

        expression = f"(A - {min_value}) / ({max_value} - {min_value})"

        alg_params = {
            'INPUT_A': input_raster_path,
            "BAND_A": 1,
            'FORMULA': expression,
            'OPTIONS':'COMPRESS=DEFLATE|ZLEVEL=6|TILED=YES',
            'OUTPUT': output_raster_path,
        }

        result = processing.run(
            "gdal:rastercalculator",
            alg_params,
            context=processing_context,
            feedback=feedback,
        )

        if result.get("OUTPUT"):
            return True, f"Normalized raster saved to : {output_raster_path}"
    
    except Exception as e:
        return False, f"Problem normalizing pathways, {e} \n"
    
