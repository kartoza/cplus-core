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

import numpy as np
import rasterio

import numpy as np
import rasterio

from qgis.PyQt import QtCore
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer
)

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

def normalize_raster_layer(
        input_path,
        output_directory=None,
        nodata_value=-9999.0,
        carbon_coefficient=0.0,
        suitability_index=0.0
):
    """
    Normalize raster to 0 - 1.
    :param input_path: Path to input raster file
    :type input_path: str
    :param output_directory: Directory to save the output raster, defaults to None
    :type output_directory: str
    :param nodata_value: NoData value to assign to the output, defaults to -9999.0
    :type nodata_value: float
    :param suitability_index: Suitability index to apply to the raster values, defaults to 0.0
    :type suitability_index: float
    :param carbon_coefficient: Carbon coefficient to apply to the raster values, defaults to 0.0
    :rttype carbon_coefficient: float
    :return: Path to the output raster file or None if an error occurs
    :rtype: str
    :return: Tuple of (output file path or None, logs)
    :rtype: tuple
    """
    logs = []
    try:
        if output_directory is None:
            output_directory = Path(input_path).parent
        output_path = os.path.join(
            f"{output_directory}",
            f"{Path(input_path).stem}_norm_{str(uuid.uuid4())[:4]}.tif"
        )
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            profile.update({
                'compress': 'deflate',
                'zlevel': 6,
                'tiled': True
            })

            data = src.read()
            nodata = src.nodata

            if nodata is not None:
                mask = data != nodata
            else:
                mask = np.ones_like(data, dtype=bool)
            
            valid = data[mask]
            min_val = valid.min()
            max_val = valid.max()

            if min_val == max_val:
                logs.append(f"Raster has no variation, skipping normalization.")
                return None, logs

            range_val = max_val - min_val if max_val != min_val else 1.0
        
            normalization_index = carbon_coefficient + suitability_index

            norm_data = np.zeros_like(data, dtype=np.float32)

            if normalization_index > 0:
                norm_data[mask] = normalization_index * (valid - min_val) / range_val
            else:
                norm_data[mask] = (valid - min_val) / range_val

            if nodata_value is not None:
                nodata = nodata_value

            if nodata is not None:
                norm_data[~mask] = nodata
                profile['nodata'] = nodata

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(norm_data)

        return output_path, logs
    except Exception as e:
            logs.append(f"Error thrown when normalizing ratser: {e}")
            logs.append(traceback.format_exc())
            return None, logs
    
def replace_nodata_value_from_reference(
        source_path,
        reference_path=None,
        output_path=None
):
    """
    Replace NoData value in raster with a specified value.
    
    :param source_path: Path to input raster file
    :type source_path: str
    :param reference_path: Path to reference raster file, not used in this function
    :type reference_path: str
    :param output_path: Path to save the output raster, defaults to None
    :type output_path: str
    :return: Tuple of (success, logs)
    :rtype: tuple
    """
    logs = []
    try:
        if output_path is None:
            output_path = os.path.join(
                f"{Path(source_path).parent}",
                f"{Path(source_path).stem}_nodata_{str(uuid.uuid4())[:4]}.tif"
            )
        with rasterio.open(source_path) as src:
            profile = src.profile.copy()
            src_data = src.read()

            with rasterio.open(reference_path) as ref_src:
                ref_nodata = ref_src.nodata
                profile.update({
                    'compress': 'deflate',
                    'zlevel': 6,
                    'tiled': True,
                    'nodata': ref_nodata
                })
                ref_data = ref_src.read()
            
                if ref_nodata is not None and src_data.shape == ref_data.shape:
                    # If reference raster has a nodata value, use it to mask the source data
                    mask = ref_data != ref_nodata
                    src_data[~mask] = ref_nodata
                else:
                    logs.append(
                        f"No nodata value found in reference raster or shapes do not match."
                        f"Shapes= {src_data.shape} and {ref_data.shape}."
                    )                 
                    return False, logs

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(src_data)
        return True, logs
    except Exception as e:
        logs.append(f"Error thrown when replacing NoData value: {e}")
        logs.append(traceback.format_exc())
        return False, logs
