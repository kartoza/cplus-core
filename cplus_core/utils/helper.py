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
from osgeo import gdal
import numpy as np
from scipy.ndimage import label
import math

from qgis.PyQt import QtCore
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer,
    QgsRasterBlock,
    Qgis,
    QgsRasterPipe,
    QgsRasterDataProvider,
    QgsRasterFileWriter,
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


def array_from_raster(input_layer: QgsRasterLayer):
    """
    Read a raster and return the pixel values as numpy array
    
    :param input_layer: Input raster layer
    :type input_layer: QgsRasterLayer

    :return: Pixel values as numpy array
    :rtype: ndarray

    """
    provider = input_layer.dataProvider()    
    extent = provider.extent()    
    height, width = input_layer.height(), input_layer.width()    
    block = provider.block(1, extent, width, height) # assuming single band raster
    array = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            array[i, j] = block.value(i, j)

    return array


def raster_from_array(
        array, 
        extent, 
        crs, 
        output_path=None, 
        layer_name="Numpy Raster"
    ) -> QgsRasterLayer:
    """
    Create a QGIS raster layer from a numpy array
    
    :param array: Input numpy array (2D or 3D)
    :type array: ndarray

    :param extent: QgsRectangle with the extent in CRS coordinates
    :type extent: QgsRectangle

    :param crs: Coordinate system
    :type crs: QgsCoordinateReferenceSystem

    :param output_path: Optional path to save as GeoTIFF (if None, creates temporary layer)
    :type output_path: str

    :param layer_name: Optional name for the layer
    :type layer_name: str
    
    Returns:
    QgsRasterLayer
    """
    
    # Determine data type based on numpy array dtype
    dtype_map = {
        np.uint8: Qgis.Byte,
        np.int16: Qgis.Int16,
        np.uint16: Qgis.UInt16,
        np.int32: Qgis.Int32,
        np.uint32: Qgis.UInt32,
        np.float32: Qgis.Float32,
        np.float64: Qgis.Float64
    }
    
    data_type = dtype_map.get(array.dtype.type, Qgis.Float32)
    
    # Get array dimensions
    if array.ndim == 2:
        height, width = array.shape
        bands = 1
        # Reshape to 3D for consistent processing
        array = array.reshape(1, height, width)
    elif array.ndim == 3:
        bands, height, width = array.shape
    else:
        raise ValueError("Array must be 2D or 3D")

    if output_path:
        # Create a raster file writer
        writer = QgsRasterFileWriter(output_path)
        writer.setOutputProviderKey('gdal')
        writer.setOutputFormat('GTiff')
        
        # Create the output raster
        provider = writer.createOneBandRaster(data_type, width, height, extent, crs)
    else:
        # Create a temporary memory layer
        provider = QgsRasterDataProvider('memory', '1', data_type, width, height, 1)
    
    # Set the data for each band
    for band in range(bands):
        # Create raster block
        block = QgsRasterBlock(data_type, width, height)
        
        # Convert numpy array to bytes for the block
        if array.dtype == np.float32:
            data_bytes = array[band].tobytes()
        else:
            # Ensure correct byte order
            data_bytes = array[band].astype(array.dtype.newbyteorder('=')).tobytes()
        
        # Write data to block
        block.setData(data_bytes)
        
        # Write block to provider
        provider.writeBlock(block, band + 1)

        # Set NoData value to 0
        provider.setNoDataValue(band + 1, 0)
    
    if output_path:
        provider.setEditable(False)
        raster_layer = QgsRasterLayer(output_path, layer_name)
    else:
        # For memory provider, we need to create a proper raster layer
        # This is a workaround since memory provider doesn't easily create layers
        uri = f"MEM::{width}:{height}:{bands}:{data_type}:[{extent.xMinimum()},{extent.yMinimum()},{extent.xMaximum()},{extent.yMaximum()}]"
        raster_layer = QgsRasterLayer(uri, layer_name, "memory")
        # Copy the data (simplified approach)
        pipe = QgsRasterPipe()
        pipe.set(provider.clone())
        raster_layer = QgsRasterLayer(pipe, layer_name)
    
    # Set CRS
    raster_layer.setCrs(crs)

    return raster_layer


def create_connectivity_raster(
    input_raster_path: str,
    output_raster_path: str,
    connectivity_type : int = 8,
    min_patch_area: float = None, 
    area_unit : str = "ha"
    ):
    """
    Computes the pixel connectivity of a given binary raster

    :param input_raster_path: Input layer path
    :type input_raster_path: str

    :param output_raster_path: Output layer path
    :type output_raster_path: str

    :param connectivity_type: Number of pixels reachable from the 
        specified pixel in 4- or 8-directional adjacency
        For 4-directional connectivity → N, S, E, W adjacency
        For 8-directional connectivity → N, S, E, W, NE, NW, SE, SW adjacency
        Default to 8
    :type connectivity_type: int

    :param min_patch_area: Minimum patch size, default to None
    :type min_patch_area: float | None

    :param area_unit: Unit of the patch size i.e ha or m2, defaulto to ha
    :type area_unit: str
    """

    logs = []

    try:
        # -----------------------
        # 1. Load raster
        # -----------------------
        input_layer = QgsRasterLayer(input_raster_path, "raster")
        if not input_layer.isValid():
            logs.append(f"Invalid raster {input_raster_path}")
            return False, logs

        arr = array_from_raster(input_layer)
        height, width = input_layer.height(), input_layer.width() 

        provider = input_layer.dataProvider()
        if provider.sourceHasNoDataValue(1):
            # Convert NoData value to 0
            nodata_value = provider.sourceNoDataValue(1)
            arr[arr == nodata_value] = 0.0
        
        # Expecting a normalized raster 0-1. Convert any value greater than 1 to 0
        arr[arr > 1] = 0.0

        # Convert to binary to ignore resistance caused by varying pixel values
        arr = (arr > 0).astype(np.uint8)
        
        # Just need gdal to get the raster GeoTransform.
        # Cannot directly get it from qgis rasterlayer because layer.rasterUnitsPerPixelY() is absolute
        # gt = [extent.xMinimum(), layer.rasterUnitsPerPixelX(), 0, extent.yMaximum(), 0, layer.rasterUnitsPerPixelY()]
        gdal_ds = gdal.Open(input_raster_path)
        gt = gdal_ds.GetGeoTransform()
        gdal_ds = None

        # pixel size in map units (assume square pixels)
        px_w = abs(gt[1])
        px_h = abs(gt[5]) if gt[5] != 0 else px_w
        
        # use average pixel size (map units) for distance scaling
        pixel_size = math.sqrt(px_w * px_h)

        # Minimum number of pixels to discriminate
        MIN_SIZE_PENALTY_K = 100
        EPS = 1e-12

        # -----------------------
        # 2. Determine the number of pixels for the minimum patch area
        # -----------------------

        if min_patch_area:
            pixel_area_m2 = abs(px_w * px_h)
            if area_unit.lower() == "ha":
                min_patch_area_m2 = min_patch_area * 10000.0
            elif area_unit.lower() == "m2":
                min_patch_area_m2 = min_patch_area
            else:
                logs.append("Patch Area Unit must be 'ha' or 'm2'")
                return False, logs

            MIN_SIZE_PENALTY_K = int(math.ceil(min_patch_area_m2 / pixel_area_m2))

        # -----------------------
        # 3. Compute connected clusters
        # -----------------------
        if connectivity_type == 4:
            struct = np.array([[0,1,0],
                            [1,1,1],
                            [0,1,0]], dtype=np.uint8)
        else:
            struct = np.ones((3,3), dtype=np.uint8)

        labeled, n_labels = label(arr == 1, structure=struct)

        cluster_size_array = np.zeros_like(labeled, dtype=np.int32)
        centroid_mean_dist_array = np.zeros_like(labeled, dtype=np.float32)
        raw_score_array = np.zeros_like(labeled, dtype=np.float32)

        # precompute pixel coordinates in map units
        rows, cols = np.indices((height, width))

        # centroid coords = pixel center: x = gt[0] + (col + 0.5)*gt[1] + (row + 0.5)*gt[2] (usually gt[2]==0)
        # y = gt[3] + (col + 0.5)*gt[4] + (row + 0.5)*gt[5] (usually gt[4]==0)

        xs = gt[0] + (cols + 0.5) * gt[1] + (rows + 0.5) * gt[2]
        ys = gt[3] + (cols + 0.5) * gt[4] + (rows + 0.5) * gt[5]

        # iterate clusters
        cluster_scores = []
        for lbl in range(1, n_labels + 1):
            mask = (labeled == lbl)
            S = int(mask.sum())
            cluster_size_array[mask] = S

            # coordinates of pixels in map units (N x 2)
            xs_pix = xs[mask].astype(float)
            ys_pix = ys[mask].astype(float)
            pts = np.column_stack((xs_pix, ys_pix))

            if S == 1:
                # Single pixel: distance = 0
                mean_dist = 0.0
            else:
                # centroid
                centroid = pts.mean(axis=0)
                # compute distances from pixels to cluster centroid (map units)
                dists = np.linalg.norm(pts - centroid, axis=1)
                mean_dist = float(dists.mean())

            centroid_mean_dist_array[mask] = mean_dist

            # estimate cluster radius from area: pixel_area * S
            pixel_area = abs(gt[1] * gt[5]) if gt[5] != 0 else (px_w * px_h)
            cluster_area = S * pixel_area
            if cluster_area <= 0:
                r_est = pixel_size / 2.0
            else:
                r_est = math.sqrt(cluster_area / math.pi)

            denom = (r_est if r_est > 0 else (pixel_size/2.0))
            compactness = math.exp(- (mean_dist / (denom + EPS)))

            k = float(MIN_SIZE_PENALTY_K)
            size_penalty = 1.0 / (1.0 + math.exp(-(S - k) / (k + EPS)))  # ranges ~0..1

            raw_score = S * compactness * size_penalty

            raw_score_array[mask] = raw_score
            cluster_scores.append(raw_score)


        if len(cluster_scores) == 0:
            logs.append(f"No clusters found for raster {input_raster_path}")
            return False, logs

        # Normalize raw_score_array over pixels that belong to clusters
        mask_clusters = cluster_size_array > 0
        raw_vals = raw_score_array[mask_clusters]
        min_raw = float(np.nanmin(raw_vals))
        max_raw = float(np.nanmax(raw_vals))
        if abs(max_raw - min_raw) < EPS:
            norm_score_array = np.zeros_like(raw_score_array, dtype=np.float32)
            norm_score_array[mask_clusters] = 1.0
        else:
            norm_score_array = np.zeros_like(raw_score_array, dtype=np.float32)
            norm_score_array[mask_clusters] = (raw_score_array[mask_clusters] - min_raw) / (max_raw - min_raw)

        # Ignore clusters with pixels less than MIN_SIZE_PENALTY_K
        # norm_score_array[cluster_size_array < MIN_SIZE_PENALTY_K] = 0

        output_layer = raster_from_array(norm_score_array, input_layer.extent(), input_layer.crs(), output_raster_path)
        if output_layer and output_layer.isValid():
            return True, logs
    
    except Exception as e:
        logs.append(
            f"Problem occured when creating connectivity layer, {str(e)}."
        )
        logs.append(traceback.format_exc())

    return False, logs


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
        
        if min_value >= 0 and max_value <= 1:
            return True, f"Layer is already normalized (min={min_value}, max={max_value})"
        

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
    
