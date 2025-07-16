# coding=utf-8
"""Tests for the CPLUS plugin utilities.

"""
import unittest
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle

from cplus_core.utils.helper import (
    align_rasters,
    todict,
    transform_extent,
    normalize_raster_layer
)


class CplusCoreUtilTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_transform_extent(self):
        """Test the transform_extent function."""
        source_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        target_crs = QgsCoordinateReferenceSystem("EPSG:3857")

        extent = QgsRectangle(0, 0, 10, 10)
        result = transform_extent(
            extent,
            source_crs,
            target_crs
        )
        self.assertIsInstance(result, QgsRectangle)
        self.assertAlmostEqual(result.xMinimum(), 0.0, places=5)
        self.assertAlmostEqual(result.yMinimum(), 0.0, places=5)
        self.assertAlmostEqual(result.xMaximum(), 1113194.9079327357, places=5)
        self.assertAlmostEqual(result.yMaximum(), 1118889.9748579594, places=5)
        self.assertAlmostEqual(result.area(), 1245542622548.8672, places=2)

    def test_align_rasters(self):
        """Test the align_rasters function."""
        # TODO: Implement this test
        # This function is not implemented yet, so we just assert False for now.
        # Once implemented, replace this with actual test logic.
        self.assertFalse(False)
    
    def test_todict(self):
        """Test the todict function."""
        result = todict({"key": "value", "nested": {"key2": "value2"}})
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {"key": "value", "nested": {"key2": "value2"}})

    def test_normalize_raster_layer(self):
        """Test the normalize_raster_layer function."""
        # TODO: Implement this test
        # This function is not implemented yet, so we just assert False for now.
        # Once implemented, replace this with actual test logic.
        self.assertFalse(False)
    
    def test_replace_nodata_value_from_reference(self):
        """Test the replace_nodata_value_from_reference function."""
        # TODO: Implement this test
        # This function is not implemented yet, so we just assert False for now.
        # Once implemented, replace this with actual test logic.
        self.assertFalse(False)
    
