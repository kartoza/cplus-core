# coding=utf-8
"""Tests for the plugin processing tasks

"""

import unittest

import os
import uuid
import processing
import datetime

from processing.core.Processing import Processing

from qgis.core import Qgis, QgsRasterLayer, QgsRectangle

from cplus_core.analysis.analysis import ScenarioAnalysisTask
from cplus_core.analysis.task_config import TaskConfig
from cplus_core.models.base import Scenario, NcsPathway, Activity, SpatialExtent
from cplus_core.utils.helper import BaseFileUtils

pathway_layer_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "pathways", "layers"
)

snapping_layer = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "snapping", "snapping_layer.tif"
)

mask_layers_paths = [
     os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "mask", "layers", "test_mask_1.shp"
    )
]

pathway_layer_path = os.path.join(pathway_layer_directory, "test_pathway_1.tif")

priority_layers_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "priority", "layers"
)

priority_layer_path_1 = os.path.join(
    priority_layers_directory, "test_priority_1.tif"
)

test_priority_groups = [{
    "uuid": "a4f76e6c-9f83-4a9c-b700-fb1ae04860a4",
    "name": "test_priority_group",
    "description": "test_priority_group_description",
    "value": 1,
}]

priority_layers = [{
    "uuid": "c931282f-db2d-4644-9786-6720b3ab206a",
    "name": "test_priority_layer",
    "description": "test_priority_layer_description",
    "selected": False,
    "path": priority_layer_path_1,
    "groups": test_priority_groups,
}]


class ScenarioAnalysisTaskTest(unittest.TestCase):
    def setUp(self):
        Processing.initialize()

    def test_scenario_pathways_weighting(self):
        """Test the weighting of NCS pathways"""

        test_pathway = NcsPathway(
            uuid=uuid.uuid4(),
            name="test_pathway",
            description="test_description",
            path=pathway_layer_path,
            priority_layers=[],
        )

        test_layer = QgsRasterLayer(test_pathway.path, test_pathway.name)

        test_extent = test_layer.extent()

        extent_string = (
            f"{test_extent.xMinimum()},{test_extent.xMaximum()},"
            f"{test_extent.yMinimum()},{test_extent.yMaximum()}"
            f" [{test_layer.crs().authid()}]"
        )

        spatial_extent = SpatialExtent(
            bbox=[test_extent.xMinimum(), test_extent.xMaximum(), test_extent.yMinimum(), test_extent.yMaximum()],
            crs=test_layer.crs().authid()
        )

        test_activity = Activity(
            uuid=uuid.uuid4(),
            name="test_activity",
            description="test_description",
            pathways=[test_pathway],
        )

        scenario = Scenario(
            uuid=uuid.uuid4(),
            name="Scenario",
            description="Scenario description",
            activities=[test_activity],
            extent=spatial_extent,
            priority_layer_groups=test_priority_groups,
            weighted_activities=[],
        )

        task_config = TaskConfig(
            scenario=scenario,
            priority_layers=priority_layers,
            priority_layer_groups=test_priority_groups,
            analysis_activities=scenario.activities,
            all_activities=[test_activity],
            snapping_enabled=False,
            snap_layer=None,
            mask_layers_paths=",".join(mask_layers_paths),
            snap_rescale=False,
            snap_method=0, # 0 = Nearest neighbour, 1 = Bilinear, 2 = Cubic
            pathway_suitability_index=0.5,
            carbon_coefficient=1.0,
            sieve_enabled=False,
            sieve_threshold=10.0,
            ncs_with_carbon=True,
            landuse_project=True,
            landuse_normalized=True,
            landuse_weighted=True,
            highest_position=True,
            base_dir=os.path.dirname(os.path.abspath(__file__)),
        )

        analysis_task = ScenarioAnalysisTask(
            task_config= task_config,
        )

        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "pathways",
        )

        scenario_directory = os.path.join(
            f"{base_dir}",
            f'scenario_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
            f"_{str(uuid.uuid4())[:4]}",
        )

        analysis_task.scenario_directory = scenario_directory

        past_stat = test_layer.dataProvider().bandStatistics(1)

        self.assertEqual(past_stat.minimumValue, 1.0)
        self.assertEqual(past_stat.maximumValue, 10.0)

        results = analysis_task.run_pathways_weighting(
            test_priority_groups,
            extent_string,
            temporary_output=True,
        )

        self.assertTrue(results)

        result_layer = QgsRasterLayer(test_pathway.path, test_pathway.name)
        self.assertTrue(result_layer.isValid())

        stat = result_layer.dataProvider().bandStatistics(1)

        self.assertEqual(stat.minimumValue, 0.5)
        self.assertEqual(stat.maximumValue, 5.0)

        self.assertTrue(BaseFileUtils.remove_dir(scenario_directory))

    def test_scenario_activities_creation(self):
        pathway_layer_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "pathways", "layers"
        )

        pathway_layer_path_1 = os.path.join(
            pathway_layer_directory, "test_pathway_1.tif"
        )

        first_test_pathway = NcsPathway(
            uuid=uuid.uuid4(),
            name="first_test_pathway",
            description="first_test_description",
            path=pathway_layer_path_1,
        )

        pathway_layer_path_2 = os.path.join(
            pathway_layer_directory, "test_pathway_2.tif"
        )

        second_test_pathway = NcsPathway(
            uuid=uuid.uuid4(),
            name="second_test_pathway",
            description="second_test_description",
            path=pathway_layer_path_2,
        )

        first_test_layer = QgsRasterLayer(
            first_test_pathway.path, first_test_pathway.name
        )
        second_test_layer = QgsRasterLayer(
            second_test_pathway.path, second_test_pathway.name
        )

        test_extent = first_test_layer.extent()

        test_activity = Activity(
            uuid=uuid.uuid4(),
            name="test_activity",
            description="test_description",
            pathways=[first_test_pathway, second_test_pathway],
        )

        spatial_extent = SpatialExtent(
            bbox=[test_extent.xMinimum(), test_extent.xMaximum(), test_extent.yMinimum(), test_extent.yMaximum()],
            crs=first_test_layer.crs().authid()
        )

        scenario = Scenario(
            uuid=uuid.uuid4(),
            name="Scenario",
            description="Scenario description",
            activities=[test_activity],
            extent=spatial_extent,
            priority_layer_groups=[],
            weighted_activities=[]
        )

        task_config = TaskConfig(
            scenario=scenario,
            priority_layers=priority_layers,
            priority_layer_groups=test_priority_groups,
            analysis_activities=scenario.activities,
            all_activities=[test_activity],
            snapping_enabled=False,
            snap_layer=None,
            mask_layers_paths=",".join(mask_layers_paths),
            snap_rescale=False,
            snap_method=0, # 0 = Nearest neighbour, 1 = Bilinear, 2 = Cubic
            pathway_suitability_index=1.0,
            carbon_coefficient=1.0,
            sieve_enabled=False,
            sieve_threshold=10.0,
            ncs_with_carbon=True,
            landuse_project=True,
            landuse_normalized=True,
            landuse_weighted=True,
            highest_position=True,
            base_dir=os.path.dirname(os.path.abspath(__file__)),
        )

        analysis_task = ScenarioAnalysisTask(
            task_config= task_config,
        )

        extent_string = (
            f"{test_extent.xMinimum()},{test_extent.xMaximum()},"
            f"{test_extent.yMinimum()},{test_extent.yMaximum()}"
            f" [{first_test_layer.crs().authid()}]"
        )

        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "pathways",
        )

        scenario_directory = os.path.join(
            f"{base_dir}",
            f'scenario_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
            f"_{str(uuid.uuid4())[:4]}",
        )

        analysis_task.scenario_directory = scenario_directory

        first_layer_stat = first_test_layer.dataProvider().bandStatistics(1)
        second_layer_stat = second_test_layer.dataProvider().bandStatistics(1)

        self.assertEqual(first_layer_stat.minimumValue, 1.0)
        self.assertEqual(first_layer_stat.maximumValue, 10.0)

        self.assertEqual(second_layer_stat.minimumValue, 7.0)
        self.assertEqual(second_layer_stat.maximumValue, 10.0)

        results = analysis_task.run_activities_analysis(
            extent_string,
            temporary_output=True,
        )

        self.assertTrue(results)

        result_layer = QgsRasterLayer(test_activity.path, test_activity.name)
        self.assertTrue(result_layer.isValid())

        stat = result_layer.dataProvider().bandStatistics(1)

        self.assertEqual(stat.minimumValue, 1.0)
        self.assertEqual(stat.maximumValue, 19.0)

        self.assertTrue(BaseFileUtils.remove_dir(scenario_directory))

    def test_scenario_activities_masking(self):
        activities_layer_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "activities", "layers"
        )

        mask_layers_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "mask", "layers"
        )

        activity_layer_path_1 = os.path.join(
            activities_layer_directory, "test_activity_1.tif"
        )
        mask_layer_path_1 = os.path.join(mask_layers_directory, "test_mask_1.shp")

        test_activity = Activity(
            uuid=uuid.uuid4(),
            name="test_activity",
            description="test_description",
            pathways=[],
            path=activity_layer_path_1,
            mask_paths=[mask_layer_path_1],
        )

        activity_layer = QgsRasterLayer(test_activity.path, test_activity.name)

        test_extent = activity_layer.extent()

        spatial_extent = SpatialExtent(
            bbox=[test_extent.xMinimum(), test_extent.xMaximum(), test_extent.yMinimum(), test_extent.yMaximum()],
            crs=activity_layer.crs().authid()
        )

        scenario = Scenario(
            uuid=uuid.uuid4(),
            name="Scenario",
            description="Scenario description",
            activities=[test_activity],
            extent=spatial_extent,
            priority_layer_groups=[],
            weighted_activities=[]
        )

        task_config = TaskConfig(
            scenario=scenario,
            priority_layers=priority_layers,
            priority_layer_groups=test_priority_groups,
            analysis_activities=scenario.activities,
            all_activities=[test_activity],
            snapping_enabled=False,
            snap_layer=None,
            mask_layers_paths=",".join(mask_layers_paths),
            snap_rescale=False,
            snap_method=0, # 0 = Nearest neighbour, 1 = Bilinear, 2 = Cubic
            pathway_suitability_index=0.0,
            carbon_coefficient=1.0,
            sieve_enabled=False,
            sieve_threshold=10.0,
            ncs_with_carbon=True,
            landuse_project=True,
            landuse_normalized=True,
            landuse_weighted=True,
            highest_position=True,
            base_dir=os.path.dirname(os.path.abspath(__file__)),
        )

        analysis_task = ScenarioAnalysisTask(
            task_config= task_config,
        )

        extent_string = (
            f"{test_extent.xMinimum()},{test_extent.xMaximum()},"
            f"{test_extent.yMinimum()},{test_extent.yMaximum()}"
            f" [{activity_layer.crs().authid()}]"
        )

        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "activities",
        )

        scenario_directory = os.path.join(
            f"{base_dir}",
            f'scenario_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
            f"_{str(uuid.uuid4())[:4]}",
        )

        analysis_task.scenario_directory = scenario_directory

        # Before masking, check if the activity layer stats are correct
        activity_layer = QgsRasterLayer(test_activity.path, test_activity.name)
        first_layer_stat = activity_layer.dataProvider().bandStatistics(1)

        self.assertEqual(first_layer_stat.minimumValue, 1.0)
        self.assertEqual(first_layer_stat.maximumValue, 19.0)

        results = analysis_task.run_internal_activities_masking(
            extent_string, temporary_output=True
        )

        self.assertTrue(results)

        self.assertIsInstance(results, bool)
        self.assertTrue(results)

        self.assertIsNotNone(test_activity.path)

        result_layer = QgsRasterLayer(test_activity.path, test_activity.name)

        result_stat = result_layer.dataProvider().bandStatistics(1)
        self.assertEqual(result_stat.minimumValue, 1.0)
        self.assertEqual(result_stat.maximumValue, 18.0)

        self.assertTrue(result_layer.isValid())

        self.assertTrue(BaseFileUtils.remove_dir(scenario_directory))

    def test_scenario_activity_normalization(self):
        "Test the normalization of activities"
        activities_layer_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "activities", "layers"
        )

        activity_layer_path_1 = os.path.join(
            activities_layer_directory, "test_activity_1.tif"
        )

        test_activity = Activity(
            uuid=uuid.uuid4(),
            name="test_activity",
            description="test_description",
            pathways=[],
            path=activity_layer_path_1,
            mask_paths=[],
        )

        activity_layer = QgsRasterLayer(test_activity.path, test_activity.name)

        test_extent = activity_layer.extent()

        spatial_extent = SpatialExtent(
            bbox=[test_extent.xMinimum(), test_extent.xMaximum(), test_extent.yMinimum(), test_extent.yMaximum()],
            crs=activity_layer.crs().authid()
        )

        scenario = Scenario(
            uuid=uuid.uuid4(),
            name="Scenario",
            description="Scenario description",
            activities=[test_activity],
            extent=spatial_extent,
            priority_layer_groups=[],
            weighted_activities=[]
        )

        task_config = TaskConfig(
            scenario=scenario,
            priority_layers=priority_layers,
            priority_layer_groups=test_priority_groups,
            analysis_activities=scenario.activities,
            all_activities=[test_activity],
            snapping_enabled=False,
            snap_layer=None,
            mask_layers_paths=",".join(mask_layers_paths),
            snap_rescale=False,
            snap_method=0, # 0 = Nearest neighbour, 1 = Bilinear, 2 = Cubic
            pathway_suitability_index=0.0,
            carbon_coefficient=1.0,
            sieve_enabled=False,
            sieve_threshold=10.0,
            ncs_with_carbon=True,
            landuse_project=True,
            landuse_normalized=True,
            landuse_weighted=True,
            highest_position=True,
            base_dir=os.path.dirname(os.path.abspath(__file__)),
        )

        analysis_task = ScenarioAnalysisTask(
            task_config= task_config,
        )

        extent_string = (
            f"{test_extent.xMinimum()},{test_extent.xMaximum()},"
            f"{test_extent.yMinimum()},{test_extent.yMaximum()}"
            f" [{activity_layer.crs().authid()}]"
        )

        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "activities",
        )

        scenario_directory = os.path.join(
            f"{base_dir}",
            f'scenario_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
            f"_{str(uuid.uuid4())[:4]}",
        )

        analysis_task.scenario_directory = scenario_directory

        # Before normalization, check if the activity layer stats are correct
        activity_layer = QgsRasterLayer(test_activity.path, test_activity.name)
        first_layer_stat = activity_layer.dataProvider().bandStatistics(1)

        self.assertEqual(first_layer_stat.minimumValue, 1.0)
        self.assertEqual(first_layer_stat.maximumValue, 19.0)

        results = analysis_task.run_activity_normalization()

        self.assertTrue(results)

        self.assertIsInstance(results, bool)
        self.assertTrue(results)

        self.assertIsNotNone(test_activity.path)

        result_layer = QgsRasterLayer(test_activity.path, test_activity.name)

        result_stat = result_layer.dataProvider().bandStatistics(1)
        self.assertEqual(result_stat.minimumValue, 0.0)
        self.assertEqual(result_stat.maximumValue, 1.0)

        self.assertTrue(result_layer.isValid())

        self.assertTrue(BaseFileUtils.remove_dir(scenario_directory))

    def test_scenario_activity_investability(self):
        "Test the activity investability analysis"

        constant_layers_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "constants", "layers"
        )

        npv_layer_path_1 = os.path.join(constant_layers_directory, "test_npv.tif")
        years_experience_layer_path_1 = os.path.join(constant_layers_directory, "test_years_experience.tif")
        market_trends_layer_path_1 = os.path.join(constant_layers_directory, "test_market_trends.tif")
        confidence_layer_path_1 = os.path.join(constant_layers_directory, "test_confidence.tif")

        self.assertTrue(os.path.exists(npv_layer_path_1))
        self.assertTrue(os.path.exists(years_experience_layer_path_1))
        self.assertTrue(os.path.exists(market_trends_layer_path_1))
        self.assertTrue(os.path.exists(confidence_layer_path_1))

        npv_layer = {
            "uuid": "a931282f-db2d-4644-9786-6720b3ab206a",
            "name": "NPV",
            "description": "test_npv_layer_description",
            "path": npv_layer_path_1,
        }

        years_experience_layer = {
            "uuid": "b931282f-db2d-4644-9786-6720b3ab206b",
            "name": "Year of Experience",
            "description": "test_years_experience_layer_description",
            "path": years_experience_layer_path_1,
        }

        market_trends_layer = {
            "uuid": "c931282f-db2d-4644-9786-6720b3ab206c",
            "name": "Market Trends",
            "description": "test_market_trends_layer_description",
            "path": market_trends_layer_path_1,
        }

        confidence_layer = {
            "uuid": "d931282f-db2d-4644-9786-6720b3ab206d",
            "name": "Confidence in ability to deliver",
            "description": "test_confidence_layer_description",
            "path": confidence_layer_path_1,
        }

        activities_layer_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "activities", "layers"
        )

        activity_layer_path = os.path.join(
            activities_layer_directory, "test_activity_2.tif"
        )

        test_activity = Activity(
            uuid=uuid.uuid4(),
            name="test_activity",
            description="test_description",
            pathways=[],
            path=activity_layer_path,
            mask_paths=[],
            constant_rasters=[npv_layer, years_experience_layer, market_trends_layer, confidence_layer]
        )

        activity_layer = QgsRasterLayer(test_activity.path, test_activity.name)

        test_extent = activity_layer.extent()

        spatial_extent = SpatialExtent(
            bbox=[test_extent.xMinimum(), test_extent.xMaximum(), test_extent.yMinimum(), test_extent.yMaximum()],
            crs=activity_layer.crs().authid()
        )

        scenario = Scenario(
            uuid=uuid.uuid4(),
            name="Scenario",
            description="Scenario description",
            activities=[test_activity],
            extent=spatial_extent,
            priority_layer_groups=[],
            weighted_activities=[]
        )

        task_config = TaskConfig(
            scenario=scenario,
            priority_layers=priority_layers,
            priority_layer_groups=test_priority_groups,
            analysis_activities=scenario.activities,
            all_activities=[test_activity],
            snapping_enabled=False,
            snap_layer=None,
            mask_layers_paths=",".join(mask_layers_paths),
            snap_rescale=False,
            snap_method=0,
            pathway_suitability_index=0.0,
            carbon_coefficient=1.0,
            sieve_enabled=False,
            sieve_threshold=10.0,
            ncs_with_carbon=True,
            landuse_project=True,
            landuse_normalized=True,
            landuse_weighted=True,
            highest_position=True,
            base_dir=os.path.dirname(os.path.abspath(__file__)),
            pixel_connectivity_enabled=False
        )

        analysis_task = ScenarioAnalysisTask(
            task_config= task_config,
        )

        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "activities",
        )

        scenario_directory = os.path.join(
            f"{base_dir}",
            f'scenario_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
            f"_{str(uuid.uuid4())[:4]}",
        )

        analysis_task.scenario_directory = scenario_directory

        # Before normalization, check if the activity layer stats are correct
        activity_layer = QgsRasterLayer(test_activity.path, test_activity.name)
        first_layer_stat = activity_layer.dataProvider().bandStatistics(1)

        self.assertEqual(first_layer_stat.minimumValue, 0.0)
        self.assertEqual(first_layer_stat.maximumValue, 1.0)

        results = analysis_task.run_investability_analysis()

        self.assertTrue(results)

        self.assertIsInstance(results, bool)
        self.assertTrue(results)

        self.assertIsNotNone(test_activity.path)

        result_layer = QgsRasterLayer(test_activity.path, test_activity.name)

        result_stat = result_layer.dataProvider().bandStatistics(1)
        self.assertAlmostEqual(result_stat.minimumValue, 0.825, places=3)
        self.assertAlmostEqual(result_stat.maximumValue, 1.825, places=3)

        self.assertTrue(result_layer.isValid())

        self.assertTrue(BaseFileUtils.remove_dir(scenario_directory))

    def test_layer_snapping(self):
        """Test the layer snapping functionality."""
        activities_layer_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "activities", "layers"
        )

        mask_layers_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "mask", "layers"
        )

        activity_layer_path_1 = os.path.join(
            activities_layer_directory, "test_activity_1.tif"
        )
        mask_layer_path_1 = os.path.join(mask_layers_directory, "test_mask_1.shp")

        test_activity = Activity(
            uuid=uuid.uuid4(),
            name="test_activity",
            description="test_description",
            pathways=[],
            path=activity_layer_path_1,
            mask_paths=[mask_layer_path_1],
        )

        test_layer = QgsRasterLayer(snapping_layer, "test_snapping_layer")

        self.assertTrue(test_layer.isValid())

        extent = test_layer.extent()
        extent_string = (
            f"{extent.xMinimum()},{extent.xMaximum()},"
            f"{extent.yMinimum()},{extent.yMaximum()}"
            f" [{test_layer.crs().authid()}]"
        )

        spatial_extent = SpatialExtent(
            bbox=[extent.xMinimum(), extent.xMaximum(), extent.yMinimum(), extent.yMaximum()],
            crs=test_layer.crs().authid()
        )

        scenario = Scenario(
            uuid=uuid.uuid4(),
            name="Scenario",
            description="Scenario description",
            activities=[test_activity],
            extent=spatial_extent,
            priority_layer_groups=[],
            weighted_activities=[]
        )

        task_config = TaskConfig(
            scenario=scenario,
            priority_layers=priority_layers,
            priority_layer_groups=test_priority_groups,
            analysis_activities=scenario.activities,
            all_activities=[test_activity],
            snapping_enabled=True,
            snap_layer=snapping_layer,
            mask_layers_paths=",".join(mask_layers_paths),
            snap_rescale=True,
            snap_method=0, # 0 = Nearest neighbour, 1 = Bilinear, 2 = Cubic
            pathway_suitability_index=0.0,
            carbon_coefficient=1.0,
            sieve_enabled=False,
            sieve_threshold=10.0,
            ncs_with_carbon=True,
            landuse_project=True,
            landuse_normalized=True,
            landuse_weighted=True,
            highest_position=True,
            base_dir=os.path.dirname(os.path.abspath(__file__)),
        )

        analysis_task = ScenarioAnalysisTask(
            task_config=task_config,
        )

        result = analysis_task.snap_analysis_data(
            extent_string
        )

        self.assertTrue(result)

    def test_reprojection(self):
        """Test the layer reprojection functionality."""
        pathways_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "pathways", "layers"
        )

        pathway_layer_path_1 = os.path.join(
            pathways_directory, "test_pathway_1.tif"
        )

        test_layer = QgsRasterLayer(pathway_layer_path_1, "test__layer")

        self.assertTrue(test_layer.isValid())

        extent = test_layer.extent()

        spatial_extent = SpatialExtent(
            bbox=[extent.xMinimum(), extent.xMaximum(), extent.yMinimum(), extent.yMaximum()],
            crs=test_layer.crs().authid()
        )

        scenario = Scenario(
            uuid=uuid.uuid4(),
            name="Scenario",
            description="Scenario description",
            activities=[],
            extent=spatial_extent,
            priority_layer_groups=[],
            weighted_activities=[]
        )

        task_config = TaskConfig(
            scenario=scenario,
            priority_layers=priority_layers,
            priority_layer_groups=test_priority_groups,
            analysis_activities=scenario.activities,
            all_activities=[],
            snapping_enabled=True,
            snap_layer=snapping_layer,
            mask_layers_paths=",".join(mask_layers_paths),
            snap_rescale=True,
            snap_method=0,
            pathway_suitability_index=0.0,
            carbon_coefficient=1.0,
            sieve_enabled=False,
            sieve_threshold=10.0,
            ncs_with_carbon=True,
            landuse_project=True,
            landuse_normalized=True,
            landuse_weighted=True,
            highest_position=True,
            base_dir=os.path.dirname(os.path.abspath(__file__)),
        )

        analysis_task = ScenarioAnalysisTask(
            task_config=task_config,
        )

        result = analysis_task.reproject_layer(
            input_path=pathway_layer_path_1,
            target_crs="EPSG:3857"
        )

        self.assertTrue(result)
        self.assertTrue(os.path.exists(result))        
        if os.path.exists(result):
            raster = QgsRasterLayer(result, "reprojected_layer")
            self.assertTrue(raster.isValid())
            self.assertEqual(raster.crs().authid(), "EPSG:3857")
            self.assertNotEqual(raster.extent(), test_layer.extent())
            os.remove(result)
        
    def test_scenario_replace_nodata_value(self):
        """Test replacing nodata value functionality."""
        pathways_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "pathways", "layers"
        )

        pathway_layer_path_1 = os.path.join(
            pathways_directory, "test_pathway_1.tif"
        )

        test_layer = QgsRasterLayer(pathway_layer_path_1, "test__layer")
        self.assertTrue(test_layer.isValid())

        test_provider = test_layer.dataProvider()
        test_no_data_value = test_provider.sourceNoDataValue(1)
        self.assertAlmostEqual(test_no_data_value, 0.0)
       
        extent = test_layer.extent()

        spatial_extent = SpatialExtent(
            bbox=[extent.xMinimum(), extent.xMaximum(), extent.yMinimum(), extent.yMaximum()],
            crs=test_layer.crs().authid()
        )

        test_pathway = NcsPathway(
            uuid=uuid.uuid4(),
            name="test_pathway",
            description="test_description",
            path=pathway_layer_path_1,
            priority_layers=[],
        )

        test_activity = Activity(
            uuid=uuid.uuid4(),
            name="test_activity",
            description="test_description",
            pathways=[test_pathway],
        )

        scenario = Scenario(
            uuid=uuid.uuid4(),
            name="Scenario",
            description="Scenario description",
            activities=[test_activity],
            extent=spatial_extent,
            priority_layer_groups=[],
            weighted_activities=[]
        )

        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "pathways",
        )

        task_config = TaskConfig(
            scenario=scenario,
            priority_layers=priority_layers,
            priority_layer_groups=test_priority_groups,
            analysis_activities=scenario.activities,
            all_activities=[test_activity],
            snapping_enabled=False,
            snap_layer=snapping_layer,
            mask_layers_paths=",".join(mask_layers_paths),
            snap_rescale=False,
            snap_method=0,
            pathway_suitability_index=0.0,
            carbon_coefficient=1.0,
            sieve_enabled=False,
            sieve_threshold=10.0,
            ncs_with_carbon=True,
            landuse_project=True,
            landuse_normalized=True,
            landuse_weighted=True,
            highest_position=True,
            base_dir=base_dir,
            nodata_value=-9999.0,  # Set the nodata value to replace
        )

        analysis_task = ScenarioAnalysisTask(
            task_config=task_config
        )

        scenario_directory = os.path.join(
            f"{base_dir}",
            f'scenario_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
            f"_{str(uuid.uuid4())[:4]}",
        )

        analysis_task.scenario_directory = scenario_directory

        result = analysis_task.run_pathways_replace_nodata(
            nodata_value=task_config.nodata_value,
        )

        self.assertTrue(result)

        self.assertNotEqual(test_pathway.path, pathway_layer_path_1)

        result_layer = QgsRasterLayer(test_pathway.path, test_pathway.name)
        result_provider = result_layer.dataProvider()
        result_no_data_value = result_provider.sourceNoDataValue(1)
        self.assertEqual(result_no_data_value, task_config.nodata_value)   
        self.assertTrue(BaseFileUtils.remove_dir(scenario_directory))
    
    def test_scenario_layer_clipping(self):
        activities_layer_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "pathways", "layers"
        )

        mask_layers_directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "mask", "layers"
        )

        pathway_layer_path_1 = os.path.join(
            activities_layer_directory, "test_pathway_1.tif"
        )
        mask_layer_path_1 = os.path.join(mask_layers_directory, "test_mask_1.shp")

        test_pathway = NcsPathway(
            uuid=uuid.uuid4(),
            name="test_pathway",
            description="test_description",
            path=pathway_layer_path_1,
            priority_layers=[],
        )

        test_activity = Activity(
            uuid=uuid.uuid4(),
            name="test_activity",
            description="test_description",
            pathways=[test_pathway],
            path="",
            mask_paths=[],
        )

        activity_layer = QgsRasterLayer(test_activity.path, test_activity.name)

        test_extent = activity_layer.extent()

        spatial_extent = SpatialExtent(
            bbox=[test_extent.xMinimum(), test_extent.xMaximum(), test_extent.yMinimum(), test_extent.yMaximum()],
            crs=activity_layer.crs().authid()
        )

        scenario = Scenario(
            uuid=uuid.uuid4(),
            name="Scenario",
            description="Scenario description",
            activities=[test_activity],
            extent=spatial_extent,
            priority_layer_groups=[],
            weighted_activities=[]
        )

        task_config = TaskConfig(
            scenario=scenario,
            priority_layers=priority_layers,
            priority_layer_groups=test_priority_groups,
            analysis_activities=scenario.activities,
            all_activities=[test_activity],
            snapping_enabled=False,
            snap_layer=None,
            mask_layers_paths=",".join(mask_layers_paths),
            snap_rescale=False,
            snap_method=0, # 0 = Nearest neighbour, 1 = Bilinear, 2 = Cubic
            pathway_suitability_index=0.0,
            carbon_coefficient=1.0,
            sieve_enabled=False,
            sieve_threshold=10.0,
            ncs_with_carbon=True,
            landuse_project=True,
            landuse_normalized=True,
            landuse_weighted=True,
            highest_position=True,
            base_dir=os.path.dirname(os.path.abspath(__file__)),
            nodata_value=-9999.0,
            studyarea_path=mask_layer_path_1,
            clip_to_studyarea=True
        )

        analysis_task = ScenarioAnalysisTask(
            task_config= task_config,
        )

        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "activities",
        )

        scenario_directory = os.path.join(
            f"{base_dir}",
            f'scenario_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
            f"_{str(uuid.uuid4())[:4]}",
        )

        analysis_task.scenario_directory = scenario_directory

        pathway_layer = QgsRasterLayer(test_pathway.path, test_pathway.name)
        pathway_layer_stat = pathway_layer.dataProvider().bandStatistics(1)

        self.assertEqual(pathway_layer_stat.minimumValue, 1.0)
        self.assertEqual(pathway_layer_stat.maximumValue, 10.0)
        self.assertAlmostEqual(pathway_layer_stat.mean, 5.583, places=3)

        results = analysis_task.clip_analysis_data(
            studyarea_path=mask_layer_path_1
        )

        self.assertIsInstance(results, bool)
        self.assertTrue(results)

        self.assertIsNotNone(test_activity.path)
        self.assertNotEqual(test_pathway.path, pathway_layer_path_1)

        result_layer = QgsRasterLayer(test_pathway.path, test_activity.name)
        self.assertTrue(result_layer.isValid())

        result_stat = result_layer.dataProvider().bandStatistics(1)
        self.assertEqual(result_stat.minimumValue, 2.0)
        self.assertEqual(result_stat.maximumValue, 10.0)
        self.assertAlmostEqual(result_stat.mean, 6, places=3)
        self.assertTrue(BaseFileUtils.remove_dir(scenario_directory))

    def tearDown(self):
        pass
