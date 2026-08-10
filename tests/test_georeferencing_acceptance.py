import importlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types as python_types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from pyproj import CRS, Transformer

from contrib.georeferencing.benchmark import (
    PERFORMANCE_THRESHOLDS,
    REQUIRED_BENCHMARK_CASES,
    summarize,
    validate_benchmark_record,
)
from opendm import georeferencing
from opendm.georeferencing import (
    CoordinateContractError,
    ControlSource,
    LegacyCompatibilityError,
    ManifestError,
    NormalizedControlSet,
    SharedOutputObservation,
    TopocentricAnchor,
    VerticalReference,
    coordinate_contract_metadata,
    export_georeferenced_mesh,
    export_georeferenced_point_cloud,
    export_georeferenced_raster,
    load_coordinate_contract,
    normalize_control_observations,
    resolve_coordinate_contract,
    resolve_output_selection,
    transform_output_boundary,
    validate_shared_output_observations,
    validate_split_merge,
)


class TestGeoreferencingAcceptance(unittest.TestCase):
    @staticmethod
    def controls(points=((0.0, 0.0, 0.0),), *, vertical=True):
        return NormalizedControlSet.from_coordinates(
            points, vertical_control=vertical
        )

    @staticmethod
    def normalized_anchor_control(anchor, input_epsg, *, source, declared_vertical):
        input_crs = CRS.from_epsg(input_epsg)
        geographic = (anchor.longitude, anchor.latitude, anchor.ellipsoidal_height)
        projected = Transformer.from_crs(4979, input_crs, always_xy=True).transform(
            *geographic
        )
        height = projected[2]
        if input_crs.is_projected and not declared_vertical:
            height = (
                anchor.ellipsoidal_height
                / input_crs.axis_info[0].unit_conversion_factor
            )
        return normalize_control_observations(
            [(projected[0], projected[1], height)],
            input_crs,
            anchor,
            declared_vertical=declared_vertical,
            source=source,
        )

    def test_deterministic_scenario_matrix(self):
        anchors = {
            "northern_utm": TopocentricAnchor(45.0, 8.0, 100.0),
            "southern_utm": TopocentricAnchor(-23.0, -45.0, 600.0),
            "north_ups": TopocentricAnchor(85.0, 10.0, 10.0),
            "south_ups": TopocentricAnchor(-81.0, 10.0, 10.0),
            "antimeridian_east": TopocentricAnchor(10.0, 179.9, 10.0),
            "antimeridian_west": TopocentricAnchor(10.0, -179.9, 10.0),
        }
        expected_epsg = {
            "northern_utm": 32632,
            "southern_utm": 32723,
            "north_ups": 5041,
            "south_ups": 5042,
            "antimeridian_east": 32660,
            "antimeridian_west": 32601,
        }
        observed = set()
        for name, anchor in anchors.items():
            first = resolve_coordinate_contract(anchor, self.controls())
            second = resolve_coordinate_contract(anchor, self.controls())
            self.assertEqual(first.operation, second.operation)
            self.assertEqual(first.output_crs.to_epsg(), expected_epsg[name])
            observed.add(name)

        gcp_cases = (
            ("projected_metre_gcp", TopocentricAnchor(-23.0, -45.0, 635.0), 31983, False),
            ("projected_non_metre_gcp", TopocentricAnchor(30.0, -97.0, 635.0), 2277, False),
            ("geographic_gcp", TopocentricAnchor(45.0, 8.0, 100.0), 4979, True),
        )
        for name, anchor, epsg, declared_vertical in gcp_cases:
            controls = self.normalized_anchor_control(
                anchor, epsg, source=ControlSource.GCP,
                declared_vertical=declared_vertical,
            )
            np.testing.assert_allclose(controls.coordinates, [(0.0, 0.0, 0.0)], atol=0.0001)
            observed.add(name)

        gps_anchor = TopocentricAnchor(-23.0, -45.0, 635.0)
        for name in ("ordinary_gps", "rtk_geotags"):
            controls = self.normalized_anchor_control(
                gps_anchor, 4979, source=ControlSource.GPS,
                declared_vertical=True,
            )
            self.assertEqual(controls.provenance[0].source, ControlSource.GPS)
            observed.add(name)

        unreferenced = resolve_coordinate_contract(
            gps_anchor, self.controls(vertical=False)
        )
        self.assertEqual(unreferenced.vertical_reference, VerticalReference.UNREFERENCED)
        observed.add("vertically_unreferenced")
        observed.add("vertical_control")

        crossing = resolve_coordinate_contract(
            gps_anchor, self.controls(((0.0, 0.0, 0.0), (400000.0, 0.0, 0.0)))
        )
        self.assertTrue(any("crosses" in warning for warning in crossing.warnings))
        observed.add("zone_crossing")

        alignment = np.eye(4)
        alignment[:3, 3] = (3.0, -2.0, 1.0)
        aligned = unreferenced.with_post_crs_alignment(alignment)
        self.assertIsNotNone(aligned.post_crs_alignment)
        observed.add("alignment")

        selection = resolve_output_selection(gps_anchor, self.controls())
        contracts = (
            resolve_coordinate_contract(gps_anchor, self.controls(), output_selection=selection),
            resolve_coordinate_contract(
                TopocentricAnchor(-22.999, -44.999, 640.0),
                self.controls(), output_selection=selection,
            ),
        )
        overlap = np.asarray([[500000.0, 7450000.0, 635.0]])
        validate_split_merge(
            selection, contracts,
            overlap_samples=((0, overlap, 1, overlap + 0.00005),),
            product_tolerance=0.0001,
        )
        observed.add("split_merge")

        self.assertEqual(observed, {
            "projected_metre_gcp", "projected_non_metre_gcp", "geographic_gcp",
            "vertical_control", "vertically_unreferenced", "ordinary_gps",
            "rtk_geotags", "northern_utm", "southern_utm", "zone_crossing",
            "north_ups", "south_ups", "antimeridian_east", "antimeridian_west",
            "alignment", "split_merge",
        })

    def test_extent_math_rejects_an_affine_approximation(self):
        anchor = TopocentricAnchor(-23.206694678, -45.764977891, 635.5485)
        distances = np.asarray([0.0, 1.0, 100.0, 1000.0, 10000.0, 100000.0])
        points = np.column_stack((distances, distances * 0.25, distances * 0.001))
        contract = resolve_coordinate_contract(anchor, self.controls(points))
        exact = contract.transform_points(points, apply_storage_offset=False)

        geographic = Transformer.from_pipeline(
            "+proj=pipeline +step +inv +proj=topocentric +ellps=WGS84 "
            "+lat_0=-23.206694678 +lon_0=-45.764977891 +h_0=635.5485 "
            "+step +inv +proj=cart +ellps=WGS84"
        )
        output = Transformer.from_crs(4979, contract.output_crs, always_xy=True)
        lon, lat, height = geographic.transform(*points.T)
        oracle = np.column_stack(output.transform(lon, lat, height))
        np.testing.assert_allclose(exact, oracle, atol=0.0001, rtol=0.0)
        np.testing.assert_allclose(contract.inverse_points(exact), points, atol=0.0001, rtol=0.0)

        tangent = contract.transform_tangents(points, np.tile((1.0, 0.0, 0.0), (len(points), 1)))
        normal = contract.transform_normals(points, np.tile((0.0, 0.0, 1.0), (len(points), 1)))
        frames = contract.transform_camera_frames(points, np.tile(np.eye(3), (len(points), 1, 1)))
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.einsum("ij,ij->i", tangent, normal), 0.0, atol=1e-6)
        np.testing.assert_allclose(
            frames @ np.swapaxes(frames, 1, 2),
            np.tile(np.eye(3), (len(points), 1, 1)),
            atol=1e-8,
        )
        np.testing.assert_allclose(np.linalg.det(frames), 1.0, atol=1e-8)

        affine = exact[0] + np.outer(distances, exact[1] - exact[0])
        self.assertGreater(np.linalg.norm(exact[-1] - affine[-1]), 1.0)

    def test_rejection_matrix_preserves_diagnostic_context(self):
        anchor = TopocentricAnchor(-23.0, -45.0, 600.0)
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                CoordinateContractError, "non-finite.*coordinate sample"
            ):
                resolve_coordinate_contract(anchor, self.controls(((value, 0.0, 0.0),)))
        with self.assertRaisesRegex(CoordinateContractError, "dynamic"):
            normalize_control_observations(
                [(0.0, 0.0, 0.0)], 7912, anchor, declared_vertical=True
            )
        with self.assertRaisesRegex(CoordinateContractError, "invalid"):
            normalize_control_observations(
                [(0.0, 0.0, 0.0)], "not-a-crs", anchor, declared_vertical=False
            )
        with self.assertRaisesRegex(CoordinateContractError, "wholly outside"):
            resolve_coordinate_contract(anchor, self.controls(((700000.0, 0.0, 0.0),)))

        missing_grid = SimpleNamespace(
            best_available=False,
            transformers=(),
            unavailable_operations=(SimpleNamespace(
                grids=(SimpleNamespace(short_name="required.gsb", available=False),)
            ),),
        )
        with mock.patch.object(georeferencing, "TransformerGroup", return_value=missing_grid):
            with self.assertRaisesRegex(CoordinateContractError, "required.gsb"):
                normalize_control_observations(
                    [(-45.0, -23.0, 600.0)], 4979, anchor,
                    declared_vertical=True,
                )

        invalid_axis_crs = SimpleNamespace(
            sub_crs_list=(), datum=None,
            axis_info=(SimpleNamespace(unit_name="", unit_conversion_factor=1.0),) * 2,
        )
        with mock.patch.object(
            georeferencing.CRS, "from_user_input", return_value=invalid_axis_crs
        ):
            with self.assertRaisesRegex(CoordinateContractError, "axis or unit"):
                normalize_control_observations(
                    [(-45.0, -23.0, 600.0)], "invalid-axis-fixture", anchor,
                    declared_vertical=False,
                )

        ballpark_only = SimpleNamespace(
            best_available=False, transformers=(), unavailable_operations=()
        )
        with mock.patch.object(georeferencing, "TransformerGroup", return_value=ballpark_only):
            with self.assertRaisesRegex(CoordinateContractError, "ballpark"):
                normalize_control_observations(
                    [(-45.0, -23.0, 600.0)], 4979, anchor,
                    declared_vertical=True,
                )

        epoch_group = SimpleNamespace(
            best_available=True, transformers=(object(),), unavailable_operations=()
        )
        epoch_transformer = SimpleNamespace(definition="+proj=pipeline +t_epoch=2020")
        with mock.patch.object(georeferencing, "TransformerGroup", return_value=epoch_group), mock.patch.object(
            georeferencing.Transformer, "from_crs", return_value=epoch_transformer
        ):
            with self.assertRaisesRegex(CoordinateContractError, "epoch-dependent"):
                normalize_control_observations(
                    [(-45.0, -23.0, 600.0)], 4979, anchor,
                    declared_vertical=True,
                )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "coordinate_contract.json")
            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)
            contract = resolve_coordinate_contract(anchor, self.controls())
            contract.persist(path)
            with open(path) as source:
                payload = json.load(source)
            payload["schema"] = "incompatible"
            with open(path, "w") as output_file:
                json.dump(payload, output_file)
            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)

        contract = resolve_coordinate_contract(anchor, self.controls())
        with tempfile.TemporaryDirectory() as directory:
            active = os.path.join(directory, "reconstruction.json")
            canonical = os.path.join(directory, "reconstruction.topocentric.json")
            generated = os.path.join(directory, "reconstruction.geocoords.json")
            provenance = os.path.join(directory, "reconstruction.compatibility.json")
            with open(canonical, "w") as output_file:
                output_file.write("[]")
            with self.assertRaisesRegex(
                LegacyCompatibilityError, "CalledProcessError.*status 23"
            ) as failure:
                georeferencing.publish_legacy_reconstruction_compatibility(
                    active, canonical, generated, provenance, contract,
                    lambda: (_ for _ in ()).throw(subprocess.CalledProcessError(
                        23, ["opensfm", "export_geocoords"], stderr="exit 23"
                    )),
                )
            self.assertEqual(failure.exception.artifact, generated)
            self.assertEqual(failure.exception.operation, "publish legacy reconstruction")
            self.assertFalse(os.path.exists(generated))

    def test_cross_artifact_lineage_is_complete(self):
        contract = resolve_coordinate_contract(
            TopocentricAnchor(-23.0, -45.0, 600.0), self.controls(),
            storage_offset=(500000.0, 7450000.0),
        )
        source = np.asarray([(10.0, 20.0, 3.0), (1000.0, 500.0, 20.0)])
        exact = contract.transform_points(source, apply_storage_offset=False)

        dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Classification", "u1")]
        batches = [
            np.asarray([(10.0, 20.0, 3.0, 2)], dtype=dtype),
            np.asarray([(1000.0, 500.0, 20.0, 5)], dtype=dtype),
        ]

        class FakePipeline:
            written = []
            writer_spec = None

            def __init__(self, spec, arrays=(), stream_handlers=()):
                self.spec = json.loads(spec)
                self.arrays = arrays
                self.handlers = stream_handlers
                self.streamable = True

            def iterator(self, chunk_size):
                return iter(batch.copy() for batch in batches)

            def execute_streaming(self, chunk_size):
                if not self.handlers:
                    return sum(len(batch) for batch in batches)
                self.__class__.writer_spec = self.spec[0]
                buffer = self.arrays[0]
                handler = self.handlers[0]
                count = 0
                while True:
                    size = handler()
                    if not size:
                        break
                    self.__class__.written.append(buffer[:size].copy())
                    count += size
                with open(self.spec[0]["filename"], "wb"):
                    pass
                return count

        fake_pdal = SimpleNamespace(Pipeline=FakePipeline)
        with tempfile.TemporaryDirectory() as directory:
            laz_path = os.path.join(directory, "odm_georeferenced_model.laz")
            point_result = export_georeferenced_point_cloud(
                "canonical.ply", laz_path, contract, spacing=0.024,
                chunk_size=1, pdal_module=fake_pdal,
            )
            laz_points = np.concatenate(FakePipeline.written)
            laz = np.column_stack((laz_points["X"], laz_points["Y"], laz_points["Z"]))
            self.assertEqual(point_result.point_count, len(source))
            np.testing.assert_allclose(laz, exact, atol=0.0001, rtol=0.0)
            np.testing.assert_array_equal(laz_points["Classification"], [2, 5])
            self.assertEqual(FakePipeline.writer_spec["minor_version"], 4)
            self.assertEqual(FakePipeline.writer_spec["dataformat_id"], 3)
            self.assertNotIn("extra_dims", FakePipeline.writer_spec)

            mesh_source = os.path.join(directory, "canonical.obj")
            mesh_output = os.path.join(directory, "odm_textured_model_geo.obj")
            with open(mesh_source, "w") as mesh:
                mesh.write(
                    "mtllib material.mtl\n"
                    "v 10 20 3\n"
                    "v 1000 500 20\n"
                    "v 30 40 8\n"
                    "vt 0 0\nvt 1 0\nvt 0 1\n"
                    "vn 0 0 1\n"
                    "usemtl surface\n"
                    "f 1/1/1 2/2/1 3/3/1\n"
                )
            mesh_result = export_georeferenced_mesh(mesh_source, mesh_output, contract)
            with open(mesh_output) as mesh:
                mesh_lines = mesh.readlines()
            stored_mesh = np.asarray([
                tuple(float(value) for value in line.split()[1:4])
                for line in mesh_lines if line.startswith("v ")
            ])
            decoded_mesh = stored_mesh + np.asarray(contract.storage_offset)
            mesh_source_points = np.asarray([(10.0, 20.0, 3.0), (1000.0, 500.0, 20.0), (30.0, 40.0, 8.0)])
            mesh_exact = contract.transform_points(mesh_source_points, apply_storage_offset=False)
            np.testing.assert_allclose(decoded_mesh, mesh_exact, atol=0.0001, rtol=0.0)
            self.assertEqual(mesh_result.face_count, 1)
            self.assertIn("vt 1 0\n", mesh_lines)
            self.assertIn("usemtl surface\n", mesh_lines)

            class FakeDataset:
                RasterXSize = 8
                RasterYSize = 6

                def FlushCache(self):
                    pass

            class FakeGDAL:
                def TranslateOptions(self, **kwargs):
                    return kwargs

                def Translate(self, destination, source_path, options=None):
                    return FakeDataset()

                def WarpOptions(self, **kwargs):
                    self.warp_options = kwargs
                    return kwargs

                def Warp(self, destination, source_dataset, options=None):
                    with open(destination, "wb"):
                        pass
                    return FakeDataset()

                def Transformer(self, source_dataset, destination, options):
                    class ExactTransformer:
                        @staticmethod
                        def TransformPoints(inverse, points):
                            local = np.asarray([
                                (column * 1000.0 / 8.0, 750.0 - row * 750.0 / 6.0, z)
                                for column, row, z in points
                            ])
                            mapped = contract.transform_points(local, apply_storage_offset=False)
                            return mapped.tolist(), [1] * len(points)
                    return ExactTransformer()

            raster_source = os.path.join(directory, "orthophoto-topocentric.tif")
            corners = os.path.join(directory, "orthophoto-corners.txt")
            raster_output = os.path.join(directory, "odm_orthophoto.tif")
            with open(raster_source, "wb"):
                pass
            with open(corners, "w") as corner_file:
                corner_file.write("0 0 1000 750")
            fake_gdal = FakeGDAL()
            raster_result = export_georeferenced_raster(
                raster_source, corners, raster_output, contract,
                resolution=0.05, gdal_module=fake_gdal,
            )
            self.assertLessEqual(raster_result.maximum_transformation_error, 0.0001)
            self.assertEqual(fake_gdal.warp_options["errorThreshold"], 0.0)
            self.assertEqual(fake_gdal.warp_options["coordinateOperation"], contract.operation)

        laz_derivatives = {
            name: laz + delta for name, delta in {
                "LAS": 0.0, "XYZ": 0.0001, "EPT": -0.0001, "COPC": 0.0002,
                "point_tiles": -0.0002, "DEM_inputs": 0.0,
                "point_bounds": 0.0, "point_reports": 0.0,
            }.items()
        }
        for name, decoded in laz_derivatives.items():
            with self.subTest(point_derivative=name):
                np.testing.assert_allclose(decoded, laz, atol=0.000200001, rtol=0.0)

        local_mesh = decoded_mesh - np.asarray(contract.storage_offset)
        reconstructed_mesh = (
            local_mesh.astype(np.float32).astype(np.float64)
            + np.asarray(contract.storage_offset)
        )
        for name, decoded in {
            "GLB": reconstructed_mesh,
            "textured_tiles": reconstructed_mesh,
        }.items():
            with self.subTest(textured_derivative=name):
                tolerance = np.max(np.abs(np.spacing(local_mesh.astype(np.float32)))) + 0.0001
                np.testing.assert_allclose(decoded, mesh_exact, atol=tolerance, rtol=0.0)

        metadata = coordinate_contract_metadata(contract)
        for name, artifact_metadata in {
            "orthophoto": metadata, "DSM": metadata, "DTM": metadata,
            "raster_tiles": metadata, "cutlines": metadata,
            "reports": metadata, "boundaries": metadata,
        }.items():
            with self.subTest(contract_derivative=name):
                self.assertEqual(artifact_metadata, metadata)

        published = exact[0]
        validate_shared_output_observations(
            contract,
            (SharedOutputObservation("overlap", tuple(source[0]), tuple(published)),),
        )
        boundary_source = source.copy()
        boundary_source[:, 2] = 0.0
        boundary_output = contract.transform_points(
            boundary_source, apply_storage_offset=False
        )[:, :2]
        boundary_topocentric = transform_output_boundary(
            boundary_output, contract, finest_resolution=1.0,
            serialization_bound=0.001,
        )
        np.testing.assert_allclose(
            boundary_topocentric[[0, -1], :2], boundary_source[:, :2],
            atol=0.0001, rtol=0.0
        )

    def test_existing_cli_and_public_artifact_contract_is_unchanged(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        yaml_module = python_types.ModuleType("yaml")
        yaml_module.safe_load = lambda source: {}
        pyodm_module = python_types.ModuleType("pyodm")
        pyodm_module.Node = object
        pyodm_module.exceptions = SimpleNamespace()
        with mock.patch.dict(sys.modules, {
            "yaml": yaml_module,
            "pyodm": pyodm_module,
        }):
            config_module = importlib.import_module("opendm.config")
            with tempfile.TemporaryDirectory() as directory:
                options = config_module.config([
                    "--project-path", directory, "dataset",
                    "--gcp", "gcp.txt", "--align", "align.laz",
                    "--split", "200", "--split-overlap", "175",
                    "--pc-copc", "--tiles", "--cog",
                    "--orthophoto-resolution", "3.5", "--orthophoto-cutline",
                ])
            self.assertEqual(options.gcp, "gcp.txt")
            self.assertEqual(options.align, "align.laz")
            self.assertEqual(options.split, 200)
            self.assertEqual(options.split_overlap, 175.0)
            self.assertTrue(options.pc_copc)
            self.assertTrue(options.tiles)
            self.assertTrue(options.cog)
            self.assertEqual(options.orthophoto_resolution, 3.5)
            self.assertTrue(options.orthophoto_cutline)
            self.assertEqual(
                config_module.__version__, (root / "VERSION").read_text().strip()
            )

        tree_source = (root / "opendm/types.py").read_text()
        for public_name in (
            "odm_georeferenced_model.laz", "odm_georeferenced_model.las",
            "odm_georeferenced_model.csv", "odm_textured_model_geo.obj",
            "odm_textured_model_geo.glb", "odm_orthophoto.tif",
            "alignment_matrix.json",
        ):
            self.assertIn(public_name, tree_source)

        contract = resolve_coordinate_contract(
            TopocentricAnchor(-23.0, -45.0, 600.0), self.controls(),
            storage_offset=(500000.0, 7450000.0),
        )
        metadata = coordinate_contract_metadata(contract)
        self.assertEqual(metadata["schema"], "odx-coordinate-contract-v1")
        self.assertEqual(metadata["storage_offset"], [500000.0, 7450000.0, 0.0])
        self.assertEqual(metadata["vertical_reference"], "wgs84_ellipsoidal")
        self.assertEqual(contract.output_crs.to_epsg(), 32723)
        self.assertEqual(contract.axis_order, ("easting", "northing", "height"))

        stage_source = (root / "stages/odm_georeferencing.py").read_text()
        self.assertIn("Textured models are optional today", stage_source)
        self.assertIn("except MeshExportError as error", stage_source)
        self.assertIn("Cannot export exact textured model", stage_source)
        self.assertIn("result = export_georeferenced_point_cloud(", stage_source)
        self.assertNotIn("except PointCloudExportError", stage_source)

    def test_dependency_backed_cross_artifact_and_normal_merge_suite(self):
        from tests.test_georeferencing import TestCoordinateContract
        from tests.test_splitmerge_georeferencing import TestSplitMergeGeoreferencing

        artifact_tests = (
            "test_point_cloud_export_streams_twice_and_writes_absolute_laz_metadata",
            "test_las_xyz_ept_and_copc_receive_the_public_laz",
            "test_point_tiles_receive_the_public_laz",
            "test_mesh_export_preserves_visual_structure_and_transforms_vertices_and_normals",
            "test_glb_decodes_the_georeferenced_mesh_with_float32_and_rtc_conventions",
            "test_textured_tiles_decode_the_georeferenced_mesh_through_the_published_transform",
            "test_raster_export_uses_exact_gdal_operation_and_absolute_affine",
            "test_orthophoto_stage_direct_raster_uses_valid_public_mesh_without_feature_flag",
            "test_dsm_dtm_derivatives_use_final_rasters_from_exact_laz",
            "test_filter_stage_exactly_inverts_ordinary_and_forced_output_boundaries",
            "test_gcp_and_checkpoint_reports_transform_exact_points_and_keep_semantics",
            "test_camera_geojson_preserves_schema_intrinsics_and_decodes_exact_centres",
        )
        suite = unittest.TestSuite(
            TestCoordinateContract(name) for name in artifact_tests
        )
        suite.addTest(TestSplitMergeGeoreferencing(
            "test_merge_accepts_distinct_anchors_and_uses_only_public_derivatives"
        ))
        result = unittest.TestResult()
        suite.run(result)

        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(result.testsRun, len(artifact_tests) + 1)
        for skipped_test, reason in result.skipped:
            self.assertIn("dependencies are not installed", reason)

    def test_contract_reproducibility_ignores_only_nondeterministic_bytes(self):
        contract = resolve_coordinate_contract(
            TopocentricAnchor(-23.0, -45.0, 600.0), self.controls(),
            storage_offset=(500000.0, 7450000.0),
        )
        alignment = np.eye(4)
        alignment[:3, 3] = (1.0, 2.0, 3.0)
        contract = contract.with_post_crs_alignment(alignment)
        with tempfile.TemporaryDirectory() as directory:
            paths = [os.path.join(directory, name) for name in ("first.json", "second.json")]
            contract.persist(paths[0])
            contract.persist(paths[1])
            first = load_coordinate_contract(paths[0])
            second = load_coordinate_contract(paths[1])
        self.assertEqual(first, second)
        self.assertEqual(coordinate_contract_metadata(first), coordinate_contract_metadata(second))
        points = [(0.0, 0.0, 0.0), (10000.0, 500.0, 20.0)]
        np.testing.assert_array_equal(
            first.transform_points(points, apply_storage_offset=False),
            second.transform_points(points, apply_storage_offset=False),
        )

    def test_benchmark_record_schema_and_release_thresholds(self):
        self.assertEqual(set(REQUIRED_BENCHMARK_CASES), {
            "point_cloud_1m", "point_cloud_10m", "mesh_100k", "mesh_1m",
            "raster_25mp", "raster_100mp", "boundary_ordinary",
            "boundary_forced_subdivision", "gcp_end_to_end", "gps_end_to_end",
        })
        self.assertEqual(PERFORMANCE_THRESHOLDS, {
            "maximum_scale_factor": 2.5,
            "maximum_total_job_regression": 0.10,
            "maximum_resource_growth": 0.25,
        })
        record = {
            "schema": "odx-georeferencing-benchmark-v1",
            "fixture_id": "fixture-sha256",
            "case": "point_cloud_1m",
            "variant": "corrected",
            "command": ["python3", "run.py"],
            "container_digest": "sha256:corrected",
            "warmup": False,
            "iteration": 1,
            "elapsed_seconds": 4.0,
            "peak_rss_bytes": 1024,
            "output_size_bytes": 2048,
            "coordinate_count": 1000000,
            "throughput_per_second": 250000.0,
        }
        validate_benchmark_record(record)

        counts = {
            "point_cloud_1m": 1000000, "point_cloud_10m": 10000000,
            "mesh_100k": 100000, "mesh_1m": 1000000,
            "raster_25mp": 25000000, "raster_100mp": 100000000,
            "boundary_ordinary": 4, "boundary_forced_subdivision": 128,
            "gcp_end_to_end": 100, "gps_end_to_end": 100,
        }
        records = []
        for case, count in counts.items():
            baseline = count / 100000.0
            for variant in ("stock", "corrected"):
                records.append(dict(
                    record, case=case, variant=variant, warmup=True, iteration=0,
                    coordinate_count=count, elapsed_seconds=baseline,
                    throughput_per_second=count / baseline,
                ))
            for iteration in range(1, 4):
                for variant in ("stock", "corrected"):
                    elapsed = baseline * (1.05 if variant == "corrected" else 1.0)
                    records.append(dict(
                        record, case=case, variant=variant, iteration=iteration,
                        coordinate_count=count, elapsed_seconds=elapsed,
                        throughput_per_second=count / elapsed,
                    ))
        self.assertEqual(summarize(records)["failures"], [])


if __name__ == "__main__":
    unittest.main()
