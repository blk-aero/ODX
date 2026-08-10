import json
import os
import shutil
import struct
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
from pyproj import CRS, Transformer

from opendm import georeferencing

try:
    import pdal
except ImportError:
    pdal = None

try:
    from osgeo import gdal
except ImportError:
    gdal = None

from opendm.georeferencing import (
    CoordinateContractError,
    ControlProvenance,
    ControlSource,
    LasEncoding,
    ManifestError,
    MeshExportError,
    PointCloudExportError,
    PUBLIC_LAZ_DIMENSIONS,
    RasterExportError,
    RasterExportResult,
    NormalizedControlSet,
    OutputSelection,
    SharedOutputObservation,
    TopocentricAnchor,
    VerticalReference,
    load_coordinate_contract,
    load_output_selection,
    load_split_merge_derivatives,
    export_georeferenced_point_cloud,
    export_georeferenced_mesh,
    export_georeferenced_raster,
    coordinate_contract_metadata,
    decoded_coordinate_bounds,
    normalize_control_observations,
    resolve_coordinate_contract,
    resolve_output_selection,
    persist_split_merge_derivatives,
    public_point_cloud_batch,
    split_merge_derivative_paths,
    resolve_stage_coordinate_contract,
    select_las_encoding,
    transform_point_cloud_batch,
    transform_output_boundary,
    transform_camera_poses,
    validate_split_merge,
    validate_shared_output_observations,
)

try:
    from opendm import point_cloud as point_cloud_module, system as odm_system, types
    from opendm.gltf import obj2glb
    from opendm.ogctiles import build_3dtiles
    from opendm.osfm import OSFMContext
    from opendm.shots import get_geojson_shots_from_opensfm, get_rotation_matrix
    from stages.odm_dem import ODMDEMStage
    from stages.odm_georeferencing import ODMGeoreferencingStage
    from stages.odm_filterpoints import ODMFilterPoints
    from stages.odm_orthophoto import ODMOrthoPhotoStage
    from stages.odm_report import ODMReport
except ImportError:
    point_cloud_module = odm_system = types = obj2glb = build_3dtiles = OSFMContext = get_geojson_shots_from_opensfm = get_rotation_matrix = ODMDEMStage = ODMFilterPoints = ODMGeoreferencingStage = ODMOrthoPhotoStage = ODMReport = None


class GeoreferencedReconstructionFixture:
    photos = []
    multi_camera = []

    @staticmethod
    def is_georeferenced():
        return True

    @staticmethod
    def has_gcp():
        return False

    @staticmethod
    def has_geotagged_photos():
        return True

    @staticmethod
    def get_proj_offset():
        return (421000.0, 7433000.0)


class TestCoordinateContract(unittest.TestCase):
    def setUp(self):
        self.anchor = TopocentricAnchor(-23.206694678, -45.764977891, 635.5485)

    def controls(self, coordinates, *, vertical_control=True, provenance=()):
        return NormalizedControlSet.from_coordinates(
            coordinates,
            vertical_control=vertical_control,
            provenance=provenance,
        )

    @staticmethod
    def alignment_fixture():
        return np.asarray([
            [0.0, -2.0, 0.0, 7.0],
            [0.5, 0.0, 0.0, -3.0],
            [0.0, 0.0, 1.5, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    @staticmethod
    def translated_contract(contract):
        alignment = np.eye(4)
        alignment[:3, 3] = [3.0, -2.0, 0.5]
        return contract.with_post_crs_alignment(alignment)

    @staticmethod
    def point_cloud_export_stub(point_count):
        def export(source, output, contract, **kwargs):
            open(output, "wb").close()
            return SimpleNamespace(
                point_count=point_count,
                encoding=LasEncoding((0.001,) * 3, (421000.0, 7433000.0, 635.0)),
            )
        return export

    def test_output_selection_is_anchor_independent_and_serializable(self):
        selection = resolve_output_selection(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0)], vertical_control=False),
            storage_offset=(421000.0, 7433000.0),
        )

        self.assertIsInstance(selection, OutputSelection)
        payload = selection.to_dict()
        self.assertNotIn("anchor", payload)
        self.assertNotIn("operation", payload)
        self.assertNotIn("post_crs_alignment", payload)
        self.assertEqual(payload["vertical_reference"], "unreferenced")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "output_selection.json")
            selection.persist(path)
            self.assertEqual(load_output_selection(path), selection)

    def test_distinct_submodel_anchors_combine_with_one_output_selection(self):
        anchors = (
            TopocentricAnchor(-23.2067, -45.7650, 635.0),
            TopocentricAnchor(-23.2050, -45.7620, 642.0),
        )
        controls = self.controls([(0.0, 0.0, 0.0)], vertical_control=True)
        selection = resolve_output_selection(
            anchors[0], controls, storage_offset=(421000.0, 7433000.0)
        )
        contracts = [
            resolve_coordinate_contract(anchor, controls, output_selection=selection)
            for anchor in anchors
        ]
        shared_geographic = (-45.7635, -23.2058, 650.0)

        def topocentric(anchor):
            pipeline = (
                "+proj=pipeline +step +proj=cart +ellps=WGS84 "
                "+step +proj=topocentric +ellps=WGS84 "
                "+lat_0={0:.17g} +lon_0={1:.17g} +h_0={2:.17g}"
            ).format(anchor.latitude, anchor.longitude, anchor.ellipsoidal_height)
            return Transformer.from_pipeline(pipeline).transform(*shared_geographic)

        exported = [
            contract.transform_points([topocentric(anchor)], apply_storage_offset=False)[0]
            for anchor, contract in zip(anchors, contracts)
        ]
        expected = np.asarray(
            Transformer.from_crs(CRS.from_epsg(4979), selection.output_crs, always_xy=True)
            .transform(*shared_geographic)
        )

        self.assertNotEqual(contracts[0].anchor, contracts[1].anchor)
        self.assertNotEqual(contracts[0].operation, contracts[1].operation)
        np.testing.assert_allclose(exported[0], expected, atol=0.0001)
        np.testing.assert_allclose(exported[1], expected, atol=0.0001)
        np.testing.assert_allclose(exported[0], exported[1], atol=0.0001)

    def test_split_merge_validates_contract_metadata_alignment_and_overlap(self):
        for latitude, longitude in ((45.0, 8.0), (-23.0, -45.0)):
            with self.subTest(latitude=latitude):
                controls = self.controls([(0.0, 0.0, 0.0)])
                first_anchor = TopocentricAnchor(latitude, longitude, 100.0)
                second_anchor = TopocentricAnchor(latitude + 0.002, longitude + 0.003, 110.0)
                selection = resolve_output_selection(
                    first_anchor, controls, storage_offset=(400000.0, 5000000.0)
                )
                alignment = np.eye(4)
                alignment[:3, 3] = [2.0, -3.0, 1.0]
                contracts = tuple(
                    resolve_coordinate_contract(
                        anchor, controls, output_selection=selection
                    ).with_post_crs_alignment(alignment)
                    for anchor in (first_anchor, second_anchor)
                )
                overlap = np.asarray([[420001.25, 7433002.5, 107.0]])

                validate_split_merge(
                    selection, contracts,
                    overlap_samples=((0, overlap, 1, overlap + 0.00005),),
                    product_tolerance=0.00009,
                )

                with self.assertRaisesRegex(CoordinateContractError, "storage offset"):
                    validate_split_merge(
                        selection,
                        (contracts[0], replace(
                            contracts[1], storage_offset=(1.0, 2.0, 0.0)
                        )),
                    )
                with self.assertRaisesRegex(CoordinateContractError, "output selection"):
                    validate_split_merge(
                        replace(selection, geometry_unit="foot"), contracts
                    )
                with self.assertRaisesRegex(CoordinateContractError, "vertical-reference"):
                    validate_split_merge(
                        selection,
                        (contracts[0], replace(
                            contracts[1],
                            vertical_reference=VerticalReference.UNREFERENCED,
                        )),
                    )
                with self.assertRaisesRegex(CoordinateContractError, "coordinate operation"):
                    validate_split_merge(
                        selection,
                        (contracts[0], replace(
                            contracts[1], operation=contracts[1].operation + " invalid"
                        )),
                    )
                with self.assertRaisesRegex(CoordinateContractError, "coordinate operation"):
                    validate_split_merge(
                        selection,
                        (contracts[0], replace(
                            contracts[1], axis_order=("northing", "easting", "height")
                        )),
                    )
                with self.assertRaisesRegex(CoordinateContractError, "post-CRS alignment"):
                    validate_split_merge(
                        selection,
                        (contracts[0], contracts[1].without_post_crs_alignment()),
                    )
                with self.assertRaisesRegex(CoordinateContractError, "overlap"):
                    validate_split_merge(
                        selection, contracts,
                        overlap_samples=((0, overlap, 1, overlap + 0.001),),
                    )

    def test_split_overlap_keeps_product_error_separate_from_export_error(self):
        controls = self.controls([(0.0, 0.0, 0.0)])
        anchors = (
            TopocentricAnchor(-23.2067, -45.7650, 635.0),
            TopocentricAnchor(-23.2050, -45.7620, 642.0),
        )
        selection = resolve_output_selection(anchors[0], controls)
        contracts = tuple(
            resolve_coordinate_contract(anchor, controls, output_selection=selection)
            for anchor in anchors
        )
        published = np.asarray([[421500.0, 7433500.0, 650.0]])

        validate_split_merge(
            selection, contracts,
            overlap_samples=((0, published, 1, published + [0.5, 0.0, 0.0]),),
            product_tolerance=1.0,
        )
        with self.assertRaisesRegex(
            CoordinateContractError,
            "product tolerance",
        ):
            validate_split_merge(
                selection, contracts,
                overlap_samples=((0, published, 1, published + [2.0, 0.0, 0.0]),),
                product_tolerance=1.0,
            )

    def test_published_overlap_uses_only_transformation_and_representation_error(self):
        contract = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0)])
        )
        topocentric = (10.0, 20.0, 3.0)
        expected = contract.transform_points(
            [topocentric], apply_storage_offset=False
        )[0]

        validate_shared_output_observations(
            contract,
            (SharedOutputObservation(
                "shared.jpg", topocentric, tuple(expected + 0.00005)
            ),),
            transformation_tolerance=0.0001,
            representation_tolerance=0.00001,
        )
        with self.assertRaisesRegex(
            CoordinateContractError, "transformation and representation tolerances"
        ):
            validate_shared_output_observations(
                contract,
                (SharedOutputObservation(
                    "shared.jpg", topocentric, tuple(expected + 0.001)
                ),),
                transformation_tolerance=0.0001,
                representation_tolerance=0.00001,
            )

    def test_split_merge_derivative_manifest_rejects_stale_contract_or_artifact_set(self):
        contract = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0)])
        )
        with tempfile.TemporaryDirectory() as directory:
            georeferencing = os.path.join(directory, "odm_georeferencing")
            os.makedirs(georeferencing)
            point_cloud = os.path.join(
                georeferencing, "odm_georeferenced_model.laz"
            )
            open(point_cloud, "wb").close()
            persist_split_merge_derivatives(directory, contract)

            self.assertEqual(
                load_split_merge_derivatives(directory, contract)[0],
                (os.path.join(
                    "odm_georeferencing", "odm_georeferenced_model.laz"
                ),),
            )
            with self.assertRaisesRegex(ManifestError, "compatible merge derivatives"):
                load_split_merge_derivatives(
                    directory, replace(
                        contract, storage_offset=(1.0, 2.0, 0.0)
                    )
                )
            os.makedirs(os.path.join(directory, "orthophoto_tiles"))
            with self.assertRaisesRegex(ManifestError, "derivative set changed"):
                load_split_merge_derivatives(directory, contract)

    def test_split_merge_derivative_manifest_covers_every_public_merge_family(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = (
                os.path.join("odm_georeferencing", "odm_georeferenced_model.laz"),
                os.path.join("odm_georeferencing", "odm_georeferenced_model.bounds.gpkg"),
                os.path.join("odm_orthophoto", "odm_orthophoto.tif"),
                os.path.join("odm_dem", "dsm.tif"),
                os.path.join("odm_report", "shots.geojson"),
                os.path.join("odm_texturing", "odm_textured_model_geo.obj"),
                "cameras.json",
            )
            for artifact in artifacts:
                path = os.path.join(directory, artifact)
                os.makedirs(os.path.dirname(path) or directory, exist_ok=True)
                open(path, "w").close()
            for artifact in ("orthophoto_tiles", "3d_tiles", "entwine_pointcloud"):
                os.makedirs(os.path.join(directory, artifact))

            discovered = split_merge_derivative_paths(directory)

            self.assertTrue(set(artifacts).issubset(discovered))
            self.assertIn("orthophoto_tiles", discovered)
            self.assertIn("3d_tiles", discovered)
            self.assertIn("entwine_pointcloud", discovered)

    @staticmethod
    def write_textured_triangle(directory, filename):
        import rasterio
        from rasterio.transform import Affine

        texture = os.path.join(directory, "texture.jpg")
        with rasterio.open(
            texture, "w", driver="JPEG", width=1, height=1, count=3,
            dtype="uint8", transform=Affine.identity(),
        ) as image:
            image.write(np.zeros((3, 1, 1), dtype=np.uint8))
        with open(os.path.join(directory, "mesh.mtl"), "w") as material:
            material.write("newmtl atlas\nmap_Kd texture.jpg\n")
        path = os.path.join(directory, filename)
        with open(path, "w") as mesh:
            mesh.write(
                "mtllib mesh.mtl\n"
                "v 0 0 0\nv 100 0 0\nv 0 20 1\n"
                "vt 0 0\nvt 1 0\nvt 0 1\n"
                "vn 0 -0.04993761694389223 0.9987523388778446\n"
                "usemtl atlas\nf 1/1/1 2/2/1 3/3/1\n"
            )
        return path

    @staticmethod
    def decode_gltf_positions(path):
        import pygltflib

        gltf = pygltflib.GLTF2().load(path)
        primitive = gltf.meshes[0].primitives[0]
        accessor = gltf.accessors[primitive.attributes.POSITION]
        view = gltf.bufferViews[accessor.bufferView]
        start = (view.byteOffset or 0) + (accessor.byteOffset or 0)
        positions = np.frombuffer(
            gltf.binary_blob(), dtype=np.float32,
            count=accessor.count * 3, offset=start,
        ).reshape((-1, 3))
        return gltf, positions

    def stage_fixture(self, directory, *, crop=0):
        args = SimpleNamespace(
            use_exif=False, crop=crop, fast_orthophoto=False,
            optimize_disk_space=False, rerun=None, rerun_all=True,
            rerun_from=None,
        )
        tree = types.ODM_Tree(directory)
        os.makedirs(tree.opensfm)
        os.makedirs(tree.odm_georeferencing)
        os.makedirs(tree.odm_filterpoints)
        with open(tree.path("opensfm", "reference_lla.json"), "w") as reference:
            json.dump(
                {"latitude": -23.206694678, "longitude": -45.764977891, "altitude": 635.5485},
                reference,
            )
        open(tree.path("opensfm", "fresh_reconstruction.marker"), "w").close()
        reconstruction = GeoreferencedReconstructionFixture()
        return args, tree, {"tree": tree, "reconstruction": reconstruction}

    def test_automatic_output_crs_uses_anchor_for_utm_and_ups(self):
        utm = resolve_coordinate_contract(self.anchor, self.controls([(0.0, 0.0, 0.0)]))
        north_ups = resolve_coordinate_contract(
            TopocentricAnchor(85.0, 10.0, 12.0), self.controls([(0.0, 0.0, 0.0)])
        )
        south_ups = resolve_coordinate_contract(
            TopocentricAnchor(-81.0, 10.0, 12.0), self.controls([(0.0, 0.0, 0.0)])
        )

        self.assertEqual(utm.output_crs.to_epsg(), 32723)
        self.assertEqual(north_ups.output_crs.to_epsg(), 5041)
        self.assertEqual(south_ups.output_crs.to_epsg(), 5042)
        self.assertEqual(utm.axis_order, ("easting", "northing", "height"))
        self.assertEqual(utm.geometry_unit, "metre")

    def test_points_match_an_independent_proj_composition(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100000.0, -40000.0, 250.0)]),
            storage_offset=(1000.0, 2000.0),
        )
        points = np.array([[0.0, 0.0, 0.0], [100000.0, -40000.0, 250.0]])

        actual = contract.transform_points(points, apply_storage_offset=False)
        topo_to_geographic = Transformer.from_pipeline(
            "+proj=pipeline "
            "+step +inv +proj=topocentric +ellps=WGS84 "
            "+lat_0=-23.206694678 +lon_0=-45.764977891 +h_0=635.5485 "
            "+step +inv +proj=cart +ellps=WGS84"
        )
        output = Transformer.from_crs(CRS.from_epsg(4979), contract.output_crs, always_xy=True)
        lon, lat, height = topo_to_geographic.transform(
            points[:, 0], points[:, 1], points[:, 2]
        )
        expected = np.column_stack(output.transform(lon, lat, height))

        np.testing.assert_allclose(actual, expected, atol=0.0001, rtol=0.0)
        np.testing.assert_allclose(contract.inverse_points(actual), points, atol=0.0001, rtol=0.0)
        np.testing.assert_allclose(
            contract.transform_points(points)[:, :2],
            actual[:, :2] - np.array([1000.0, 2000.0]),
            atol=1e-10,
        )

    def test_directions_and_camera_frames_use_local_differential(self):
        contract = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0), (100000.0, 100000.0, 100.0)])
        )
        points = np.array([[100000.0, 100000.0, 100.0]])
        tangents = contract.transform_tangents(points, [[1.0, 0.0, 0.0]])
        normals = contract.transform_normals(points, [[0.0, 0.0, 1.0]])
        frames = contract.transform_camera_frames(points, [np.eye(3)])
        epsilon = 0.001
        topo_to_geographic = Transformer.from_pipeline(
            "+proj=pipeline "
            "+step +inv +proj=topocentric +ellps=WGS84 "
            "+lat_0=-23.206694678 +lon_0=-45.764977891 +h_0=635.5485 "
            "+step +inv +proj=cart +ellps=WGS84"
        )
        output = Transformer.from_crs(CRS.from_epsg(4979), contract.output_crs, always_xy=True)

        def oracle(point):
            lon, lat, height = topo_to_geographic.transform(*point)
            return np.asarray(output.transform(lon, lat, height))

        expected_tangent = (
            oracle(points[0] + [epsilon, 0.0, 0.0])
            - oracle(points[0] - [epsilon, 0.0, 0.0])
        ) / (2.0 * epsilon)

        np.testing.assert_allclose(tangents[0], expected_tangent, rtol=1e-6, atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(normals[0])), 1.0, places=6)
        self.assertAlmostEqual(float(np.dot(tangents[0], normals[0])), 0.0, places=6)
        np.testing.assert_allclose(frames[0].T @ frames[0], np.eye(3), atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.det(frames[0])), 1.0, places=8)

    def test_output_boundary_vertices_round_trip_exactly_without_offset_approximation(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100000.0, 100000.0, 100.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        topocentric = np.array([
            [0.0, 0.0, 0.0],
            [100000.0, 0.0, 0.0],
            [100000.0, 100000.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        output = contract.transform_points(
            topocentric, apply_storage_offset=False
        )[:, :2]

        transformed = transform_output_boundary(
            output,
            contract,
            finest_resolution=1000000.0,
            serialization_bound=1e-9,
        )

        decoded_output = contract.transform_points(
            transformed, apply_storage_offset=False
        )
        np.testing.assert_allclose(decoded_output[:, :2], output, atol=0.0001, rtol=0.0)
        offset_approximation = output - np.asarray(contract.storage_offset)[:2]
        self.assertGreater(
            np.max(np.abs(offset_approximation - topocentric[:, :2])), 1.0
        )

    def test_output_boundary_subdivides_only_above_resolution_and_serialization_tolerance(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100000.0, 100000.0, 100.0)]),
        )
        source = np.array([[-100000.0, 80000.0, 0.0], [100000.0, 80000.0, 0.0]])
        output = contract.transform_points(source, apply_storage_offset=False)

        forced = transform_output_boundary(
            output,
            contract,
            finest_resolution=0.01,
            serialization_bound=1e-9,
        )
        serialization_limited = transform_output_boundary(
            output,
            contract,
            finest_resolution=0.01,
            serialization_bound=1000000.0,
        )

        self.assertGreater(len(forced), len(output))
        self.assertEqual(len(serialization_limited), len(output))
        midpoint = output[0] + (output[1] - output[0]) * 0.5
        expected_midpoint = contract.inverse_points([midpoint])[0]
        self.assertTrue(np.any(np.linalg.norm(forced - expected_midpoint, axis=1) < 0.0001))

    def test_decoded_bounds_contain_complete_geometry_with_tight_tolerance(self):
        coordinates = np.array([
            [10.0, 20.0, -3.0],
            [11.0, 27.0, 4.0],
            [19.0, 21.0, 2.0],
            [12.0, 23.0, 1.0],
        ])

        bounds = decoded_coordinate_bounds(coordinates, tolerance=0.025)

        np.testing.assert_allclose(bounds.minimum, [9.975, 19.975, -3.025])
        np.testing.assert_allclose(bounds.maximum, [19.025, 27.025, 4.025])
        self.assertTrue(np.all(coordinates >= bounds.minimum))
        self.assertTrue(np.all(coordinates <= bounds.maximum))

    def test_coordinate_metadata_exposes_vertical_offset_warning_and_alignment_state(self):
        base = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0)], vertical_control=False),
            storage_offset=(421000.0, 7433000.0),
        )
        alignment = np.eye(4)
        alignment[:3, 3] = [1.0, 2.0, 3.0]
        contract = base.with_post_crs_alignment(alignment)

        metadata = coordinate_contract_metadata(contract)

        self.assertEqual(metadata["schema"], "odx-coordinate-contract-v1")
        self.assertEqual(metadata["crs_wkt"], contract.output_crs_wkt)
        self.assertEqual(metadata["storage_offset"], [421000.0, 7433000.0, 0.0])
        self.assertEqual(metadata["vertical_reference"], "unreferenced")
        self.assertEqual(metadata["warnings"], list(contract.warnings))
        self.assertEqual(metadata["post_crs_alignment"], list(contract.post_crs_alignment))
        referenced = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0)])
        )
        self.assertEqual(
            coordinate_contract_metadata(referenced)["vertical_reference"],
            "wgs84_ellipsoidal",
        )

    def test_camera_pose_serialization_uses_exact_centres_and_proper_local_rotations(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100000.0, 100000.0, 100.0)]),
        )
        contract = self.translated_contract(contract)
        centres = np.array([[0.0, 0.0, 10.0], [100000.0, 100000.0, 50.0]])
        world_to_camera = np.array([
            np.eye(3),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ])

        poses = transform_camera_poses(centres, world_to_camera, contract)

        expected_centres = contract.transform_points(centres, apply_storage_offset=False)
        np.testing.assert_allclose(poses.centres, expected_centres, atol=0.0001, rtol=0.0)
        for rotation in poses.world_to_camera_rotations:
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)
        self.assertFalse(np.allclose(
            poses.world_to_camera_rotations[0], poses.world_to_camera_rotations[1]
        ))

    @unittest.skipIf(ODMFilterPoints is None, "ODX filter-stage dependencies are not installed")
    def test_filter_stage_exactly_inverts_ordinary_and_forced_output_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            os.makedirs(tree.opensfm)
            os.makedirs(tree.odm_georeferencing)
            with open(tree.path("opensfm", "reference_lla.json"), "w") as reference:
                json.dump({
                    "latitude": self.anchor.latitude,
                    "longitude": self.anchor.longitude,
                    "altitude": self.anchor.ellipsoidal_height,
                }, reference)
            open(tree.path("opensfm", "fresh_reconstruction.marker"), "w").close()
            reconstruction = GeoreferencedReconstructionFixture()
            contract = resolve_coordinate_contract(
                self.anchor,
                self.controls([(0.0, 0.0, 0.0)], vertical_control=False),
                storage_offset=reconstruction.get_proj_offset(),
            )
            args = SimpleNamespace(
                auto_boundary=False, use_exif=False, fast_orthophoto=False,
                orthophoto_resolution=1.0, dsm=False, dtm=False,
                pc_filter=2.5, pc_sample=0.0, max_concurrency=2,
                optimize_disk_space=False, rerun=None, rerun_all=True,
                rerun_from=None,
            )
            fixtures = [
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                np.array([[-100000.0, 80000.0, 0.0], [100000.0, 80000.0, 0.0]]),
            ]
            transformed_boundaries = []
            with mock.patch("stages.odm_filterpoints.point_cloud.filter") as filtering, mock.patch(
                "stages.odm_filterpoints.point_cloud.ply_info",
                return_value={"vertex_count": 1},
            ):
                for topocentric in fixtures:
                    output_boundary = contract.transform_points(
                        topocentric, apply_storage_offset=False
                    )[:, :2].tolist()
                    outputs = {
                        "tree": tree,
                        "reconstruction": reconstruction,
                        "boundary": output_boundary,
                    }
                    ODMFilterPoints("odm_filterpoints", args).process(args, outputs)
                    transformed_boundaries.append(np.asarray(
                        filtering.call_args.kwargs["boundary"]
                    ))

            self.assertEqual(len(transformed_boundaries[0]), 2)
            self.assertGreater(len(transformed_boundaries[1]), 2)
            for transformed, topocentric in zip(transformed_boundaries, fixtures):
                np.testing.assert_allclose(
                    [transformed[0], transformed[-1]], topocentric[:, :2],
                    atol=0.0001, rtol=0.0,
                )
            self.assertEqual(
                outputs["coordinate_contract"].vertical_reference.value,
                "unreferenced",
            )

    @unittest.skipIf(OSFMContext is None, "OpenSfM report dependencies are not installed")
    def test_gcp_and_checkpoint_reports_transform_exact_points_and_keep_semantics(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (1000.0, 500.0, 10.0)]),
        )
        contract = self.translated_contract(contract)
        records = [
            {"id": "gcp-1", "coordinates": [0.0, 0.0, 0.0],
             "observations": [{"shot_id": "a.jpg"}], "error": [0.1, 0.2, 0.3]},
            {"id": "CHK-1", "coordinates": [1000.0, 500.0, 10.0],
             "observations": [{"shot_id": "b.jpg"}], "error": [-0.1, 0.0, 0.2]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            os.makedirs(os.path.join(directory, "stats"))
            stats_records = [
                {
                    **record,
                    "error": dict(zip(("x", "y", "z"), record["error"])),
                }
                for record in records
            ]
            with open(os.path.join(directory, "stats", "stats.json"), "w") as output:
                json.dump({"gcp_errors": {"details": stats_records}}, output)
            exported = OSFMContext(directory).ground_control_points(
                contract.output_crs.to_proj4(), coordinate_contract=contract
            )

        expected = contract.transform_points(
            [record["coordinates"] for record in records], apply_storage_offset=False
        )
        np.testing.assert_allclose(
            [record["coordinates"] for record in exported], expected,
            atol=0.0001, rtol=0.0,
        )
        self.assertEqual([record["id"] for record in exported], ["gcp-1", "CHK-1"])
        self.assertEqual(exported[1]["observations"], records[1]["observations"])

    @unittest.skipIf(
        get_geojson_shots_from_opensfm is None,
        "OpenSfM camera-report dependencies are not installed",
    )
    def test_camera_geojson_preserves_schema_intrinsics_and_decodes_exact_centres(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100000.0, 100000.0, 100.0)]),
        )
        contract = self.translated_contract(contract)
        reconstruction = [{
            "cameras": {
                "camera-1": {"focal": 0.82, "width": 4000, "height": 3000},
            },
            "shots": {
                "a.jpg": {"camera": "camera-1", "rotation": [0.0, 0.0, 0.0],
                          "translation": [0.0, 0.0, -10.0], "capture_time": 12},
                "b.jpg": {"camera": "camera-1", "rotation": [0.0, 0.0, 0.0],
                          "translation": [-100000.0, -100000.0, -50.0], "capture_time": 13},
            },
        }]
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "reconstruction.topocentric.json")
            with open(source, "w") as output:
                json.dump(reconstruction, output)
            report = get_geojson_shots_from_opensfm(
                source, coordinate_contract=contract
            )

        self.assertEqual(report["type"], "FeatureCollection")
        self.assertEqual(report["coordinate_contract"]["crs_wkt"], contract.output_crs_wkt)
        expected_properties = {
            "filename", "camera", "focal", "width", "height", "capture_time",
            "translation", "rotation",
        }
        self.assertEqual(set(report["features"][0]["properties"]), expected_properties)
        self.assertEqual(report["features"][0]["properties"]["focal"], 0.82)
        expected_centres = contract.transform_points(
            [[0.0, 0.0, 10.0], [100000.0, 100000.0, 50.0]],
            apply_storage_offset=False,
        )
        for index, feature in enumerate(report["features"]):
            np.testing.assert_allclose(
                feature["properties"]["translation"], expected_centres[index],
                atol=0.0001, rtol=0.0,
            )
            rotation = get_rotation_matrix(feature["properties"]["rotation"])
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-8)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=8)

    def test_alignment_is_derived_and_ordered_before_storage_offset(self):
        base = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0)]), storage_offset=(1000.0, 2000.0)
        )
        matrix = np.eye(4)
        matrix[:3, 3] = [4.0, 5.0, 6.0]
        aligned = base.with_post_crs_alignment(matrix)
        exact = base.transform_points([[0.0, 0.0, 0.0]], apply_storage_offset=False)

        self.assertIsNone(base.post_crs_alignment)
        np.testing.assert_allclose(
            aligned.transform_points([[0.0, 0.0, 0.0]])[0],
            exact[0] + [4.0 - 1000.0, 5.0 - 2000.0, 6.0],
        )
        singular = np.eye(4)
        singular[:3, :3] = 0.0
        with self.assertRaisesRegex(CoordinateContractError, "invertible"):
            base.with_post_crs_alignment(singular)

    def test_aligned_contract_records_distinct_ordered_operations(self):
        base = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0)])
        )
        matrix = self.alignment_fixture()
        metadata = coordinate_contract_metadata(
            base.with_post_crs_alignment(matrix)
        )

        self.assertEqual(
            [operation["kind"] for operation in metadata["operations"]],
            ["crs", "post_crs_alignment"],
        )
        self.assertEqual(metadata["operations"][0]["definition"], base.operation)
        self.assertEqual(metadata["operations"][1]["matrix"], matrix.flatten().tolist())
        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, "coordinate_contract.json")
            base.with_post_crs_alignment(matrix).persist(manifest)
            with open(manifest) as source:
                persisted = json.load(source)
            self.assertEqual(persisted["operations"], metadata["operations"])
            self.assertEqual(
                load_coordinate_contract(manifest).post_crs_alignment,
                tuple(matrix.flat),
            )
            persisted["operations"].reverse()
            with open(manifest, "w") as output:
                json.dump(persisted, output)
            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(manifest)

    def test_alignment_positions_normals_cameras_and_boundary_share_one_order(self):
        base = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100.0, 50.0, 10.0)]),
        )
        matrix = self.alignment_fixture()
        aligned = base.with_post_crs_alignment(matrix)
        source = np.asarray([[10.0, 20.0, 2.0], [30.0, -5.0, 7.0]])
        base_positions = base.transform_points(source, apply_storage_offset=False)
        expected_positions = (
            np.column_stack((base_positions, np.ones(len(base_positions)))) @ matrix.T
        )[:, :3]

        point_positions = aligned.transform_points(source, apply_storage_offset=False)
        cloud = np.zeros(2, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
        cloud["X"], cloud["Y"], cloud["Z"] = source.T
        decoded_cloud = transform_point_cloud_batch(aligned, cloud)
        cloud_positions = np.column_stack((
            decoded_cloud["X"], decoded_cloud["Y"], decoded_cloud["Z"],
        ))
        poses = transform_camera_poses(source, np.repeat(np.eye(3)[None], 2, axis=0), aligned)
        np.testing.assert_allclose(point_positions, expected_positions, atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(cloud_positions, expected_positions, atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(poses.centres, expected_positions, atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(aligned.inverse_points(expected_positions), source, atol=1e-7)
        selected_canonical_boundary = base.inverse_points(base_positions)
        published_aligned_boundary = aligned.transform_points(
            selected_canonical_boundary, apply_storage_offset=False
        )
        np.testing.assert_allclose(
            published_aligned_boundary, expected_positions, atol=1e-7, rtol=0.0,
        )

        source_normals = np.asarray([[0.25, -0.5, 1.0], [1.0, 0.5, -0.25]])
        base_normals = base.transform_normals(source, source_normals)
        expected_normals = (np.linalg.inv(matrix[:3, :3]).T @ base_normals.T).T
        expected_normals /= np.linalg.norm(expected_normals, axis=1)[:, None]
        np.testing.assert_allclose(
            aligned.transform_normals(source, source_normals),
            expected_normals, atol=1e-6, rtol=0.0,
        )

        boundary = np.vstack((expected_positions, expected_positions[0]))
        recovered = transform_output_boundary(
            boundary, aligned, finest_resolution=1.0, serialization_bound=0.01,
        )
        np.testing.assert_allclose(recovered[[0, -1]], source[[0, 0]], atol=1e-7)

        with tempfile.TemporaryDirectory() as directory:
            mesh_source = os.path.join(directory, "topocentric.obj")
            mesh_output = os.path.join(directory, "public.obj")
            with open(mesh_source, "w") as mesh:
                mesh.write(
                    "v {} {} {}\n"
                    "v {} {} {}\n"
                    "v 0 0 0\n"
                    "f 1 2 3\n".format(*source[0], *source[1])
                )
            export_georeferenced_mesh(
                mesh_source, mesh_output, aligned, apply_storage_offset=False
            )
            with open(mesh_output) as mesh:
                mesh_positions = np.asarray([
                    [float(value) for value in line.split()[1:4]]
                    for line in mesh if line.startswith("v ")
                ])
            np.testing.assert_allclose(
                mesh_positions[:2], expected_positions, atol=1e-7, rtol=0.0,
            )

    def test_mesh_export_preserves_visual_structure_and_transforms_vertices_and_normals(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100000.0, 20000.0, 100.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        source_vertices = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 20.0, 1.0],
        ])
        source_normal = np.cross(
            source_vertices[1] - source_vertices[0],
            source_vertices[2] - source_vertices[0],
        )
        source_normal /= np.linalg.norm(source_normal)
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "odm_textured_model_geo_topocentric.obj")
            output = os.path.join(directory, "odm_textured_model_geo.obj")
            material_path = os.path.join(directory, "odm_textured_model_geo.mtl")
            texture_path = os.path.join(directory, "texture.jpg")
            with open(material_path, "wb") as material:
                material.write(b"newmtl atlas\nmap_Kd texture.jpg\n")
            with open(texture_path, "wb") as texture:
                texture.write(b"unchanged-texture-bytes")
            with open(source, "w") as mesh:
                mesh.write(
                    "mtllib odm_textured_model_geo.mtl\n"
                    "o textured-model\n"
                    "v 0 0 0\n"
                    "v 100 0 0\n"
                    "v 0 20 1\n"
                    "vt 0 0\nvt 1 0\nvt 0 1\n"
                    "vn {} {} {}\n"
                    "usemtl atlas\n"
                    "f 1/1/1 2/2/1 3/3/1\n"
                    "f 3/3/1 2/2/1 1/1/1\n".format(*source_normal)
                )

            result = export_georeferenced_mesh(source, output, contract)
            with open(output) as mesh:
                lines = mesh.readlines()

            vertices = np.array([
                list(map(float, line.split()[1:4]))
                for line in lines if line.startswith("v ")
            ])
            normals = np.array([
                list(map(float, line.split()[1:4]))
                for line in lines if line.startswith("vn ")
            ])
            expected = contract.transform_points(source_vertices)
            absolute = vertices + np.array([421000.0, 7433000.0, 0.0])
            expected_normals = contract.transform_normals(
                source_vertices, np.repeat(source_normal[np.newaxis, :], 3, axis=0)
            )

            self.assertEqual(result.vertex_count, 3)
            self.assertEqual(result.face_count, 2)
            np.testing.assert_allclose(absolute, contract.transform_points(
                source_vertices, apply_storage_offset=False
            ), atol=0.0001, rtol=0.0)
            self.assertTrue(np.allclose(vertices, expected, atol=0.0001, rtol=0.0))
            np.testing.assert_allclose(normals, expected_normals, atol=1e-10, rtol=0.0)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
            self.assertIn("mtllib odm_textured_model_geo.mtl\n", lines)
            self.assertIn("vt 1 0\n", lines)
            self.assertIn("usemtl atlas\n", lines)
            self.assertEqual([line for line in lines if line.startswith("f ")],
                             [
                                 "f 1/1/1 2/2/2 3/3/3\n",
                                 "f 3/3/3 2/2/2 1/1/1\n",
                             ])
            with open(material_path, "rb") as material:
                self.assertEqual(material.read(), b"newmtl atlas\nmap_Kd texture.jpg\n")
            with open(texture_path, "rb") as texture:
                self.assertEqual(texture.read(), b"unchanged-texture-bytes")
            with open(source) as mesh:
                self.assertIn("v 100 0 0\n", mesh.readlines())

    def test_absolute_mesh_export_does_not_subtract_the_storage_offset(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.obj")
            output = os.path.join(directory, "absolute.obj")
            with open(source, "w") as mesh:
                mesh.write("v 0 0 0\n")

            export_georeferenced_mesh(
                source, output, contract, apply_storage_offset=False
            )
            with open(output) as mesh:
                actual = np.array(list(map(float, mesh.readline().split()[1:4])))

            np.testing.assert_allclose(
                actual,
                contract.transform_points([[0.0, 0.0, 0.0]], apply_storage_offset=False)[0],
                atol=0.0001,
                rtol=0.0,
            )

    @unittest.skipIf(obj2glb is None, "ODX GLB dependencies are not installed")
    def test_glb_decodes_the_georeferenced_mesh_with_float32_and_rtc_conventions(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100.0, 20.0, 1.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        contract = self.translated_contract(contract)
        source_vertices = np.array([
            [0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 20.0, 1.0]
        ])
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_textured_triangle(directory, "topocentric.obj")
            public = os.path.join(directory, "odm_textured_model_geo.obj")
            glb_path = os.path.join(directory, "odm_textured_model_geo.glb")
            export_georeferenced_mesh(source, public, contract)
            obj2glb(public, glb_path, rtc=contract.storage_offset, draco_compression=False)

            gltf, decoded = self.decode_gltf_positions(glb_path)
            expected = contract.transform_points(source_vertices)
            tolerance = max(float(np.max(np.spacing(np.abs(expected.astype(np.float32))))), 0.0001)

            np.testing.assert_allclose(decoded, expected, atol=tolerance, rtol=0.0)
            self.assertEqual(
                gltf.extensions["CESIUM_RTC"]["center"],
                [contract.storage_offset[0], contract.storage_offset[1], 0.0],
            )

    @unittest.skipIf(
        build_3dtiles is None or shutil.which("Obj2Tiles") is None,
        "ODX textured-tile dependencies are not installed",
    )
    def test_textured_tiles_decode_the_georeferenced_mesh_through_the_published_transform(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100.0, 20.0, 1.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        contract = self.translated_contract(contract)
        with tempfile.TemporaryDirectory() as directory:
            source = self.write_textured_triangle(directory, "topocentric.obj")
            public = os.path.join(directory, "odm_textured_model_geo.obj")
            export_georeferenced_mesh(source, public, contract)
            with open(public) as public_mesh:
                expected_local = np.array([
                    list(map(float, line.split()[1:4]))
                    for line in public_mesh if line.startswith("v ")
                ], dtype=np.float64)

            output = os.path.join(directory, "tiles")
            reference = os.path.join(directory, "reference_lla.json")
            with open(reference, "w") as reference_file:
                json.dump({
                    "latitude": contract.anchor.latitude,
                    "longitude": contract.anchor.longitude,
                    "altitude": contract.anchor.ellipsoidal_height,
                }, reference_file)
            from opendm.ogctiles import build_textured_model
            build_textured_model(public, output, reference, rerun=True)

            with open(os.path.join(output, "tileset.json")) as tileset_file:
                tileset = json.load(tileset_file)
            transform = np.asarray(tileset["root"]["transform"], dtype=np.float64).reshape(
                (4, 4), order="F"
            )
            b3dm_path = next(
                os.path.join(root, name)
                for root, _, names in os.walk(output)
                for name in names if name.endswith(".b3dm")
            )
            with open(b3dm_path, "rb") as tile:
                header = tile.read(28)
                (
                    magic, _, _, feature_json_byte_length,
                    feature_binary_byte_length, batch_json_byte_length,
                    batch_binary_byte_length,
                ) = struct.unpack("<4s6I", header)
                self.assertEqual(magic, b"b3dm")
                tile.seek(
                    28 + feature_json_byte_length + feature_binary_byte_length
                    + batch_json_byte_length + batch_binary_byte_length
                )
                glb_data = tile.read()
            glb_path = os.path.join(directory, "decoded-tile.glb")
            with open(glb_path, "wb") as glb_file:
                glb_file.write(glb_data)
            _, decoded_local = self.decode_gltf_positions(glb_path)
            expected_float = expected_local.astype(np.float32)
            decoded_order = np.lexsort(decoded_local.T[::-1])
            expected_order = np.lexsort(expected_float.T[::-1])
            np.testing.assert_allclose(
                decoded_local[decoded_order], expected_float[expected_order],
                atol=max(float(np.max(np.spacing(np.abs(expected_float)))), 0.0001),
                rtol=0.0,
            )
            decoded_absolute = (
                transform @ np.column_stack((decoded_local, np.ones(len(decoded_local)))).T
            ).T[:, :3]
            expected_absolute = (
                transform @ np.column_stack((expected_float, np.ones(len(expected_float)))).T
            ).T[:, :3]
            decoded_absolute_order = np.lexsort(decoded_absolute.T[::-1])
            expected_absolute_order = np.lexsort(expected_absolute.T[::-1])
            np.testing.assert_allclose(
                decoded_absolute[decoded_absolute_order],
                expected_absolute[expected_absolute_order],
                atol=0.001,
                rtol=0.0,
            )

    def test_mesh_export_rejects_invalid_faces_without_publishing(self):
        contract = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0)])
        )
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.obj")
            output = os.path.join(directory, "output.obj")
            with open(source, "w") as mesh:
                mesh.write("v 0 0 0\nvn 0 0 1\nf 2//1 1//1 1//1\n")

            with self.assertRaisesRegex(MeshExportError, "vertex index"):
                export_georeferenced_mesh(source, output, contract)
            self.assertFalse(os.path.exists(output))

    @unittest.skipIf(ODMGeoreferencingStage is None, "ODX stage dependencies are not installed")
    def test_stage_exports_mesh_only_from_the_canonical_topocentric_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            args, tree, outputs = self.stage_fixture(directory)
            os.makedirs(tree.odm_texturing)
            os.makedirs(tree.odm_25dtexturing)
            open(tree.filtered_point_cloud, "w").close()
            public_obj = os.path.join(tree.odm_texturing, tree.odm_textured_model_obj)
            topocentric_obj = os.path.splitext(public_obj)[0] + "_topocentric.obj"
            with open(topocentric_obj, "w") as mesh:
                mesh.write("v 0 0 0\nv 100 0 0\nv 0 100 0\nf 1 2 3\n")

            stage = ODMGeoreferencingStage("odm_georeferencing", args)
            with mock.patch(
                "stages.odm_georeferencing.export_georeferenced_point_cloud",
                side_effect=self.point_cloud_export_stub(3),
            ), mock.patch("stages.odm_georeferencing.point_cloud.post_point_cloud_steps"):
                stage.process(args, outputs)

            with open(public_obj) as mesh:
                vertices = np.array([
                    list(map(float, line.split()[1:4]))
                    for line in mesh if line.startswith("v ")
                ])
            expected = outputs["coordinate_contract"].transform_points(
                [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]
            )
            np.testing.assert_allclose(vertices, expected, atol=0.0001, rtol=0.0)
            with open(topocentric_obj) as mesh:
                self.assertEqual(mesh.readline(), "v 0 0 0\n")

    @unittest.skipIf(ODMGeoreferencingStage is None, "ODX stage dependencies are not installed")
    def test_optional_mesh_failure_warns_without_publishing_an_approximation(self):
        with tempfile.TemporaryDirectory() as directory:
            args, tree, outputs = self.stage_fixture(directory)
            os.makedirs(tree.odm_texturing)
            os.makedirs(tree.odm_25dtexturing)
            open(tree.filtered_point_cloud, "w").close()
            public_obj = os.path.join(tree.odm_texturing, tree.odm_textured_model_obj)
            with open(public_obj, "w") as mesh:
                mesh.write("v 0 0 0\n")

            stage = ODMGeoreferencingStage("odm_georeferencing", args)
            with mock.patch(
                "stages.odm_georeferencing.export_georeferenced_point_cloud",
                side_effect=self.point_cloud_export_stub(1),
            ), mock.patch(
                "stages.odm_georeferencing.point_cloud.post_point_cloud_steps"
            ), mock.patch("stages.odm_georeferencing.log.WARNING") as warning:
                stage.process(args, outputs)

            self.assertFalse(os.path.exists(public_obj))
            self.assertIn("rerun from mvs_texturing", warning.call_args.args[0])

    def test_crossing_output_area_warns_but_wholly_outside_fails(self):
        crossing = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0), (400000.0, 0.0, 0.0)])
        )
        self.assertIn("crosses", crossing.warnings[0])

        with self.assertRaisesRegex(CoordinateContractError, "wholly outside"):
            resolve_coordinate_contract(self.anchor, self.controls([(700000.0, 0.0, 0.0)]))

    def test_manifest_round_trip_preserves_the_operation(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0)], vertical_control=False),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "coordinate_contract.json")
            contract.persist(path)
            loaded = load_coordinate_contract(path)

            self.assertEqual(loaded.operation, contract.operation)
            self.assertEqual(loaded.vertical_reference, "unreferenced")
            with open(path) as manifest:
                self.assertEqual(json.load(manifest)["schema"], "odx-coordinate-contract-v1")

    def test_vertically_unreferenced_contract_preserves_relative_z(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(1000.0, 2000.0, 25.0)], vertical_control=False),
        )
        transformed = contract.transform_points(
            [(1000.0, 2000.0, 25.0)], apply_storage_offset=False
        )

        self.assertEqual(transformed[0, 2], 25.0)
        np.testing.assert_allclose(
            contract.inverse_points(transformed),
            [(1000.0, 2000.0, 25.0)],
            atol=0.0001,
            rtol=0.0,
        )

    def test_missing_or_incompatible_manifest_requires_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "coordinate_contract.json")
            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)

            contract = resolve_coordinate_contract(self.anchor, self.controls([(0.0, 0.0, 0.0)]))
            contract.persist(path)
            with open(path) as manifest:
                payload = json.load(manifest)
            payload["axis_order"] = ["northing", "easting", "height"]
            with open(path, "w") as manifest:
                json.dump(payload, manifest)
            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)
            with open(path, "w") as manifest:
                json.dump({"schema": "old"}, manifest)
            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)

    def test_stage_creates_then_reloads_the_same_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = os.path.join(directory, "coordinate_contract.json")
            reference_path = os.path.join(directory, "reference_lla.json")
            with open(reference_path, "w") as reference:
                json.dump(
                    {"latitude": -23.206694678, "longitude": -45.764977891, "altitude": 635.5485},
                    reference,
                )
            gps_provenance = ControlProvenance(
                source=ControlSource.GPS,
                input_crs_wkt=CRS.from_epsg(4979).to_wkt(),
                operation="test operation",
                grids=(),
                accuracy=0.0,
                axis_order=("easting", "northing", "height"),
                height_assumptions=("GPS altitude is WGS 84 ellipsoidal height in metres",),
                warnings=(),
                versions=(("proj", "test"),),
            )
            controls = self.controls(
                [(0.0, 0.0, 0.0)], provenance=(gps_provenance,)
            )
            created = resolve_stage_coordinate_contract(
                manifest_path,
                reference_path,
                (421722.0, 7433393.0),
                controls=controls,
                fresh_reconstruction=True,
            )
            reloaded = resolve_stage_coordinate_contract(
                manifest_path,
                reference_path,
                (421722.0, 7433393.0),
                controls=controls,
            )

            self.assertEqual(reloaded, created)
            self.assertEqual(reloaded.control_provenance[0].source, ControlSource.GPS)
            with self.assertRaisesRegex(ManifestError, "does not match reconstructed"):
                resolve_stage_coordinate_contract(
                    manifest_path,
                    reference_path,
                    (0.0, 0.0),
                    controls=controls,
                )

    def test_submodel_stage_creates_and_deterministically_reloads_from_parent_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = os.path.join(directory, "coordinate_contract.json")
            reference_path = os.path.join(directory, "reference_lla.json")
            submodel_anchor = TopocentricAnchor(-23.205, -45.762, 640.0)
            with open(reference_path, "w") as reference:
                json.dump({
                    "latitude": submodel_anchor.latitude,
                    "longitude": submodel_anchor.longitude,
                    "altitude": submodel_anchor.ellipsoidal_height,
                }, reference)
            controls = self.controls([(0.0, 0.0, 0.0)], vertical_control=False)
            selection = resolve_output_selection(
                self.anchor, controls, storage_offset=(421000.0, 7433000.0)
            )

            created = resolve_stage_coordinate_contract(
                manifest_path, reference_path, (1.0, 2.0), controls=controls,
                output_selection=selection, fresh_reconstruction=True,
            )
            reloaded = resolve_stage_coordinate_contract(
                manifest_path, reference_path, (9.0, 8.0), controls=controls,
                output_selection=selection,
            )

            self.assertEqual(created, reloaded)
            self.assertEqual(created.anchor, submodel_anchor)
            self.assertEqual(created.storage_offset, selection.storage_offset)
            self.assertEqual(created.output_crs_wkt, selection.output_crs_wkt)

    def test_submodel_missing_contract_requires_fresh_reconstruction_even_with_parent_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = os.path.join(directory, "reference_lla.json")
            with open(reference_path, "w") as reference:
                json.dump({
                    "latitude": self.anchor.latitude,
                    "longitude": self.anchor.longitude,
                    "altitude": self.anchor.ellipsoidal_height,
                }, reference)
            controls = self.controls([(0.0, 0.0, 0.0)])
            selection = resolve_output_selection(self.anchor, controls)

            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                resolve_stage_coordinate_contract(
                    os.path.join(directory, "coordinate_contract.json"),
                    reference_path, (0.0, 0.0), controls=controls,
                    output_selection=selection,
                )

    def test_stage_rerun_without_manifest_requires_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                resolve_stage_coordinate_contract(
                    os.path.join(directory, "coordinate_contract.json"),
                    os.path.join(directory, "reference_lla.json"),
                    (0.0, 0.0),
                    controls=self.controls(
                        [(0.0, 0.0, 0.0)], vertical_control=False
                    ),
                )

    def test_stage_reloads_the_persisted_post_crs_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = os.path.join(directory, "coordinate_contract.json")
            reference_path = os.path.join(directory, "reference_lla.json")
            with open(reference_path, "w") as reference:
                json.dump(
                    {"latitude": -23.206694678, "longitude": -45.764977891, "altitude": 635.5485},
                    reference,
                )
            controls = self.controls([(0.0, 0.0, 0.0)])
            base = resolve_stage_coordinate_contract(
                manifest_path, reference_path, (421000.0, 7433000.0),
                controls=controls, fresh_reconstruction=True,
            )
            matrix = np.eye(4)
            matrix[:3, 3] = [0.25, -0.5, 1.0]
            aligned = base.with_post_crs_alignment(matrix)
            aligned.persist(manifest_path)

            reloaded = resolve_stage_coordinate_contract(
                manifest_path, reference_path, (421000.0, 7433000.0),
                controls=controls,
            )

            self.assertEqual(reloaded, aligned)

    def test_fresh_reconstruction_replaces_an_incompatible_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = os.path.join(directory, "coordinate_contract.json")
            reference_path = os.path.join(directory, "reference_lla.json")
            with open(manifest_path, "w") as manifest:
                json.dump({"schema": "old"}, manifest)
            with open(reference_path, "w") as reference:
                json.dump(
                    {"latitude": -23.206694678, "longitude": -45.764977891, "altitude": 635.5485},
                    reference,
                )

            recovered = resolve_stage_coordinate_contract(
                manifest_path,
                reference_path,
                (0.0, 0.0),
                controls=self.controls(
                    [(0.0, 0.0, 0.0)], vertical_control=False
                ),
                fresh_reconstruction=True,
            )

            self.assertEqual(load_coordinate_contract(manifest_path), recovered)
    def test_control_normalization_honors_input_crs_without_selecting_output(self):
        anchor = TopocentricAnchor(30.0, -97.0, 635.5485)
        input_crs = CRS.from_epsg(2277)
        source = Transformer.from_crs(CRS.from_epsg(4979), input_crs, always_xy=True)
        x, y, _ = source.transform(-97.0, 30.0, 635.5485)
        z = 635.5485 / input_crs.axis_info[0].unit_conversion_factor

        controls = normalize_control_observations(
            [[x, y, z]], input_crs, anchor, declared_vertical=False
        )
        contract = resolve_coordinate_contract(anchor, controls)

        np.testing.assert_allclose(controls.coordinates[0], [0.0, 0.0, 0.0], atol=0.0001)
        self.assertEqual(contract.output_crs.to_epsg(), 32614)
        self.assertIn("undeclared GCP height", controls.provenance[0].height_assumptions[0])

    def test_invalid_coordinates_and_dynamic_crs_are_rejected_with_context(self):
        with self.assertRaisesRegex(CoordinateContractError, "coordinate sample"):
            resolve_coordinate_contract(
                self.anchor, self.controls([(float("nan"), 0.0, 0.0)])
            )
        with self.assertRaisesRegex(CoordinateContractError, "dynamic"):
            normalize_control_observations(
                [[0.0, 0.0, 0.0]], CRS.from_epsg(7912), self.anchor, declared_vertical=True
            )
        contract = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0)])
        )
        with self.assertRaisesRegex(CoordinateContractError, "camera frame transformation"):
            contract.transform_camera_frames(
                [(0.0, 0.0, 0.0)], [np.full((3, 3), np.nan)]
            )

    def test_point_cloud_batch_transforms_positions_normals_and_preserves_dimensions(self):
        contract = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0), (100000.0, 100000.0, 100.0)])
        )
        source = np.array(
            [
                (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 27, 4),
                (100000.0, 100000.0, 100.0, 1.0, 0.0, 0.0, 91, 2),
            ],
            dtype=[
                ("X", "f8"), ("Y", "f8"), ("Z", "f8"),
                ("NormalX", "f4"), ("NormalY", "f4"), ("NormalZ", "f4"),
                ("Intensity", "u2"), ("Classification", "u1"),
            ],
        )

        transformed = transform_point_cloud_batch(contract, source)

        expected = contract.transform_points(
            np.column_stack((source["X"], source["Y"], source["Z"])),
            apply_storage_offset=False,
        )
        np.testing.assert_allclose(
            np.column_stack((transformed["X"], transformed["Y"], transformed["Z"])),
            expected,
            atol=0.0001,
            rtol=0.0,
        )
        np.testing.assert_array_equal(transformed["Intensity"], source["Intensity"])
        np.testing.assert_array_equal(transformed["Classification"], source["Classification"])
        normals = np.column_stack(
            (transformed["NormalX"], transformed["NormalY"], transformed["NormalZ"])
        )
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
        self.assertEqual(transformed.dtype["NormalX"], source.dtype["NormalX"])
        self.assertEqual(len(transformed), len(source))

    def test_public_point_cloud_batch_keeps_stock_fields_and_omits_internal_fields(self):
        source = np.array(
            [(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 7, 4, 9, 2)],
            dtype=[
                ("X", "f8"), ("Y", "f8"), ("Z", "f8"),
                ("NormalX", "f4"), ("NormalY", "f4"), ("NormalZ", "f4"),
                ("Intensity", "u2"), ("views", "u1"),
                ("UserData", "u1"), ("Classification", "u1"),
            ],
        )

        public = public_point_cloud_batch(source)

        self.assertEqual(
            public.dtype.names,
            ("X", "Y", "Z", "Intensity", "Classification", "UserData"),
        )
        np.testing.assert_array_equal(public["UserData"], [9])
        self.assertEqual(len(public), len(source))

    def test_las_encoding_uses_spacing_cap_absolute_offset_and_signed_int32(self):
        bounds = np.array(
            [[421722.1234, 7433393.4567, 635.1234], [421822.9876, 7433494.0123, 700.9876]]
        )

        encoding = select_las_encoding(bounds, spacing=0.024)

        self.assertEqual(encoding.scale, (0.001, 0.001, 0.001))
        self.assertNotEqual(encoding.offset[:2], (1000.0, 2000.0))
        encoded = np.rint((bounds - encoding.offset) / encoding.scale)
        self.assertGreaterEqual(encoded.min(), np.iinfo(np.int32).min)
        self.assertLessEqual(encoded.max(), np.iinfo(np.int32).max)
        decoded = encoded * encoding.scale + encoding.offset
        np.testing.assert_array_less(np.abs(decoded - bounds), 0.0005000001)

    def test_point_cloud_batch_rejects_non_finite_points_and_count_mismatches(self):
        contract = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0)])
        )
        invalid = np.array(
            [(float("nan"), 0.0, 0.0)], dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")]
        )
        with self.assertRaisesRegex(CoordinateContractError, "georeferenced point cloud"):
            transform_point_cloud_batch(contract, invalid)

        with self.assertRaisesRegex(CoordinateContractError, "point count changed"):
            transform_point_cloud_batch(
                contract,
                np.array([(0.0, 0.0, 0.0)], dtype=invalid.dtype),
                expected_count=2,
            )

    def test_point_cloud_export_streams_twice_and_writes_absolute_laz_metadata(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (100.0, 50.0, 10.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        dtype = [("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Intensity", "u2")]
        batches = [
            np.array([(0.0, 0.0, 0.0, 7)], dtype=dtype),
            np.array([(100.0, 50.0, 10.0, 9)], dtype=dtype),
        ]

        class FakePipeline:
            source_passes = 0
            writer_spec = None
            written = []

            def __init__(self, spec, arrays=(), stream_handlers=()):
                self.spec = json.loads(spec)
                self.arrays = arrays
                self.handlers = stream_handlers
                self.streamable = True

            def iterator(self, chunk_size):
                self.__class__.source_passes += 1
                return iter([batch.copy() for batch in batches])

            def execute(self):
                self.__class__.writer_spec = self.spec
                buffer = self.arrays[0]
                handler = self.handlers[0]
                count = 0
                while True:
                    size = handler()
                    if size == 0:
                        break
                    self.__class__.written.append(buffer[:size].copy())
                    count += size
                open(self.spec[0]["filename"], "wb").close()
                return count


            def execute_streaming(self, chunk_size):
                return self.execute()

        class FakePdal:
            Pipeline = FakePipeline

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "odm_georeferenced_model.laz")
            result = export_georeferenced_point_cloud(
                "canonical.ply", output, contract, spacing=0.024,
                chunk_size=1, pdal_module=FakePdal,
            )

            self.assertEqual(result.point_count, 2)
            self.assertIsNotNone(result.bounds)
            self.assertEqual(FakePipeline.source_passes, 2)
            writer = FakePipeline.writer_spec[0]
            self.assertEqual(writer["a_srs"], contract.output_crs_wkt)
            self.assertEqual(writer["minor_version"], 4)
            self.assertEqual(writer["dataformat_id"], 3)
            self.assertTrue(writer["enhanced_srs_vlrs"])
            self.assertNotIn("extra_dims", writer)
            self.assertEqual(writer["scale_x"], 0.001)
            self.assertNotEqual(writer["offset_x"], contract.storage_offset[0])
            written = np.concatenate(FakePipeline.written)
            expected = contract.transform_points(
                [[0.0, 0.0, 0.0], [100.0, 50.0, 10.0]],
                apply_storage_offset=False,
            )
            np.testing.assert_allclose(
                np.column_stack((written["X"], written["Y"], written["Z"])), expected,
                atol=0.0001, rtol=0.0,
            )
            written_coordinates = np.column_stack(
                (written["X"], written["Y"], written["Z"])
            )
            self.assertTrue(np.all(written_coordinates >= result.bounds.minimum))
            self.assertTrue(np.all(written_coordinates <= result.bounds.maximum))
            np.testing.assert_allclose(
                result.bounds.minimum,
                np.min(written_coordinates, axis=0) - 0.0005,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                result.bounds.maximum,
                np.max(written_coordinates, axis=0) + 0.0005,
                atol=1e-10,
            )
            np.testing.assert_array_equal(written["Intensity"], [7, 9])

            FakePipeline.written = []
            rerun = export_georeferenced_point_cloud(
                "canonical.ply", output, contract, spacing=0.024,
                chunk_size=1, pdal_module=FakePdal,
            )
            self.assertEqual(FakePipeline.source_passes, 4)
            self.assertEqual(rerun.encoding, result.encoding)
            np.testing.assert_array_equal(np.concatenate(FakePipeline.written), written)

    @unittest.skipIf(pdal is None, "PDAL bindings are not installed")
    def test_point_cloud_export_decodes_with_crs_scale_dimensions_and_exact_positions(self):
        contract = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0), (100.0, 50.0, 10.0)])
        )
        source_points = np.array([[0.0, 0.0, 0.0], [100.0, 50.0, 10.0]])
        expected = contract.transform_points(source_points, apply_storage_offset=False)
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "canonical.ply")
            output = os.path.join(directory, "odm_georeferenced_model.laz")
            with open(source, "w") as point_cloud:
                point_cloud.write(
                    "ply\nformat ascii 1.0\nelement vertex 2\n"
                    "property double x\nproperty double y\nproperty double z\n"
                    "property float nx\nproperty float ny\nproperty float nz\n"
                    "property ushort intensity\nproperty uchar views\nend_header\n"
                    "0 0 0 0 0 1 7 4\n100 50 10 1 0 0 9 2\n"
                )

            result = export_georeferenced_point_cloud(
                source, output, contract, spacing=0.024, chunk_size=1,
            )
            decoded_pipeline = pdal.Pipeline(json.dumps([output]))
            decoded_count = decoded_pipeline.execute()
            decoded = decoded_pipeline.arrays[0]
            quickinfo = pdal.Pipeline(json.dumps([output])).quickinfo["readers.las"]
            metadata = quickinfo["metadata"]

            self.assertEqual(decoded_count, result.point_count)
            self.assertEqual(decoded_count, 2)
            self.assertIn("Intensity", decoded.dtype.names)
            self.assertIn("UserData", decoded.dtype.names)
            self.assertNotIn("NormalX", decoded.dtype.names)
            self.assertNotIn("NormalY", decoded.dtype.names)
            self.assertNotIn("NormalZ", decoded.dtype.names)
            self.assertNotIn("views", decoded.dtype.names)
            self.assertEqual(set(decoded.dtype.names), set(PUBLIC_LAZ_DIMENSIONS))
            self.assertTrue(quickinfo["srs"]["wkt"])
            self.assertEqual(int(metadata["minor_version"]), 4)
            self.assertEqual(int(metadata["dataformat_id"]), 3)
            self.assertEqual(int(metadata["point_length"]), 34)
            self.assertEqual(float(metadata["scale_x"]), 0.001)
            self.assertEqual(float(metadata["scale_y"]), 0.001)
            self.assertEqual(float(metadata["scale_z"]), 0.001)
            for axis in "xyz":
                self.assertAlmostEqual(
                    float(metadata["offset_" + axis]),
                    result.encoding.offset["xyz".index(axis)],
                    delta=1e-8,
                )
            decoded_coordinates = np.column_stack(
                (decoded["X"], decoded["Y"], decoded["Z"])
            )
            np.testing.assert_allclose(
                decoded_coordinates, expected, atol=0.0005 + 0.0001, rtol=0.0
            )
            self.assertTrue(np.all(decoded_coordinates >= result.bounds.minimum))
            self.assertTrue(np.all(decoded_coordinates <= result.bounds.maximum))
            np.testing.assert_array_equal(decoded["Intensity"], [7, 9])
            np.testing.assert_array_equal(decoded["UserData"], [4, 2])

            las_output = os.path.join(directory, "odm_georeferenced_model.las")
            xyz_output = os.path.join(directory, "odm_georeferenced_model.csv")
            pdal.Pipeline(json.dumps([
                output,
                {"type": "writers.las", "filename": las_output, "forward": "all"},
            ])).execute_streaming(chunk_size=1)
            pdal.Pipeline(json.dumps([
                output,
                {
                    "type": "writers.text", "filename": xyz_output,
                    "format": "csv", "order": "X,Y,Z", "keep_unspecified": False,
                },
            ])).execute_streaming(chunk_size=1)
            las_pipeline = pdal.Pipeline(json.dumps([las_output]))
            las_pipeline.execute()
            las_coordinates = np.column_stack(
                (las_pipeline.arrays[0]["X"], las_pipeline.arrays[0]["Y"], las_pipeline.arrays[0]["Z"])
            )
            xyz_coordinates = np.loadtxt(xyz_output, delimiter=",", skiprows=1)
            np.testing.assert_allclose(las_coordinates, decoded_coordinates, atol=0.0005, rtol=0.0)
            np.testing.assert_allclose(
                xyz_coordinates,
                decoded_coordinates,
                atol=max(result.encoding.scale) * 0.5,
                rtol=0.0,
            )

    @unittest.skipIf(
        pdal is None or point_cloud_module is None or shutil.which("lasmerge") is None,
        "PDAL/lasmerge ODX integration dependencies are not installed",
    )
    def test_distinct_anchor_point_cloud_exports_merge_into_one_decoded_artifact(self):
        anchors = (
            TopocentricAnchor(-23.2067, -45.7650, 635.0),
            TopocentricAnchor(-23.2050, -45.7620, 642.0),
        )
        controls = self.controls([(0.0, 0.0, 0.0)])
        selection = resolve_output_selection(anchors[0], controls)
        contracts = [
            resolve_coordinate_contract(anchor, controls, output_selection=selection)
            for anchor in anchors
        ]
        geographic_points = (
            (-45.7635, -23.2058, 650.0),
            (-45.7635, -23.2058, 650.0),
        )
        expected = np.asarray(
            Transformer.from_crs(
                CRS.from_epsg(4979), selection.output_crs, always_xy=True
            ).transform(*np.asarray(geographic_points).T)
        ).T
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for index, (anchor, contract, geographic) in enumerate(zip(
                anchors, contracts, geographic_points
            )):
                source = os.path.join(directory, "submodel-{}.ply".format(index))
                output = os.path.join(directory, "submodel-{}.laz".format(index))
                to_topocentric = Transformer.from_pipeline(
                    "+proj=pipeline +step +proj=cart +ellps=WGS84 "
                    "+step +proj=topocentric +ellps=WGS84 "
                    "+lat_0={0:.17g} +lon_0={1:.17g} +h_0={2:.17g}".format(
                        anchor.latitude, anchor.longitude, anchor.ellipsoidal_height
                    )
                )
                local = to_topocentric.transform(*geographic)
                with open(source, "w") as point_cloud:
                    point_cloud.write(
                        "ply\nformat ascii 1.0\nelement vertex 1\n"
                        "property double x\nproperty double y\nproperty double z\n"
                        "end_header\n{:.17g} {:.17g} {:.17g}\n".format(*local)
                    )
                export_georeferenced_point_cloud(source, output, contract, spacing=0.01)
                outputs.append(output)
            merged = os.path.join(directory, "merged.laz")

            point_cloud_module.merge(outputs, merged, rerun=True)
            pipeline = pdal.Pipeline(json.dumps([merged]))
            self.assertEqual(pipeline.execute(), 2)
            decoded = np.column_stack((
                pipeline.arrays[0]["X"], pipeline.arrays[0]["Y"],
                pipeline.arrays[0]["Z"],
            ))

        np.testing.assert_allclose(
            decoded[np.argsort(decoded[:, 0])], expected[np.argsort(expected[:, 0])],
            atol=0.0006, rtol=0.0,
        )
        np.testing.assert_allclose(decoded[0], decoded[1], atol=0.0006, rtol=0.0)

    @unittest.skipIf(ODMGeoreferencingStage is None, "ODX stage dependencies are not installed")
    def test_stage_exports_canonical_points_deterministically_and_propagates_required_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            args, tree, outputs = self.stage_fixture(directory, crop=1)
            open(tree.filtered_point_cloud, "w").close()
            calls = []

            def export(source, output, contract, **kwargs):
                calls.append((source, output, contract, kwargs))
                open(output, "wb").close()
                return SimpleNamespace(
                    point_count=2,
                    encoding=LasEncoding((0.001,) * 3, (421722.0, 7433393.0, 635.0)),
                )

            stage = ODMGeoreferencingStage("odm_georeferencing", args)
            with mock.patch(
                "stages.odm_georeferencing.export_georeferenced_point_cloud",
                side_effect=export,
            ), mock.patch(
                "stages.odm_georeferencing.point_cloud.post_point_cloud_steps"
            ), mock.patch("stages.odm_georeferencing.Cropper") as cropper:
                stage.process(args, outputs)
                stage.process(args, outputs)

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0], tree.filtered_point_cloud)
            self.assertEqual(calls[0][1], tree.odm_georeferencing_model_laz)
            self.assertEqual(calls[0][2], calls[1][2])
            self.assertIsNone(calls[0][2].post_crs_alignment)
            self.assertEqual(cropper.return_value.create_bounds_gpkg.call_count, 2)
            self.assertTrue(all(
                call.args[0] == tree.odm_georeferencing_model_laz
                for call in cropper.return_value.create_bounds_gpkg.call_args_list
            ))

            with mock.patch(
                "stages.odm_georeferencing.export_georeferenced_point_cloud",
                side_effect=PointCloudExportError(
                    "non-finite point", artifact=tree.odm_georeferencing_model_laz,
                    operation="write exact LAZ",
                ),
            ), mock.patch("stages.odm_georeferencing.point_cloud.post_point_cloud_steps"):
                with self.assertRaisesRegex(PointCloudExportError, "odm_georeferenced_model.laz"):
                    stage.process(args, outputs)

    def test_canonical_reconstruction_is_restored_for_later_opensfm_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            active = os.path.join(directory, "reconstruction.json")
            canonical = os.path.join(directory, "reconstruction.topocentric.json")
            with open(active, "w") as reconstruction_file:
                json.dump({"frame": "legacy-compatibility"}, reconstruction_file)
            with open(canonical, "w") as reconstruction_file:
                json.dump({"frame": "topocentric"}, reconstruction_file)

            georeferencing.materialize_canonical_reconstruction(active, canonical)

            with open(active) as reconstruction_file:
                self.assertEqual(json.load(reconstruction_file), {"frame": "topocentric"})
            with open(canonical) as reconstruction_file:
                self.assertEqual(json.load(reconstruction_file), {"frame": "topocentric"})

    def test_canonical_reconstruction_path_excludes_the_compatibility_view(self):
        tree = SimpleNamespace(
            opensfm_reconstruction="reconstruction.json",
            opensfm_topocentric_reconstruction="reconstruction.topocentric.json",
        )

        self.assertEqual(
            georeferencing.canonical_reconstruction_path(tree, True),
            "reconstruction.topocentric.json",
        )
        self.assertEqual(
            georeferencing.canonical_reconstruction_path(tree, False),
            "reconstruction.json",
        )
    def test_legacy_reconstruction_adapter_publishes_atomically_with_provenance(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            active = os.path.join(directory, "reconstruction.json")
            canonical = os.path.join(directory, "reconstruction.topocentric.json")
            generated = os.path.join(directory, "reconstruction.geocoords.json")
            provenance = os.path.join(directory, "reconstruction.compatibility.json")
            canonical_payload = [{"frame": "topocentric"}]
            compatibility_payload = [{"frame": "legacy-affine"}]
            with open(active, "w") as reconstruction_file:
                json.dump({"frame": "stale-legacy"}, reconstruction_file)
            with open(canonical, "w") as reconstruction_file:
                json.dump(canonical_payload, reconstruction_file)

            def stock_export():
                with open(active) as reconstruction_file:
                    self.assertEqual(json.load(reconstruction_file), canonical_payload)
                with open(generated, "w") as reconstruction_file:
                    json.dump(compatibility_payload, reconstruction_file)

            georeferencing.publish_legacy_reconstruction_compatibility(
                active, canonical, generated, provenance, contract, stock_export
            )

            with open(active) as reconstruction_file:
                self.assertEqual(json.load(reconstruction_file), compatibility_payload)
            with open(canonical) as reconstruction_file:
                self.assertEqual(json.load(reconstruction_file), canonical_payload)
            with open(provenance) as provenance_file:
                metadata = json.load(provenance_file)
            self.assertEqual(metadata["representation"], "affine_approximation")
            self.assertFalse(metadata["exact"])
            self.assertEqual(metadata["source"], "reconstruction.topocentric.json")
            self.assertEqual(metadata["output_crs"], contract.output_crs_wkt)
            self.assertEqual(metadata["xy_offset"], [421000.0, 7433000.0])

    def test_legacy_reconstruction_failure_is_required_without_partial_publication(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            active = os.path.join(directory, "reconstruction.json")
            canonical = os.path.join(directory, "reconstruction.topocentric.json")
            generated = os.path.join(directory, "reconstruction.geocoords.json")
            provenance = os.path.join(directory, "reconstruction.compatibility.json")
            canonical_payload = [{"frame": "topocentric"}]
            with open(active, "w") as reconstruction_file:
                json.dump({"frame": "stale-legacy"}, reconstruction_file)
            with open(canonical, "w") as reconstruction_file:
                json.dump(canonical_payload, reconstruction_file)

            def failing_stock_export():
                with open(generated, "w") as reconstruction_file:
                    json.dump({"frame": "partial"}, reconstruction_file)
                raise RuntimeError("stock export failed")

            with self.assertRaisesRegex(RuntimeError, "stock export failed"):
                georeferencing.publish_legacy_reconstruction_compatibility(
                    active, canonical, generated, provenance, contract,
                    failing_stock_export,
                )

            with open(active) as reconstruction_file:
                self.assertEqual(json.load(reconstruction_file), canonical_payload)
            self.assertFalse(os.path.exists(generated))
            self.assertFalse(os.path.exists(provenance))

    def test_legacy_reconstruction_rerun_never_projects_the_prior_compatibility_view(self):
        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            active = os.path.join(directory, "reconstruction.json")
            canonical = os.path.join(directory, "reconstruction.topocentric.json")
            generated = os.path.join(directory, "reconstruction.geocoords.json")
            provenance = os.path.join(directory, "reconstruction.compatibility.json")
            canonical_payload = [{"coordinates": [1.0, 2.0, 3.0]}]
            with open(canonical, "w") as reconstruction_file:
                json.dump(canonical_payload, reconstruction_file)
            with open(active, "w") as reconstruction_file:
                json.dump(canonical_payload, reconstruction_file)
            exported_inputs = []

            def stock_export():
                with open(active) as reconstruction_file:
                    exported_inputs.append(json.load(reconstruction_file))
                with open(generated, "w") as reconstruction_file:
                    json.dump([{"coordinates": [10.0, 20.0, 3.0]}], reconstruction_file)

            for _ in range(2):
                georeferencing.publish_legacy_reconstruction_compatibility(
                    active, canonical, generated, provenance, contract, stock_export
                )

            self.assertEqual(exported_inputs, [canonical_payload, canonical_payload])

    @unittest.skipIf(ODMGeoreferencingStage is None, "ODX stage dependencies are not installed")
    def test_full_export_flow_isolates_and_republishes_the_compatibility_view(self):
        with tempfile.TemporaryDirectory() as directory:
            args, tree, outputs = self.stage_fixture(directory)
            open(tree.filtered_point_cloud, "wb").close()
            canonical_payload = [{"frame": "topocentric", "shots": {}}]
            compatibility_payload = [{"frame": "legacy-affine", "shots": {}}]
            with open(tree.opensfm_reconstruction, "w") as reconstruction_file:
                json.dump(canonical_payload, reconstruction_file)
            with open(tree.opensfm_topocentric_reconstruction, "w") as reconstruction_file:
                json.dump(canonical_payload, reconstruction_file)
            events = []

            def export_cloud(source, output, contract, **kwargs):
                events.append("exact-public-cloud")
                open(output, "wb").close()
                return SimpleNamespace(
                    point_count=1,
                    encoding=LasEncoding((0.001,) * 3, (421000.0, 7433000.0, 635.0)),
                    bounds=None,
                )

            def run_opensfm(command):
                events.append("stock-opensfm-compatibility")
                with open(tree.opensfm_reconstruction) as reconstruction_file:
                    self.assertEqual(json.load(reconstruction_file), canonical_payload)
                with open(tree.opensfm_geocoords_reconstruction, "w") as reconstruction_file:
                    json.dump(compatibility_payload, reconstruction_file)

            with mock.patch(
                "stages.odm_georeferencing.export_georeferenced_point_cloud",
                side_effect=export_cloud,
            ), mock.patch(
                "stages.odm_georeferencing.point_cloud.post_point_cloud_steps"
            ), mock.patch(
                "stages.odm_georeferencing.OSFMContext.run", side_effect=run_opensfm
            ) as stock_export:
                stage = ODMGeoreferencingStage("odm_georeferencing", args)
                stage.process(args, outputs)
                stage.process(args, outputs)

            self.assertEqual(events, [
                "exact-public-cloud", "stock-opensfm-compatibility",
                "exact-public-cloud", "stock-opensfm-compatibility",
            ])
            stock_export.assert_has_calls([mock.call(
                'export_geocoords --reconstruction --proj "+proj=utm +zone=23 +south +datum=WGS84 +units=m +no_defs +type=crs" --offset-x 421000.0 --offset-y 7433000.0'
            )] * 2)
            with open(tree.opensfm_reconstruction) as reconstruction_file:
                self.assertEqual(json.load(reconstruction_file), compatibility_payload)
            with open(tree.opensfm_topocentric_reconstruction) as reconstruction_file:
                self.assertEqual(json.load(reconstruction_file), canonical_payload)
            with open(tree.opensfm_compatibility_provenance) as provenance_file:
                provenance = json.load(provenance_file)
            self.assertEqual(provenance["representation"], "affine_approximation")
            self.assertEqual(provenance["source"], "reconstruction.topocentric.json")
            self.assertEqual(provenance["output_crs"], outputs["coordinate_contract"].output_crs_wkt)
            self.assertEqual(provenance["xy_offset"], [421000.0, 7433000.0])

            dem_args = SimpleNamespace(
                dsm=False, dtm=False, dem_resolution=5.0, ignore_gsd=False,
                rerun=None, rerun_all=False, rerun_from=None,
            )
            with mock.patch("stages.odm_dem.gsd.cap_resolution", return_value=5.0) as cap:
                ODMDEMStage("odm_dem", dem_args).process(
                    dem_args,
                    {"tree": tree, "reconstruction": GeoreferencedReconstructionFixture()},
                )
            self.assertEqual(
                cap.call_args.args[1], tree.opensfm_topocentric_reconstruction
            )

            report_args = SimpleNamespace(
                skip_report=True, rerun=None, rerun_all=False, rerun_from=None,
            )
            with mock.patch(
                "stages.odm_report.get_geojson_shots_from_opensfm",
                return_value={"type": "FeatureCollection", "features": []},
            ) as report_export:
                ODMReport("odm_report", report_args).process(
                    report_args,
                    {
                        "tree": tree,
                        "reconstruction": GeoreferencedReconstructionFixture(),
                        "coordinate_contract": outputs["coordinate_contract"],
                    },
                )
            self.assertEqual(
                report_export.call_args.args[0],
                tree.opensfm_topocentric_reconstruction,
            )

    @unittest.skipIf(ODMGeoreferencingStage is None, "ODX stage dependencies are not installed")
    def test_stage_resolves_alignment_before_publishing_public_derivatives(self):
        with tempfile.TemporaryDirectory() as directory:
            args, tree, outputs = self.stage_fixture(directory)
            open(tree.filtered_point_cloud, "wb").close()
            tree.odm_align_file = os.path.join(directory, "align.laz")
            open(tree.odm_align_file, "wb").close()
            alignment = np.asarray([
                [1.0, 0.1, 0.0, 3.0],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ])
            calls = []

            def export(source, output, contract, **kwargs):
                calls.append((output, contract))
                open(output, "wb").close()
                return SimpleNamespace(
                    point_count=2,
                    encoding=LasEncoding((0.001,) * 3, (421000.0, 7433000.0, 635.0)),
                    bounds=None,
                )

            with mock.patch(
                "stages.odm_georeferencing.export_georeferenced_point_cloud",
                side_effect=export,
            ), mock.patch(
                "stages.odm_georeferencing.compute_alignment_matrix",
                return_value=alignment,
            ) as compute, mock.patch(
                "stages.odm_georeferencing.point_cloud.post_point_cloud_steps"
            ):
                ODMGeoreferencingStage("odm_georeferencing", args).process(args, outputs)

            unaligned = os.path.splitext(tree.odm_georeferencing_model_laz)[0] + "_unaligned.laz"
            alignment_input = os.path.splitext(tree.odm_georeferencing_model_laz)[0] + "_alignment_input.laz"
            self.assertEqual(
                [call[0] for call in calls],
                [alignment_input, unaligned, tree.odm_georeferencing_model_laz],
            )
            self.assertIsNone(calls[0][1].post_crs_alignment)
            self.assertIsNone(calls[1][1].post_crs_alignment)
            self.assertEqual(calls[2][1].post_crs_alignment, tuple(alignment.flat))
            compute.assert_called_once_with(alignment_input, tree.odm_align_file, mock.ANY)
            self.assertEqual(outputs["coordinate_contract"], calls[2][1])

    @unittest.skipIf(
        pdal is None or ODMGeoreferencingStage is None,
        "PDAL or ODX stage dependencies are not installed",
    )
    def test_stage_decodes_exact_output_from_the_canonical_topocentric_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            args, tree, outputs = self.stage_fixture(directory)
            with open(tree.filtered_point_cloud, "w") as point_cloud:
                point_cloud.write(
                    "ply\nformat ascii 1.0\nelement vertex 2\n"
                    "property double x\nproperty double y\nproperty double z\n"
                    "property double nx\nproperty double ny\nproperty double nz\n"
                    "property ushort intensity\nproperty uchar views\nend_header\n"
                    "0 0 0 0 0 1 7 4\n100 50 10 1 0 0 9 2\n"
                )
            stage = ODMGeoreferencingStage("odm_georeferencing", args)
            with mock.patch("stages.odm_georeferencing.point_cloud.post_point_cloud_steps"):
                stage.process(args, outputs)
                first_pipeline = pdal.Pipeline(json.dumps([tree.odm_georeferencing_model_laz]))
                first_pipeline.execute()
                first = first_pipeline.arrays[0].copy()
                base_contract = outputs["coordinate_contract"]
                alignment = np.eye(4)
                alignment[:3, 3] = [0.25, -0.5, 1.0]
                aligned_contract = base_contract.with_post_crs_alignment(alignment)
                aligned_contract.persist(
                    tree.path("odm_georeferencing", "coordinate_contract.json")
                )
                outputs["coordinate_contract"] = aligned_contract
                stage.process(args, outputs)
                second_pipeline = pdal.Pipeline(json.dumps([tree.odm_georeferencing_model_laz]))
                second_pipeline.execute()
                second = second_pipeline.arrays[0].copy()
                stage.process(args, outputs)
                third_pipeline = pdal.Pipeline(json.dumps([tree.odm_georeferencing_model_laz]))
                third_pipeline.execute()
                third = third_pipeline.arrays[0]
                unaligned_path = os.path.splitext(tree.odm_georeferencing_model_laz)[0] + "_unaligned.laz"
                unaligned_pipeline = pdal.Pipeline(json.dumps([unaligned_path]))
                unaligned_pipeline.execute()
                unaligned = unaligned_pipeline.arrays[0]

            expected_base = base_contract.transform_points(
                [[0.0, 0.0, 0.0], [100.0, 50.0, 10.0]],
                apply_storage_offset=False,
            )
            expected_aligned = aligned_contract.transform_points(
                [[0.0, 0.0, 0.0], [100.0, 50.0, 10.0]],
                apply_storage_offset=False,
            )
            first_coordinates = np.column_stack((first["X"], first["Y"], first["Z"]))
            second_coordinates = np.column_stack((second["X"], second["Y"], second["Z"]))
            np.testing.assert_allclose(first_coordinates, expected_base, atol=0.0006, rtol=0.0)
            np.testing.assert_allclose(
                second_coordinates, expected_aligned, atol=0.0006, rtol=0.0
            )
            np.testing.assert_array_equal(third, second)
            np.testing.assert_allclose(
                np.column_stack((unaligned["X"], unaligned["Y"], unaligned["Z"])),
                expected_base, atol=0.0006, rtol=0.0,
            )
            self.assertEqual(len(second), 2)
            self.assertIn("Intensity", second.dtype.names)
            self.assertTrue(
                pdal.Pipeline(json.dumps([tree.odm_georeferencing_model_laz]))
                .quickinfo["readers.las"]["srs"]["wkt"]
            )

    @unittest.skipIf(
        pdal is None or point_cloud_module is None,
        "PDAL or ODX derivative dependencies are not installed",
    )
    def test_las_xyz_ept_and_copc_receive_the_public_laz(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "canonical.ply")
            public_laz = os.path.join(directory, "odm_georeferenced_model.laz")
            with open(source, "w") as point_cloud:
                point_cloud.write(
                    "ply\nformat ascii 1.0\nelement vertex 1\n"
                    "property double x\nproperty double y\nproperty double z\n"
                    "property float nx\nproperty float ny\nproperty float nz\n"
                    "property ushort intensity\nproperty uchar views\nend_header\n"
                    "0 0 0 0 0 1 7 4\n"
                )
            contract = resolve_coordinate_contract(
                self.anchor, self.controls([(0.0, 0.0, 0.0)])
            )
            export_georeferenced_point_cloud(
                source, public_laz, contract, spacing=0.024, chunk_size=1,
            )
            decoded = pdal.Pipeline(json.dumps([public_laz]))
            self.assertEqual(decoded.execute(), 1)
            self.assertEqual(
                set(decoded.arrays[0].dtype.names), set(PUBLIC_LAZ_DIMENSIONS)
            )
            tree = SimpleNamespace(
                odm_georeferencing=directory,
                odm_georeferencing_model_laz=public_laz,
                odm_georeferencing_xyz_file=os.path.join(directory, "model.csv"),
                odm_georeferencing_model_las=os.path.join(directory, "model.las"),
                entwine_pointcloud=os.path.join(directory, "ept"),
            )
            args = SimpleNamespace(
                pc_classify=False, pc_csv=True, pc_las=True, pc_ept=True,
                pc_copc=True, max_concurrency=2,
            )
            with mock.patch("opendm.point_cloud.system.run") as run, mock.patch(
                "opendm.point_cloud.entwine.build"
            ) as build_ept, mock.patch(
                "opendm.point_cloud.entwine.build_copc"
            ) as build_copc:
                point_cloud_module.post_point_cloud_steps(args, tree, rerun=True)

            self.assertTrue(all(public_laz in call.args[0] for call in run.call_args_list))
            build_ept.assert_called_once()
            build_copc.assert_called_once()
            self.assertEqual(build_ept.call_args.args[0], [public_laz])
            self.assertEqual(build_copc.call_args.args[0], [public_laz])

    @unittest.skipIf(build_3dtiles is None, "ODX tile dependencies are not installed")
    def test_point_tiles_receive_the_public_laz(self):
        with tempfile.TemporaryDirectory() as directory:
            public_laz = os.path.join(directory, "odm_georeferenced_model.laz")
            open(public_laz, "wb").close()
            tree = SimpleNamespace(
                odm_georeferencing_model_laz=public_laz,
                ogc_tiles=os.path.join(directory, "tiles"),
                odm_texturing=os.path.join(directory, "texturing"),
                odm_25dtexturing=os.path.join(directory, "texturing25d"),
                odm_textured_model_obj="model.obj",
                opensfm=directory,
                odm_georeferencing=directory,
            )
            os.makedirs(tree.odm_texturing)
            open(os.path.join(tree.odm_texturing, tree.odm_textured_model_obj), "w").close()
            os.mkdir(tree.ogc_tiles)
            with mock.patch("opendm.ogctiles.build_textured_model") as build_model_tiles, mock.patch(
                "opendm.ogctiles.build_pointcloud"
            ) as build_point_tiles:
                build_3dtiles(
                    SimpleNamespace(max_concurrency=2), tree,
                    SimpleNamespace(), rerun=True,
                )
            build_point_tiles.assert_called_once_with(
                public_laz, os.path.join(tree.ogc_tiles, "pointcloud"), 2, True
            )
            self.assertEqual(
                build_model_tiles.call_args.args[0],
                os.path.join(tree.odm_texturing, tree.odm_textured_model_obj),
            )

    def test_raster_export_uses_exact_gdal_operation_and_absolute_affine(self):
        class FakeDataset:
            RasterXSize = 8
            RasterYSize = 6

            def FlushCache(self):
                pass

        class FakeGDAL:
            GDT_Unknown = 0

            def __init__(self):
                self.translate_options = None
                self.warp_options = None

            def TranslateOptions(self, **kwargs):
                self.translate_options = kwargs
                return kwargs

            def Translate(self, destination, source, options=None):
                self.translate = (destination, source, options)
                return FakeDataset()

            def WarpOptions(self, **kwargs):
                self.warp_options = kwargs
                return kwargs

            def Warp(self, destination, source, options=None):
                self.warp = (destination, source, options)
                open(destination, "wb").close()
                return FakeDataset()

            def Transformer(self, source, destination, options):
                self.transformer_options = options

                class Transformer:
                    @staticmethod
                    def TransformPoints(inverse, points):
                        topocentric = np.asarray([
                            (column * 1000.0 / 8.0,
                             750.0 - row * 750.0 / 6.0, z)
                            for column, row, z in points
                        ])
                        transformed = contract.transform_points(
                            topocentric, apply_storage_offset=False
                        )
                        return transformed.tolist(), [1] * len(points)

                return Transformer()

        contract = resolve_coordinate_contract(
            self.anchor,
            self.controls([(0.0, 0.0, 0.0), (1000.0, 750.0, 0.0)]),
            storage_offset=(421000.0, 7433000.0),
        )
        gdal = FakeGDAL()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "working.tif")
            corners = os.path.join(directory, "corners.txt")
            output = os.path.join(directory, "odm_orthophoto.tif")
            open(source, "wb").close()
            with open(corners, "w") as corner_file:
                corner_file.write("0 0 1000 750")

            result = export_georeferenced_raster(
                source, corners, output, contract,
                resolution=0.05,
                creation_options={"TILED": "YES", "COMPRESS": "DEFLATE"},
                memory_limit_mb=59_989.6,
                gdal_module=gdal,
            )

        self.assertIsInstance(result, RasterExportResult)
        self.assertEqual(result.source_bounds, (0.0, 0.0, 1000.0, 750.0))
        self.assertEqual(gdal.translate_options["outputBounds"], [0.0, 750.0, 1000.0, 0.0])
        self.assertEqual(gdal.warp_options["errorThreshold"], 0.0)
        self.assertEqual(gdal.warp_options["coordinateOperation"], contract.operation)
        self.assertEqual(gdal.warp_options["dstSRS"], contract.output_crs_wkt)
        self.assertEqual(gdal.warp_options["xRes"], 0.05)
        self.assertEqual(gdal.warp_options["yRes"], 0.05)
        self.assertEqual(gdal.warp_options["resampleAlg"], "near")
        self.assertTrue(gdal.warp_options["multithread"])
        self.assertEqual(gdal.warp_options["warpMemoryLimit"], 31_451_827_404.8)
        self.assertLessEqual(result.maximum_transformation_error, 0.0001)
        self.assertNotIn("421000", gdal.warp_options["coordinateOperation"])
        self.assertNotIn("7433000", gdal.warp_options["coordinateOperation"])

    @unittest.skipIf(ODMOrthoPhotoStage is None, "ODX orthophoto dependencies are not installed")
    def test_orthophoto_stage_fails_closed_without_persisted_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            os.makedirs(tree.odm_orthophoto)
            os.makedirs(tree.odm_texturing)
            public = os.path.join(tree.odm_texturing, tree.odm_textured_model_obj)
            with open(public, "w") as mesh:
                mesh.write("v 0 0 0\n")
            contract = resolve_coordinate_contract(
                self.anchor, self.controls([(0.0, 0.0, 0.0)])
            )
            args = SimpleNamespace(
                skip_orthophoto=False, orthophoto_resolution=5.0,
                ignore_gsd=False, use_3dmesh=True, primary_band=None,
                max_concurrency=2, orthophoto_no_tiled=False,
                orthophoto_compression="DEFLATE", orthophoto_cutline=False,
                build_overviews=False, orthophoto_png=False,
                orthophoto_kmz=False, tiles=False, cog=False,
                crop=0, boundary=None, optimize_disk_space=False,
                rerun=None, rerun_all=True, rerun_from=None,
            )
            reconstruction = GeoreferencedReconstructionFixture()
            reconstruction.photos = [SimpleNamespace(band_name="RGB")]
            outputs = {
                "tree": tree, "reconstruction": reconstruction,
                "coordinate_contract": contract, "large": False,
            }
            stage = ODMOrthoPhotoStage("odm_orthophoto", args)
            with mock.patch("stages.odm_orthophoto.system.run") as run:
                with self.assertRaisesRegex(CoordinateContractError, "requires a readable persisted coordinate contract"):
                    stage.process(args, outputs)
                run.assert_not_called()

    @unittest.skipIf(ODMOrthoPhotoStage is None, "ODX orthophoto dependencies are not installed")
    def test_orthophoto_stage_direct_raster_uses_valid_public_mesh_without_feature_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            os.makedirs(tree.odm_orthophoto)
            os.makedirs(tree.odm_texturing)
            os.makedirs(tree.odm_georeferencing)
            public = os.path.join(tree.odm_texturing, tree.odm_textured_model_obj)
            with open(public, "w") as mesh:
                mesh.write("v 0 0 0\n")
            contract = resolve_coordinate_contract(
                self.anchor, self.controls([(0.0, 0.0, 0.0)]),
                storage_offset=(421000.0, 7433000.0),
            )
            contract.persist(os.path.join(tree.odm_georeferencing, "coordinate_contract.json"))
            args = SimpleNamespace(
                skip_orthophoto=False, orthophoto_resolution=5.0,
                ignore_gsd=False, use_3dmesh=True, primary_band=None,
                max_concurrency=2, orthophoto_no_tiled=False,
                orthophoto_compression="DEFLATE", orthophoto_cutline=False,
                build_overviews=False, orthophoto_png=False,
                orthophoto_kmz=False, tiles=False, cog=False,
                crop=0, boundary=None, optimize_disk_space=False,
                rerun=None, rerun_all=True, rerun_from=None,
            )
            reconstruction = GeoreferencedReconstructionFixture()
            reconstruction.photos = [SimpleNamespace(band_name="RGB")]
            outputs = {
                "tree": tree, "reconstruction": reconstruction,
                "coordinate_contract": contract, "large": False,
            }
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                open(tree.odm_orthophoto_tif, "wb").close()

            stage = ODMOrthoPhotoStage("odm_orthophoto", args)
            with mock.patch(
                "stages.odm_orthophoto.gsd.cap_resolution", return_value=5.0
            ), mock.patch("stages.odm_orthophoto.system.run", side_effect=run), mock.patch(
                "stages.odm_orthophoto.orthophoto.post_orthophoto_steps"
            ):
                stage.process(args, outputs)

            self.assertIn(public, calls[0])
            self.assertIn("-utm_north_offset 7433000.0", calls[0])
            self.assertIn("-utm_east_offset 421000.0", calls[0])
            self.assertIn("-a_srs", calls[0])
            with open(os.path.join(tree.odm_georeferencing, "raster-export.json")) as evidence:
                self.assertEqual(json.load(evidence)["mode"], "direct")

    @unittest.skipIf(ODMOrthoPhotoStage is None, "ODX orthophoto dependencies are not installed")
    def test_orthophoto_stage_fails_closed_without_valid_public_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            os.makedirs(tree.odm_orthophoto)
            os.makedirs(tree.odm_texturing)
            os.makedirs(tree.odm_georeferencing)
            contract = resolve_coordinate_contract(self.anchor, self.controls([(0.0, 0.0, 0.0)]))
            contract.persist(os.path.join(tree.odm_georeferencing, "coordinate_contract.json"))
            args = SimpleNamespace(
                skip_orthophoto=False, orthophoto_resolution=5.0,
                ignore_gsd=False, use_3dmesh=True, primary_band=None,
                max_concurrency=2, orthophoto_no_tiled=False,
                orthophoto_compression="DEFLATE", orthophoto_cutline=False,
                build_overviews=False, orthophoto_png=False,
                orthophoto_kmz=False, tiles=False, cog=False,
                crop=0, boundary=None, optimize_disk_space=False,
                rerun=None, rerun_all=True, rerun_from=None,
            )
            reconstruction = GeoreferencedReconstructionFixture()
            reconstruction.photos = [SimpleNamespace(band_name="RGB")]
            outputs = {"tree": tree, "reconstruction": reconstruction, "coordinate_contract": contract, "large": False}

            stage = ODMOrthoPhotoStage("odm_orthophoto", args)
            with mock.patch("stages.odm_orthophoto.system.run") as run:
                with self.assertRaisesRegex(CoordinateContractError, "requires nonempty public georeferenced meshes"):
                    stage.process(args, outputs)
                run.assert_not_called()

    @unittest.skipIf(ODMOrthoPhotoStage is None, "ODX orthophoto dependencies are not installed")
    def test_orthophoto_stage_fails_closed_for_empty_public_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            os.makedirs(tree.odm_orthophoto)
            os.makedirs(tree.odm_texturing)
            os.makedirs(tree.odm_georeferencing)
            public = os.path.join(tree.odm_texturing, tree.odm_textured_model_obj)
            open(public, "w").close()
            contract = resolve_coordinate_contract(self.anchor, self.controls([(0.0, 0.0, 0.0)]))
            contract.persist(os.path.join(tree.odm_georeferencing, "coordinate_contract.json"))
            args = SimpleNamespace(
                skip_orthophoto=False, orthophoto_resolution=5.0,
                ignore_gsd=False, use_3dmesh=True, primary_band=None,
                max_concurrency=2, orthophoto_no_tiled=False,
                orthophoto_compression="DEFLATE", orthophoto_cutline=False,
                build_overviews=False, orthophoto_png=False,
                orthophoto_kmz=False, tiles=False, cog=False,
                crop=0, boundary=None, optimize_disk_space=False,
                rerun=None, rerun_all=True, rerun_from=None,
            )
            reconstruction = GeoreferencedReconstructionFixture()
            reconstruction.photos = [SimpleNamespace(band_name="RGB")]
            outputs = {"tree": tree, "reconstruction": reconstruction, "coordinate_contract": contract, "large": False}

            stage = ODMOrthoPhotoStage("odm_orthophoto", args)
            with mock.patch("stages.odm_orthophoto.system.run") as run:
                with self.assertRaisesRegex(CoordinateContractError, "requires nonempty public georeferenced meshes"):
                    stage.process(args, outputs)
                run.assert_not_called()

    @unittest.skipIf(ODMOrthoPhotoStage is None, "ODX orthophoto dependencies are not installed")
    def test_orthophoto_stage_fails_closed_when_contract_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            os.makedirs(tree.odm_orthophoto)
            os.makedirs(tree.odm_texturing)
            os.makedirs(tree.odm_georeferencing)
            public = os.path.join(tree.odm_texturing, tree.odm_textured_model_obj)
            with open(public, "w") as mesh:
                mesh.write("v 0 0 0\n")
            stage_contract = resolve_coordinate_contract(
                self.anchor, self.controls([(0.0, 0.0, 0.0)]),
                storage_offset=(421000.0, 7433000.0),
            )
            persisted = resolve_coordinate_contract(
                self.anchor, self.controls([(0.0, 0.0, 0.0)]),
                storage_offset=(421100.0, 7433000.0),
            )
            persisted.persist(os.path.join(tree.odm_georeferencing, "coordinate_contract.json"))
            args = SimpleNamespace(
                skip_orthophoto=False, orthophoto_resolution=5.0,
                ignore_gsd=False, use_3dmesh=True, primary_band=None,
                max_concurrency=2, orthophoto_no_tiled=False,
                orthophoto_compression="DEFLATE", orthophoto_cutline=False,
                build_overviews=False, orthophoto_png=False,
                orthophoto_kmz=False, tiles=False, cog=False,
                crop=0, boundary=None, optimize_disk_space=False,
                rerun=None, rerun_all=True, rerun_from=None,
            )
            reconstruction = GeoreferencedReconstructionFixture()
            reconstruction.photos = [SimpleNamespace(band_name="RGB")]
            outputs = {"tree": tree, "reconstruction": reconstruction, "coordinate_contract": stage_contract, "large": False}

            stage = ODMOrthoPhotoStage("odm_orthophoto", args)
            with mock.patch("stages.odm_orthophoto.system.run") as run:
                with self.assertRaisesRegex(CoordinateContractError, "does not match the stage contract"):
                    stage.process(args, outputs)
                run.assert_not_called()

    def test_raster_export_rejects_invalid_control_and_preserves_context(self):
        contract = resolve_coordinate_contract(
            self.anchor, self.controls([(0.0, 0.0, 0.0)])
        )
        with tempfile.TemporaryDirectory() as directory:
            corners = os.path.join(directory, "corners.txt")
            with open(corners, "w") as corner_file:
                corner_file.write("0 0 nan 10")
            with self.assertRaisesRegex(RasterExportError, "corner control"):
                export_georeferenced_raster(
                    os.path.join(directory, "working.tif"), corners,
                    os.path.join(directory, "output.tif"), contract,
                    resolution=0.05,
                )

    @unittest.skipIf(gdal is None, "GDAL Python bindings are not installed")
    def test_exact_raster_fixtures_cover_utm_boundaries_and_poles(self):
        scenarios = (
            ("northern", TopocentricAnchor(45.0, 8.0, 100.0), 2000.0, 250.0, None),
            ("southern", TopocentricAnchor(-23.0, -45.0, 600.0), 2000.0, 250.0, None),
            ("zone-crossing", TopocentricAnchor(0.0, -174.1, 10.0), 30000.0, 3750.0, None),
            ("north-polar", TopocentricAnchor(85.0, 10.0, 10.0), 2000.0, 250.0, None),
            ("south-polar", TopocentricAnchor(-81.0, 10.0, 10.0), 2000.0, 250.0, None),
            (
                "southern-aligned", TopocentricAnchor(-23.0, -45.0, 600.0),
                2000.0, 250.0,
                np.asarray([
                    [0.99995, -0.0099998, 0.25, 4.0],
                    [0.0099998, 0.99995, -0.4, -6.0],
                    [0.0, 0.0, 1.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, anchor, extent, resolution, alignment in scenarios:
                with self.subTest(name=name):
                    source = os.path.join(directory, name + "-working.tif")
                    corners = os.path.join(directory, name + "-corners.txt")
                    output = os.path.join(directory, name + "-output.tif")
                    dataset = gdal.GetDriverByName("GTiff").Create(
                        source, 8, 6, 4, gdal.GDT_UInt16,
                    )
                    for band_index in range(1, 5):
                        band = dataset.GetRasterBand(band_index)
                        band.Fill(17 * band_index)
                        band.SetNoDataValue(65535)
                    dataset = None
                    with open(corners, "w") as corner_file:
                        corner_file.write("0 0 {0} {0}".format(extent))
                    contract = resolve_coordinate_contract(
                        anchor,
                        NormalizedControlSet.from_coordinates(
                            [(0.0, 0.0, 0.0), (extent, extent, 0.0)],
                            vertical_control=False,
                        ),
                    )
                    if alignment is not None:
                        contract = contract.with_post_crs_alignment(alignment)

                    result = export_georeferenced_raster(
                        source, corners, output, contract,
                        resolution=resolution,
                        creation_options={"TILED": "YES"},
                        memory_limit_mb=64,
                    )
                    raster = gdal.Open(output)
                    self.assertIsNotNone(raster)
                    self.assertEqual(
                        CRS.from_wkt(raster.GetProjection()).to_epsg(),
                        contract.output_crs.to_epsg(),
                    )
                    transform = raster.GetGeoTransform()
                    self.assertEqual(transform[2], 0.0)
                    self.assertEqual(transform[4], 0.0)
                    self.assertLess(transform[5], 0.0)
                    self.assertAlmostEqual(transform[1], resolution)
                    self.assertAlmostEqual(transform[5], -resolution)
                    self.assertGreater(abs(transform[0]), 1000.0)
                    self.assertEqual(raster.RasterCount, 4)
                    self.assertEqual(raster.GetRasterBand(1).DataType, gdal.GDT_UInt16)
                    self.assertEqual(raster.GetRasterBand(1).GetNoDataValue(), 65535)
                    self.assertLessEqual(result.maximum_transformation_error, 0.0001)

                    fractions = np.linspace(0.0, 1.0, 5)
                    sample_xy = np.asarray([
                        (extent * x, extent * y, 0.0)
                        for y in fractions for x in fractions
                    ])
                    topocentric = gdal.Translate(
                        os.path.join(directory, name + "-mapping.vrt"),
                        source, format="VRT",
                        outputBounds=[0.0, extent, extent, 0.0],
                        outputSRS="EPSG:4978",
                    )
                    transformer = gdal.Transformer(topocentric, None, [
                        "SRC_SRS=EPSG:4978",
                        "DST_SRS={}".format(contract.output_crs_wkt),
                        "COORDINATE_OPERATION={}".format(contract.operation),
                    ])
                    pixels = [
                        (topocentric.RasterXSize * x,
                         topocentric.RasterYSize * (1.0 - y), 0.0)
                        for y in fractions for x in fractions
                    ]
                    mapped, successful = transformer.TransformPoints(False, pixels)
                    self.assertTrue(all(successful))
                    topo_to_geographic = Transformer.from_pipeline(
                        "+proj=pipeline +step +inv +proj=topocentric +ellps=WGS84 "
                        "+lat_0={:.15g} +lon_0={:.15g} +h_0={:.15g} "
                        "+step +inv +proj=cart +ellps=WGS84".format(
                            anchor.latitude, anchor.longitude, anchor.ellipsoidal_height
                        )
                    )
                    output_transform = Transformer.from_crs(
                        CRS.from_epsg(4979), contract.output_crs, always_xy=True
                    )
                    lon, lat, height = topo_to_geographic.transform(*sample_xy.T)
                    independent = np.column_stack(
                        output_transform.transform(lon, lat, height)
                    )
                    if alignment is None:
                        np.testing.assert_allclose(
                            np.asarray(mapped)[:, :2], independent[:, :2],
                            atol=0.0001, rtol=0.0,
                        )
                    else:
                        independent[:, 2] = sample_xy[:, 2]
                        independent = (
                            np.column_stack((independent, np.ones(len(independent))))
                            @ alignment.T
                        )[:, :3]

                    edge = np.linspace(0.0, extent, 21)
                    source_footprint = np.asarray(
                        [(x, 0.0, 0.0) for x in edge]
                        + [(extent, y, 0.0) for y in edge]
                        + [(x, extent, 0.0) for x in edge]
                        + [(0.0, y, 0.0) for y in edge]
                    )
                    lon, lat, height = topo_to_geographic.transform(*source_footprint.T)
                    exact_footprint = np.column_stack(
                        output_transform.transform(lon, lat, height)
                    )
                    if alignment is not None:
                        exact_footprint[:, 2] = source_footprint[:, 2]
                        exact_footprint = (
                            np.column_stack((
                                exact_footprint, np.ones(len(exact_footprint))
                            )) @ alignment.T
                        )[:, :3]
                    exact_footprint = exact_footprint[:, :2]
                    left = transform[0]
                    top = transform[3]
                    right = left + raster.RasterXSize * transform[1]
                    bottom = top + raster.RasterYSize * transform[5]
                    tolerance = resolution * 0.5 + 0.0001
                    self.assertAlmostEqual(
                        left, np.min(exact_footprint[:, 0]), delta=tolerance
                    )
                    self.assertAlmostEqual(
                        right, np.max(exact_footprint[:, 0]), delta=tolerance
                    )
                    self.assertAlmostEqual(
                        bottom, np.min(exact_footprint[:, 1]), delta=tolerance
                    )
                    self.assertAlmostEqual(
                        top, np.max(exact_footprint[:, 1]), delta=tolerance
                    )
                    raster = None

    @unittest.skipIf(ODMDEMStage is None, "ODX DEM dependencies are not installed")
    def test_dem_receives_the_public_laz(self):
        with tempfile.TemporaryDirectory() as directory:
            public_laz = os.path.join(directory, "odm_georeferenced_model.laz")
            open(public_laz, "wb").close()
            dem_tree = SimpleNamespace(
                odm_georeferencing_model_laz=public_laz,
                opensfm_reconstruction=os.path.join(directory, "reconstruction.json"),
                opensfm_topocentric_reconstruction=os.path.join(directory, "reconstruction.topocentric.json"),
                filtered_point_cloud_stats=os.path.join(directory, "point_cloud_stats.json"),
                odm_georeferencing=directory,
                path=lambda *parts: os.path.join(directory, *parts),
            )
            dem_args = SimpleNamespace(
                dsm=True, dtm=False, dem_resolution=5.0, ignore_gsd=False,
                dem_gapfill_steps=0, dem_euclidean_map=False,
                dem_decimation=1, max_concurrency=2, crop=0, boundary=None,
                optimize_disk_space=False, tiles=False, cog=False,
                rerun=None, rerun_all=False, rerun_from=None,
            )
            reconstruction = SimpleNamespace(
                is_georeferenced=lambda: True, has_gcp=lambda: False,
                has_geotagged_photos=lambda: True, photos=[],
            )
            dem_stage = ODMDEMStage("odm_dem", dem_args)
            with mock.patch("stages.odm_dem.gsd.cap_resolution", return_value=5.0), mock.patch(
                "stages.odm_dem.commands.get_dem_radius_steps", return_value=[1]
            ), mock.patch("stages.odm_dem.commands.create_dem") as create_dem, mock.patch(
                "stages.odm_dem.add_raster_meta_tags"
            ):
                dem_stage.process(dem_args, {"tree": dem_tree, "reconstruction": reconstruction, "large": False})
            self.assertEqual(create_dem.call_args.args[0], public_laz)

    @unittest.skipIf(ODMDEMStage is None, "ODX DEM dependencies are not installed")
    def test_dsm_dtm_derivatives_use_final_rasters_from_exact_laz(self):
        with tempfile.TemporaryDirectory() as directory:
            public_laz = os.path.join(directory, "odm_georeferenced_model.laz")
            open(public_laz, "wb").close()
            dem_tree = SimpleNamespace(
                odm_georeferencing_model_laz=public_laz,
                opensfm_reconstruction=os.path.join(directory, "reconstruction.json"),
                opensfm_topocentric_reconstruction=os.path.join(directory, "reconstruction.topocentric.json"),
                filtered_point_cloud_stats=os.path.join(directory, "point_cloud_stats.json"),
                odm_georeferencing=directory,
                path=lambda *parts: os.path.join(directory, *parts),
            )
            args = SimpleNamespace(
                dsm=True, dtm=True, dem_resolution=5.0, ignore_gsd=False,
                dem_gapfill_steps=0, dem_euclidean_map=False,
                dem_decimation=1, max_concurrency=2, crop=0, boundary=None,
                optimize_disk_space=False, tiles=True, cog=True,
                rerun=None, rerun_all=False, rerun_from=None,
            )
            reconstruction = SimpleNamespace(
                is_georeferenced=lambda: True, has_gcp=lambda: False,
                has_geotagged_photos=lambda: True, photos=[],
            )

            def create_dem(source, product, **kwargs):
                self.assertEqual(source, public_laz)
                open(os.path.join(directory, "odm_dem", product + ".tif"), "wb").close()

            stage = ODMDEMStage("odm_dem", args)
            with mock.patch("stages.odm_dem.gsd.cap_resolution", return_value=5.0), mock.patch(
                "stages.odm_dem.commands.get_dem_radius_steps", return_value=[1]
            ), mock.patch(
                "stages.odm_dem.commands.create_dem", side_effect=create_dem
            ) as create, mock.patch(
                "stages.odm_dem.add_raster_meta_tags"
            ) as tags, mock.patch(
                "stages.odm_dem.generate_dem_tiles"
            ) as tiles, mock.patch(
                "stages.odm_dem.convert_to_cogeo"
            ) as cog:
                stage.process(
                    args,
                    {"tree": dem_tree, "reconstruction": reconstruction, "large": False},
                )

            self.assertEqual([call.args[1] for call in create.call_args_list], ["dsm", "dtm"])
            final_rasters = [
                os.path.join(directory, "odm_dem", "dsm.tif"),
                os.path.join(directory, "odm_dem", "dtm.tif"),
            ]
            self.assertEqual([call.args[0] for call in tags.call_args_list], final_rasters)
            self.assertEqual([call.args[0] for call in tiles.call_args_list], final_rasters)
            self.assertEqual([call.args[0] for call in cog.call_args_list], final_rasters)

    @unittest.skipIf(ODMReport is None, "ODX report dependencies are not installed")
    def test_skip_report_keeps_current_optional_output_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            opensfm = os.path.join(directory, "opensfm")
            os.makedirs(opensfm)
            topocentric = os.path.join(opensfm, "reconstruction.topocentric.json")
            with open(topocentric, "w") as output:
                json.dump([], output)
            tree = SimpleNamespace(
                odm_report=os.path.join(directory, "odm_report"),
                opensfm=opensfm,
                opensfm_topocentric_reconstruction=topocentric,
                opensfm_reconstruction=os.path.join(opensfm, "reconstruction.json"),
                odm_orthophoto_tif=os.path.join(directory, "missing.tif"),
            )
            args = SimpleNamespace(
                skip_report=True, rerun=None, rerun_all=False, rerun_from=None,
            )
            contract = resolve_coordinate_contract(
                self.anchor, self.controls([(0.0, 0.0, 0.0)])
            )
            report = {"type": "FeatureCollection", "features": []}
            with mock.patch(
                "stages.odm_report.get_geojson_shots_from_opensfm",
                return_value=report,
            ) as export_shots:
                ODMReport("odm_report", args).process(
                    args,
                    {
                        "tree": tree,
                        "reconstruction": GeoreferencedReconstructionFixture(),
                        "coordinate_contract": contract,
                    },
                )
            export_shots.assert_called_once_with(
                topocentric, coordinate_contract=contract
            )
            with open(os.path.join(tree.odm_report, "shots.geojson")) as output:
                self.assertEqual(json.load(output), report)

    @unittest.skipIf(ODMReport is None, "ODX report dependencies are not installed")
    def test_report_statistics_receive_the_public_laz(self):
        with tempfile.TemporaryDirectory() as directory:
            public_laz = os.path.join(directory, "odm_georeferenced_model.laz")
            open(public_laz, "wb").close()
            reconstruction = GeoreferencedReconstructionFixture()
            report_root = os.path.join(directory, "report_project")
            opensfm = os.path.join(report_root, "opensfm")
            os.makedirs(os.path.join(opensfm, "stats"))
            with open(os.path.join(opensfm, "stats", "stats.json"), "w") as stats:
                json.dump({}, stats)
            topocentric_reconstruction = os.path.join(
                opensfm, "reconstruction.topocentric.json"
            )
            with open(topocentric_reconstruction, "w") as reconstruction_file:
                json.dump([], reconstruction_file)
            report_tree = SimpleNamespace(
                odm_report=os.path.join(report_root, "odm_report"),
                opensfm=opensfm,
                opensfm_reconstruction=os.path.join(opensfm, "reconstruction.json"),
                opensfm_topocentric_reconstruction=topocentric_reconstruction,
                odm_georeferencing_model_laz=public_laz,
                odm_georeferencing_alignment_matrix=os.path.join(directory, "missing-matrix.json"),
                odm_filterpoints=os.path.join(report_root, "odm_filterpoints"),
                odm_georeferencing=directory,
                odm_orthophoto_tif=os.path.join(directory, "missing-ortho.tif"),
                path=lambda *parts: os.path.join(report_root, *parts),
            )
            report_args = SimpleNamespace(
                skip_report=False, fast_orthophoto=False, crop=0, boundary=None,
                dsm=False, dtm=False,
                rerun=None, rerun_all=False, rerun_from=None,
            )
            report_stage = ODMReport("odm_report", report_args)
            point_stats = {"stats": {}, "num_points": 2}
            contract = resolve_coordinate_contract(
                self.anchor,
                self.controls([(0.0, 0.0, 0.0)], vertical_control=False),
            )
            with mock.patch(
                "stages.odm_report.get_geojson_shots_from_opensfm", return_value=None
            ) as export_shots, mock.patch(
                "stages.odm_report.generate_point_cloud_stats", return_value=point_stats
            ) as generate_stats, mock.patch(
                "stages.odm_report.gsd.opensfm_reconstruction_average_gsd", return_value=1.0
            ), mock.patch("stages.odm_report.OSFMContext.export_report"):
                report_stage.process(
                    report_args,
                    {
                        "tree": report_tree, "reconstruction": reconstruction,
                        "start_time": odm_system.now_raw(),
                        "coordinate_contract": contract,
                    },
                )
            self.assertEqual(generate_stats.call_args.args[0], public_laz)
            self.assertEqual(
                export_shots.call_args.args[0], topocentric_reconstruction
            )
            self.assertIs(
                export_shots.call_args.kwargs["coordinate_contract"], contract
            )
            with open(os.path.join(report_tree.odm_report, "stats.json")) as stats:
                report = json.load(stats)
            self.assertEqual(
                report["coordinate_contract"]["vertical_reference"], "unreferenced"
            )


if __name__ == "__main__":
    unittest.main()
