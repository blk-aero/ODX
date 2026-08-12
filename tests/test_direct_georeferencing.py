import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from opendm import system, types
from opendm.georeferencing import (
    CoordinateContractError,
    TopocentricAnchor,
    export_georeferenced_point_cloud,
    export_georeferenced_mesh,
    load_coordinate_contract,
    materialize_canonical_reconstruction,
    preserve_canonical_reconstruction,
    publish_legacy_reconstruction_compatibility,
    resolve_coordinate_contract,
    resolve_stage_coordinate_contract,
)
from stages.odm_georeferencing import ODMGeoreferencingStage
from stages.odm_orthophoto import ODMOrthoPhotoStage


class GeoreferencedReconstruction:
    photos = [SimpleNamespace(band_name="RGB")]
    multi_camera = None

    def __init__(self, offset=(421000.0, 7433000.0)):
        self.offset = offset
        self.georef = SimpleNamespace(
            proj4=lambda: "+proj=utm +zone=23 +south +datum=WGS84 +units=m +no_defs"
        )

    @staticmethod
    def is_georeferenced():
        return True

    @staticmethod
    def has_gcp():
        return False

    def get_proj_offset(self):
        return self.offset


def stage_args(**overrides):
    values = {
        "auto_boundary": False,
        "boundary": None,
        "crop": 0,
        "fast_orthophoto": False,
        "max_concurrency": 1,
        "optimize_disk_space": False,
        "pc_classify": False,
        "pc_copc": False,
        "pc_csv": False,
        "pc_ept": False,
        "pc_las": False,
        "primary_band": None,
        "rerun": None,
        "rerun_all": True,
        "rerun_from": None,
        "use_exif": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def orthophoto_args(**overrides):
    values = {
        "boundary": None,
        "build_overviews": False,
        "cog": False,
        "crop": 0,
        "ignore_gsd": False,
        "max_concurrency": 1,
        "optimize_disk_space": False,
        "orthophoto_compression": "DEFLATE",
        "orthophoto_cutline": False,
        "orthophoto_kmz": False,
        "orthophoto_no_tiled": False,
        "orthophoto_png": False,
        "orthophoto_resolution": 5.0,
        "primary_band": None,
        "rerun": None,
        "rerun_all": True,
        "rerun_from": None,
        "skip_orthophoto": False,
        "tiles": False,
        "use_3dmesh": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def write_reference(tree):
    os.makedirs(tree.opensfm, exist_ok=True)
    with open(tree.path("opensfm", "reference_lla.json"), "w") as reference:
        json.dump(
            {
                "latitude": -23.206694678,
                "longitude": -45.764977891,
                "altitude": 0.0,
            },
            reference,
        )


def write_triangle(path):
    with open(path, "w") as mesh:
        mesh.write(
            "mtllib mesh.mtl\n"
            "v 0 0 0\n"
            "v 100 0 0\n"
            "v 0 20 1\n"
            "vt 0 0\nvt 1 0\nvt 0 1\n"
            "usemtl atlas\n"
            "f 1/1 2/2 3/3\n"
        )


class TestDirectGeoreferencingStage(unittest.TestCase):
    def test_rejects_unsupported_direct_combinations_before_publication(self):
        combinations = ("align", "boundary", "auto_boundary", "submodel")
        for combination in combinations:
            with self.subTest(combination=combination), tempfile.TemporaryDirectory() as directory:
                root = (
                    os.path.join(directory, "submodel_0000")
                    if combination == "submodel"
                    else directory
                )
                tree = types.ODM_Tree(root)
                os.makedirs(tree.odm_georeferencing, exist_ok=True)
                args = stage_args(auto_boundary=combination == "auto_boundary")
                outputs = {
                    "tree": tree,
                    "reconstruction": GeoreferencedReconstruction(),
                }
                if combination == "align":
                    tree.odm_align_file = os.path.join(root, "align.laz")
                if combination in ("boundary", "auto_boundary"):
                    outputs["boundary"] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

                with self.assertRaises(system.ExitException) as raised:
                    ODMGeoreferencingStage("odm_georeferencing", args).process(
                        args, outputs
                    )
                expected_message = {
                    "align": "--align",
                    "boundary": "--boundary",
                    "auto_boundary": "--auto-boundary",
                    "submodel": "split submodels",
                }[combination]
                self.assertIn(expected_message, str(raised.exception))

                self.assertFalse(
                    os.path.exists(
                        tree.path("odm_georeferencing", "coordinate_contract.json")
                    )
                )
                self.assertFalse(os.path.exists(tree.odm_georeferencing_model_laz))

    def test_publishes_exact_geometry_before_the_stock_compatibility_view(self):
        try:
            import pdal
        except ImportError:
            self.skipTest("PDAL Python bindings are not installed")

        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            for path in (
                tree.odm_filterpoints,
                tree.odm_georeferencing,
                tree.odm_texturing,
                tree.odm_25dtexturing,
            ):
                os.makedirs(path, exist_ok=True)
            write_reference(tree)

            canonical_payload = [{"frame": "topocentric"}]
            compatibility_payload = [{"frame": "stock-affine"}]
            for path in (
                tree.opensfm_reconstruction,
                tree.opensfm_topocentric_reconstruction,
            ):
                with open(path, "w") as reconstruction_file:
                    json.dump(canonical_payload, reconstruction_file)

            with open(tree.filtered_point_cloud, "w") as point_cloud:
                point_cloud.write(
                    "ply\nformat ascii 1.0\nelement vertex 2\n"
                    "property double x\nproperty double y\nproperty double z\n"
                    "property ushort intensity\nproperty uchar views\nend_header\n"
                    "0 0 0 7 4\n100 50 10 9 2\n"
                )

            public_mesh = os.path.join(
                tree.odm_texturing, tree.odm_textured_model_obj
            )
            canonical_mesh = os.path.splitext(public_mesh)[0] + "_topocentric.obj"
            write_triangle(public_mesh)
            write_triangle(canonical_mesh)

            args = stage_args()
            outputs = {
                "tree": tree,
                "reconstruction": GeoreferencedReconstruction(),
                "fresh_reconstruction": True,
            }

            def stock_export(command, **_kwargs):
                self.assertIn("export_geocoords --reconstruction", command)
                self.assertTrue(os.path.isfile(tree.odm_georeferencing_model_laz))
                with open(public_mesh) as mesh:
                    self.assertNotEqual(mesh.readlines()[1], "v 0 0 0\n")
                with open(tree.opensfm_reconstruction) as reconstruction_file:
                    self.assertEqual(json.load(reconstruction_file), canonical_payload)
                with open(tree.opensfm_geocoords_reconstruction, "w") as generated:
                    json.dump(compatibility_payload, generated)

            with mock.patch("opendm.system.run", side_effect=stock_export):
                ODMGeoreferencingStage("odm_georeferencing", args).process(
                    args, outputs
                )

            contract = load_coordinate_contract(
                tree.path("odm_georeferencing", "coordinate_contract.json")
            )
            with open(public_mesh) as mesh:
                vertices = np.asarray(
                    [
                        [float(value) for value in line.split()[1:4]]
                        for line in mesh
                        if line.startswith("v ")
                    ]
                )
            np.testing.assert_allclose(
                vertices,
                contract.transform_points(
                    [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 20.0, 1.0]],
                    apply_storage_offset=True,
                ),
                atol=0.0001,
                rtol=0.0,
            )
            with open(tree.opensfm_reconstruction) as reconstruction_file:
                self.assertEqual(json.load(reconstruction_file), compatibility_payload)
            with open(tree.opensfm_topocentric_reconstruction) as reconstruction_file:
                self.assertEqual(json.load(reconstruction_file), canonical_payload)


class TestDirectExportAdapters(unittest.TestCase):
    def test_direct_boundary_creates_then_reloads_the_same_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = os.path.join(directory, "reference_lla.json")
            manifest = os.path.join(directory, "coordinate_contract.json")
            with open(reference, "w") as stream:
                json.dump(
                    {
                        "latitude": -23.206694678,
                        "longitude": -45.764977891,
                        "altitude": 0.0,
                    },
                    stream,
                )

            created = resolve_stage_coordinate_contract(
                manifest,
                reference,
                (421000.0, 7433000.0),
                fresh_reconstruction=True,
            )
            reloaded = resolve_stage_coordinate_contract(
                manifest, reference, (421000.0, 7433000.0)
            )

            self.assertEqual(reloaded, created)
            self.assertEqual(load_coordinate_contract(manifest), created)

    def test_point_cloud_streams_exact_public_coordinates(self):
        contract = resolve_coordinate_contract(
            TopocentricAnchor(-23.206694678, -45.764977891, 0.0)
        )
        dtype = [
            ("X", "f8"),
            ("Y", "f8"),
            ("Z", "f8"),
            ("Intensity", "u2"),
        ]
        source_batches = [
            np.asarray([(0.0, 0.0, 0.0, 7)], dtype=dtype),
            np.asarray([(100.0, 50.0, 10.0, 9)], dtype=dtype),
        ]

        class Pipeline:
            read_passes = 0
            written = []

            def __init__(self, specification, arrays=(), stream_handlers=()):
                self.specification = json.loads(specification)
                self.arrays = arrays
                self.stream_handlers = stream_handlers
                self.streamable = True

            def iterator(self, chunk_size):
                self.__class__.read_passes += 1
                return iter(batch.copy() for batch in source_batches)

            def execute_streaming(self, chunk_size):
                buffer = self.arrays[0]
                load = self.stream_handlers[0]
                count = 0
                while True:
                    size = load()
                    if not size:
                        break
                    self.__class__.written.append(buffer[:size].copy())
                    count += size
                open(self.specification[0]["filename"], "wb").close()
                return count

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "odm_georeferenced_model.laz")
            result = export_georeferenced_point_cloud(
                "canonical.ply",
                output,
                contract,
                spacing=0.02,
                chunk_size=1,
                pdal_module=SimpleNamespace(Pipeline=Pipeline),
            )

        written = np.concatenate(Pipeline.written)
        self.assertEqual(Pipeline.read_passes, 2)
        self.assertEqual(result.point_count, 2)
        np.testing.assert_allclose(
            np.column_stack((written["X"], written["Y"], written["Z"])),
            contract.transform_points([[0.0, 0.0, 0.0], [100.0, 50.0, 10.0]]),
            atol=0.0001,
            rtol=0.0,
        )

    def test_textured_mesh_preserves_visual_structure_and_xy_local_storage(self):
        contract = resolve_coordinate_contract(
            TopocentricAnchor(-23.206694678, -45.764977891, 0.0),
            storage_offset=(421000.0, 7433000.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "canonical.obj")
            output = os.path.join(directory, "public.obj")
            material = os.path.join(directory, "mesh.mtl")
            texture = os.path.join(directory, "texture.jpg")
            write_triangle(source)
            with open(material, "wb") as stream:
                stream.write(b"newmtl atlas\nmap_Kd texture.jpg\n")
            with open(texture, "wb") as stream:
                stream.write(b"texture-bytes")

            export_georeferenced_mesh(source, output, contract)

            with open(output) as mesh:
                lines = mesh.readlines()
            vertices = np.asarray(
                [
                    [float(value) for value in line.split()[1:4]]
                    for line in lines
                    if line.startswith("v ")
                ]
            )
            np.testing.assert_allclose(
                vertices,
                contract.transform_points(
                    [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 20.0, 1.0]],
                    apply_storage_offset=True,
                ),
                atol=0.0001,
                rtol=0.0,
            )
            self.assertIn("mtllib mesh.mtl\n", lines)
            self.assertIn("vt 1 0\n", lines)
            self.assertIn("usemtl atlas\n", lines)
            self.assertIn("f 1/1 2/2 3/3\n", lines)
            with open(material, "rb") as stream:
                self.assertEqual(stream.read(), b"newmtl atlas\nmap_Kd texture.jpg\n")
            with open(texture, "rb") as stream:
                self.assertEqual(stream.read(), b"texture-bytes")

    def test_compatibility_publication_always_starts_from_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            active = os.path.join(directory, "reconstruction.json")
            canonical = os.path.join(
                directory, "reconstruction.topocentric.json"
            )
            generated = os.path.join(directory, "reconstruction.geocoords.json")
            canonical_payload = [{"frame": "topocentric"}]
            compatibility_payload = [{"frame": "stock-affine"}]
            with open(active, "w") as reconstruction:
                json.dump(canonical_payload, reconstruction)

            preserve_canonical_reconstruction(active, canonical)
            with open(active, "w") as reconstruction:
                json.dump(compatibility_payload, reconstruction)
            materialize_canonical_reconstruction(active, canonical)

            inputs = []

            def stock_export():
                with open(active) as reconstruction:
                    inputs.append(json.load(reconstruction))
                with open(generated, "w") as reconstruction:
                    json.dump(compatibility_payload, reconstruction)

            for _ in range(2):
                publish_legacy_reconstruction_compatibility(
                    active, canonical, generated, stock_export
                )

            self.assertEqual(inputs, [canonical_payload, canonical_payload])
            with open(active) as reconstruction:
                self.assertEqual(json.load(reconstruction), compatibility_payload)
            with open(canonical) as reconstruction:
                self.assertEqual(json.load(reconstruction), canonical_payload)


class TestDirectOrthophotoStage(unittest.TestCase):
    def fixture(self, directory):
        tree = types.ODM_Tree(directory)
        for path in (
            tree.opensfm,
            tree.odm_georeferencing,
            tree.odm_orthophoto,
            tree.odm_texturing,
        ):
            os.makedirs(path, exist_ok=True)
        write_reference(tree)
        with open(tree.opensfm_topocentric_reconstruction, "w") as reconstruction:
            json.dump([{"frame": "topocentric"}], reconstruction)
        contract = resolve_coordinate_contract(
            TopocentricAnchor(-23.206694678, -45.764977891, 0.0),
            storage_offset=(421000.0, 7433000.0),
        )
        return tree, contract

    def test_fails_before_rendering_without_a_persisted_contract_or_valid_mesh(self):
        cases = (
            "missing_contract",
            "missing_mesh",
            "empty_mesh",
            "mismatched_contract",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                tree, contract = self.fixture(directory)
                public_mesh = os.path.join(
                    tree.odm_texturing, tree.odm_textured_model_obj
                )
                if case == "empty_mesh":
                    with open(public_mesh, "w") as mesh:
                        mesh.write("# empty\n")
                elif case != "missing_mesh":
                    write_triangle(public_mesh)
                if case != "missing_contract":
                    persisted = (
                        resolve_coordinate_contract(
                            TopocentricAnchor(-23.206694678, -45.764977891, 0.0),
                            storage_offset=(421100.0, 7433000.0),
                        )
                        if case == "mismatched_contract"
                        else contract
                    )
                    persisted.persist(
                        tree.path(
                            "odm_georeferencing", "coordinate_contract.json"
                        )
                    )

                args = orthophoto_args()
                outputs = {
                    "tree": tree,
                    "reconstruction": GeoreferencedReconstruction(),
                    "coordinate_contract": contract,
                    "large": False,
                }
                with mock.patch(
                    "stages.odm_orthophoto.gsd.cap_resolution", return_value=5.0
                ), mock.patch("stages.odm_orthophoto.system.run") as renderer:
                    with self.assertRaises(CoordinateContractError):
                        ODMOrthoPhotoStage("odm_orthophoto", args).process(
                            args, outputs
                        )
                    renderer.assert_not_called()

    def test_direct_render_uses_only_the_public_mesh_and_persisted_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, contract = self.fixture(directory)
            public_mesh = os.path.join(
                tree.odm_texturing, tree.odm_textured_model_obj
            )
            write_triangle(public_mesh)
            contract.persist(
                tree.path("odm_georeferencing", "coordinate_contract.json")
            )
            args = orthophoto_args()
            outputs = {
                "tree": tree,
                "reconstruction": GeoreferencedReconstruction(),
                "coordinate_contract": contract,
                "large": False,
            }
            commands = []

            def render(command, **_kwargs):
                commands.append(command)
                open(tree.odm_orthophoto_tif, "wb").close()

            with mock.patch(
                "stages.odm_orthophoto.gsd.cap_resolution", return_value=5.0
            ) as resolution, mock.patch(
                "stages.odm_orthophoto.system.run", side_effect=render
            ), mock.patch(
                "stages.odm_orthophoto.orthophoto.post_orthophoto_steps"
            ):
                ODMOrthoPhotoStage("odm_orthophoto", args).process(args, outputs)

            self.assertEqual(
                resolution.call_args.args[1],
                tree.opensfm_topocentric_reconstruction,
            )
            self.assertIn(public_mesh, commands[0])
            self.assertIn("-utm_north_offset 7433000.0", commands[0])
            self.assertIn("-utm_east_offset 421000.0", commands[0])
            self.assertIn("-a_srs", commands[0])


if __name__ == "__main__":
    unittest.main()
