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
    export_georeferenced_mesh,
    export_georeferenced_point_cloud,
    resolve_coordinate_contract,
)
from stages.odm_georeferencing import ODMGeoreferencingStage
from stages.odm_orthophoto import ODMOrthoPhotoStage


class GeoreferencedReconstruction:
    photos = [SimpleNamespace(band_name="RGB")]
    multi_camera = None

    @staticmethod
    def is_georeferenced():
        return True

    @staticmethod
    def has_gcp():
        return False


def stage_args(**overrides):
    values = {"auto_boundary": False, "boundary": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def orthophoto_args(**overrides):
    values = {
        "ignore_gsd": False,
        "max_concurrency": 1,
        "orthophoto_compression": "DEFLATE",
        "orthophoto_cutline": False,
        "orthophoto_no_tiled": False,
        "orthophoto_resolution": 5.0,
        "primary_band": None,
        "skip_orthophoto": False,
        "use_3dmesh": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def write_point_cloud(path):
    with open(path, "w") as point_cloud:
        point_cloud.write(
            "ply\nformat ascii 1.0\nelement vertex 2\n"
            "property double x\nproperty double y\nproperty double z\n"
            "property ushort intensity\nproperty uchar views\nend_header\n"
            "0 0 0 7 4\n100 50 10 9 2\n"
        )


class TestDirectGeoreferencingStage(unittest.TestCase):
    def test_rejects_unsupported_direct_combinations_before_artifacts(self):
        cases = (
            ("align", "--align"),
            ("boundary", "--boundary"),
            ("auto_boundary", "--auto-boundary"),
            ("submodel", "split submodels"),
        )
        for case, message in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = (
                    os.path.join(directory, "submodel_0000")
                    if case == "submodel"
                    else directory
                )
                tree = types.ODM_Tree(root)
                args = stage_args(auto_boundary=case == "auto_boundary")
                outputs = {
                    "tree": tree,
                    "reconstruction": GeoreferencedReconstruction(),
                }
                if case == "align":
                    tree.odm_align_file = os.path.join(root, "align.laz")
                if case == "boundary":
                    outputs["boundary"] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

                with self.assertRaisesRegex(system.ExitException, message):
                    ODMGeoreferencingStage("odm_georeferencing", args).process(
                        args, outputs
                    )

                self.assertFalse(
                    os.path.exists(
                        tree.path("odm_georeferencing", "coordinate_contract.json")
                    )
                )
                self.assertFalse(os.path.exists(tree.odm_georeferencing_model_laz))


class TestDirectExports(unittest.TestCase):
    anchor = TopocentricAnchor(-23.206694678, -45.764977891, 0.0)

    def test_writes_exact_public_laz_with_pdal(self):
        try:
            import pdal
        except ImportError:
            self.skipTest("requires the project container's PDAL Python bindings")

        contract = resolve_coordinate_contract(self.anchor)
        points = [[0.0, 0.0, 0.0], [100.0, 50.0, 10.0]]
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "canonical.ply")
            output = os.path.join(directory, "odm_georeferenced_model.laz")
            write_point_cloud(source)

            result = export_georeferenced_point_cloud(
                source, output, contract, spacing=0.02, chunk_size=1
            )
            reader = pdal.Pipeline(json.dumps([output]))
            reader.execute()
            exported = np.concatenate(reader.arrays)

        self.assertEqual(result.point_count, len(points))
        np.testing.assert_allclose(
            np.column_stack((exported["X"], exported["Y"], exported["Z"])),
            contract.transform_points(points),
            atol=0.0006,
            rtol=0.0,
        )

    def test_writes_xy_local_mesh_without_changing_visual_structure(self):
        contract = resolve_coordinate_contract(
            self.anchor, storage_offset=(421000.0, 7433000.0)
        )
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "canonical.obj")
            output = os.path.join(directory, "public.obj")
            write_triangle(source)
            with open(source) as mesh:
                source_lines = mesh.readlines()

            result = export_georeferenced_mesh(source, output, contract)
            with open(output) as mesh:
                lines = mesh.readlines()

        vertices = np.asarray(
            [
                [float(value) for value in line.split()[1:4]]
                for line in lines
                if line.startswith("v ")
            ]
        )
        self.assertEqual((result.vertex_count, result.face_count), (3, 1))
        np.testing.assert_allclose(
            vertices,
            contract.transform_points(
                [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 20.0, 1.0]],
                apply_storage_offset=True,
            ),
            atol=0.0001,
            rtol=0.0,
        )
        self.assertEqual(
            [line for line in lines if not line.startswith("v ")],
            [line for line in source_lines if not line.startswith("v ")],
        )


class TestDirectOrthophotoStage(unittest.TestCase):
    anchor = TopocentricAnchor(-23.206694678, -45.764977891, 0.0)

    def fixture(self, directory):
        tree = types.ODM_Tree(directory)
        for path in (
            tree.opensfm,
            tree.odm_georeferencing,
            tree.odm_texturing,
        ):
            os.makedirs(path, exist_ok=True)
        with open(tree.opensfm_topocentric_reconstruction, "w") as reconstruction:
            json.dump([{}], reconstruction)
        return tree, resolve_coordinate_contract(
            self.anchor, storage_offset=(421000.0, 7433000.0)
        )

    def test_rejects_invalid_contract_or_mesh_before_renderer(self):
        cases = (
            "missing_contract",
            "invalid_contract",
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
                        mesh.write("# no geometry\n")
                elif case != "missing_mesh":
                    write_triangle(public_mesh)

                contract_path = tree.path(
                    "odm_georeferencing", "coordinate_contract.json"
                )
                if case == "invalid_contract":
                    with open(contract_path, "w") as manifest:
                        json.dump({}, manifest)
                elif case != "missing_contract":
                    persisted = (
                        resolve_coordinate_contract(
                            self.anchor, storage_offset=(421100.0, 7433000.0)
                        )
                        if case == "mismatched_contract"
                        else contract
                    )
                    persisted.persist(contract_path)

                outputs = {
                    "tree": tree,
                    "reconstruction": GeoreferencedReconstruction(),
                    "coordinate_contract": contract,
                    "large": False,
                }
                args = orthophoto_args()
                with mock.patch(
                    "stages.odm_orthophoto.gsd.cap_resolution", return_value=5.0
                ), mock.patch("stages.odm_orthophoto.system.run") as renderer:
                    with self.assertRaises(CoordinateContractError):
                        ODMOrthoPhotoStage("odm_orthophoto", args).process(
                            args, outputs
                        )
                    renderer.assert_not_called()

    def test_renders_with_the_persisted_contract_and_public_mesh(self):
        with tempfile.TemporaryDirectory() as directory:
            tree, contract = self.fixture(directory)
            public_mesh = os.path.join(
                tree.odm_texturing, tree.odm_textured_model_obj
            )
            write_triangle(public_mesh)
            contract.persist(
                tree.path("odm_georeferencing", "coordinate_contract.json")
            )
            outputs = {
                "tree": tree,
                "reconstruction": GeoreferencedReconstruction(),
                "coordinate_contract": contract,
                "large": False,
            }
            args = orthophoto_args()

            with mock.patch(
                "stages.odm_orthophoto.gsd.cap_resolution", return_value=5.0
            ) as resolution, mock.patch(
                "stages.odm_orthophoto.system.run"
            ) as renderer, mock.patch(
                "stages.odm_orthophoto.orthophoto.post_orthophoto_steps"
            ):
                ODMOrthoPhotoStage("odm_orthophoto", args).process(args, outputs)

        self.assertEqual(
            resolution.call_args.args[1], tree.opensfm_topocentric_reconstruction
        )
        renderer.assert_called_once()
        command = renderer.call_args.args[0]
        self.assertIn(public_mesh, command)
        self.assertIn("-utm_north_offset 7433000.0", command)
        self.assertIn("-utm_east_offset 421000.0", command)
        self.assertIn("-a_srs", command)


if __name__ == "__main__":
    unittest.main()
