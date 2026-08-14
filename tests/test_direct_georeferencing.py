import json
import os
import shlex
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from opendm import types
from opendm.georeferencing import (
    TopocentricAnchor,
    VerticalReference,
    export_georeferenced_mesh,
    export_georeferenced_point_cloud,
    load_coordinate_contract,
    resolve_coordinate_contract,
)
from stages.odm_georeferencing import ODMGeoreferencingStage, _vertical_reference
from stages.odm_orthophoto import ODMOrthoPhotoStage


class GeoreferencedReconstruction:
    multi_camera = None

    def __init__(self, photos=None, offset=(421000.0, 7433000.0), gcp=None):
        self.photos = photos or [SimpleNamespace(band_name="RGB", altitude=100.0)]
        self.offset = offset
        self.gcp = gcp
        self.georef = SimpleNamespace(
            utm_offset=lambda: offset,
            utm_east_offset=offset[0],
            utm_north_offset=offset[1],
            proj4=lambda: "EPSG:32723",
        )

    @staticmethod
    def is_georeferenced():
        return True

    def has_gcp(self):
        return self.gcp is not None and self.gcp.exists()

    def get_proj_offset(self):
        return self.offset

    @staticmethod
    def get_proj_srs():
        return "EPSG:32723"


def stage_args(**overrides):
    values = {
        "auto_boundary": False,
        "boundary": None,
        "crop": 0,
        "fast_orthophoto": False,
        "force_gps": False,
        "optimize_disk_space": False,
        "primary_band": None,
        "rerun": None,
        "rerun_all": True,
        "rerun_from": None,
        "skip_3dmodel": False,
        "skip_orthophoto": False,
        "use_3dmesh": True,
    }
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


def write_triangle_with_normals(path):
    with open(path, "w") as mesh:
        mesh.write(
            "mtllib mesh.mtl\n"
            "v 0 0 0\n"
            "v 0 100 0\n"
            "v 0 0 50\n"
            "vt 0 0\nvt 1 0\nvt 0 1\n"
            "vn 1 0 0\n"
            "usemtl atlas\n"
            "f 1/1/1 2/2/1 3/3/1\n"
        )


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


def prepare_stage_files(tree, canonical_mesh=True):
    for path in (
        tree.odm_filterpoints,
        tree.odm_georeferencing,
        tree.odm_texturing,
    ):
        os.makedirs(path, exist_ok=True)
    write_reference(tree)
    for path in (
        tree.opensfm_reconstruction,
        tree.opensfm_topocentric_reconstruction,
    ):
        with open(path, "w") as reconstruction:
            json.dump([{"frame": "topocentric"}], reconstruction)
    public_mesh = os.path.join(tree.odm_texturing, tree.odm_textured_model_obj)
    if canonical_mesh:
        write_triangle(os.path.splitext(public_mesh)[0] + "_topocentric.obj")
    return public_mesh


def fake_point_export(_source, output, _contract, **_kwargs):
    open(output, "wb").close()
    return SimpleNamespace(point_count=1)


def write_point_cloud(path):
    with open(path, "w") as point_cloud:
        point_cloud.write(
            "ply\nformat ascii 1.0\nelement vertex 2\n"
            "property double x\nproperty double y\nproperty double z\n"
            "property ushort intensity\nproperty uchar views\nend_header\n"
            "0 0 0 7 4\n100 50 10 9 2\n"
        )


class TestDirectGeoreferencingStage(unittest.TestCase):
    def test_uses_coordinate_contract_with_options(self):
        cases = ("align", "boundary", "auto_boundary", "submodel")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = os.path.join(directory, "submodel_0000") if case == "submodel" else directory
                tree = types.ODM_Tree(root)
                public_mesh = prepare_stage_files(tree)
                write_triangle(public_mesh)
                args = stage_args(
                    auto_boundary=case == "auto_boundary",
                    boundary="boundary.json" if case == "boundary" else None,
                )
                outputs = {
                    "tree": tree,
                    "reconstruction": GeoreferencedReconstruction(),
                    "fresh_reconstruction": True,
                }
                if case == "align":
                    tree.odm_align_file = os.path.join(root, "align.laz")
                if case == "boundary":
                    outputs["boundary"] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

                with mock.patch(
                    "stages.odm_georeferencing.export_georeferenced_point_cloud",
                    side_effect=fake_point_export,
                ), mock.patch(
                    "stages.odm_georeferencing.point_cloud.post_point_cloud_steps"
                ), mock.patch(
                    "stages.odm_georeferencing.publish_legacy_reconstruction_compatibility"
                ), mock.patch(
                    "stages.odm_georeferencing.compute_alignment_matrix", return_value=None
                ), mock.patch(
                    "stages.odm_georeferencing.export_to_bounds_files"
                ), mock.patch(
                    "stages.odm_georeferencing.system.run"
                ):
                    ODMGeoreferencingStage("odm_georeferencing", args).process(
                        args, outputs
                    )

                self.assertTrue(os.path.exists(public_mesh))
                self.assertIn("coordinate_contract", outputs)

    def test_mesh_failure_clears_contract_before_stock_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            public_mesh = prepare_stage_files(tree)
            write_triangle(public_mesh)
            args = stage_args()
            outputs = {
                "tree": tree,
                "reconstruction": GeoreferencedReconstruction(),
                "fresh_reconstruction": True,
                "stock_georeferenced_reconstruction": True,
            }

            with mock.patch(
                "stages.odm_georeferencing.export_georeferenced_point_cloud",
                side_effect=fake_point_export,
            ), mock.patch(
                "stages.odm_georeferencing.export_georeferenced_mesh",
                side_effect=RuntimeError("invalid canonical mesh"),
            ), mock.patch(
                "stages.odm_georeferencing.point_cloud.post_point_cloud_steps"
            ), mock.patch(
                "stages.odm_georeferencing.publish_legacy_reconstruction_compatibility"
            ) as publish_compatibility, mock.patch(
                "stages.odm_georeferencing.system.run"
            ):
                ODMGeoreferencingStage("odm_georeferencing", args).process(
                    args, outputs
                )

        self.assertNotIn("coordinate_contract", outputs)
        publish_compatibility.assert_not_called()

    def test_quotes_stock_gcp_vlr_json(self):
        class GCP:
            def exists(self):
                return True

            @staticmethod
            def iter_entries():
                return iter(())

        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            prepare_stage_files(tree)
            zip_path = tree.path(
                "odm_georeferencing", "ground_control_points.zip"
            )
            with open(zip_path, "wb") as vlr:
                vlr.write(b"gcp")
            args = stage_args(skip_3dmodel=True)
            outputs = {
                "tree": tree,
                "reconstruction": GeoreferencedReconstruction(gcp=GCP()),
                "fresh_reconstruction": True,
            }

            with mock.patch(
                "stages.odm_georeferencing.OSFMContext.ground_control_points",
                return_value=[],
            ), mock.patch(
                "stages.odm_georeferencing.export_georeferenced_point_cloud",
                side_effect=RuntimeError("PDAL unavailable"),
            ), mock.patch(
                "stages.odm_georeferencing.point_cloud.post_point_cloud_steps"
            ), mock.patch(
                "stages.odm_georeferencing.publish_legacy_reconstruction_compatibility"
            ), mock.patch(
                "stages.odm_georeferencing.system.run"
            ) as run:
                ODMGeoreferencingStage("odm_georeferencing", args).process(
                    args, outputs
                )

        command = run.call_args.args[0]
        vlr_argument = next(
            argument
            for argument in shlex.split(command)
            if argument.startswith("--writers.las.vlrs=")
        )
        vlr = json.loads(vlr_argument.split("=", 1)[1])
        self.assertIsInstance(vlr, dict)
        self.assertEqual(vlr["filename"], zip_path.replace(os.sep, "/"))

    def test_selects_vertical_state_and_preserves_relative_z(self):
        class GCP:
            def exists(self):
                return True

            @staticmethod
            def iter_entries():
                return iter(
                    [
                        SimpleNamespace(
                            z=120.0, is_checkpoint=lambda: False
                        )
                    ]
                )

        args = stage_args()
        self.assertEqual(
            _vertical_reference(
                GeoreferencedReconstruction(
                    photos=[SimpleNamespace(band_name="RGB", altitude=100.0)]
                ),
                args,
            ),
            VerticalReference.WGS84_ELLIPSOIDAL,
        )
        self.assertEqual(
            _vertical_reference(
                GeoreferencedReconstruction(
                    photos=[SimpleNamespace(band_name="RGB", altitude=100.0)],
                    gcp=GCP(),
                ),
                stage_args(force_gps=True),
            ),
            VerticalReference.WGS84_ELLIPSOIDAL,
        )
        self.assertEqual(
            _vertical_reference(
                GeoreferencedReconstruction(
                    photos=[SimpleNamespace(band_name="RGB", altitude=None)],
                    gcp=GCP(),
                ),
                args,
            ),
            VerticalReference.WGS84_ELLIPSOIDAL,
        )

        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            prepare_stage_files(tree)
            reconstruction = GeoreferencedReconstruction(
                photos=[SimpleNamespace(band_name="RGB", altitude=None)]
            )
            args = stage_args()
            outputs = {
                "tree": tree,
                "reconstruction": reconstruction,
                "fresh_reconstruction": True,
            }

            with mock.patch(
                "stages.odm_georeferencing.export_georeferenced_point_cloud",
                side_effect=fake_point_export,
            ), mock.patch(
                "stages.odm_georeferencing.point_cloud.post_point_cloud_steps"
            ), mock.patch(
                "stages.odm_georeferencing.publish_legacy_reconstruction_compatibility"
            ):
                ODMGeoreferencingStage("odm_georeferencing", args).process(
                    args, outputs
                )

            contract = load_coordinate_contract(
                tree.path("odm_georeferencing", "coordinate_contract.json")
            )
            self.assertEqual(
                contract.vertical_reference, VerticalReference.UNREFERENCED
            )
            self.assertEqual(
                contract.transform_points([(10.0, 20.0, 37.5)])[0, 2], 37.5
            )

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

    def test_writes_xy_local_mesh_with_valid_normals_and_visual_structure(self):
        contract = resolve_coordinate_contract(
            self.anchor, storage_offset=(421000.0, 7433000.0)
        )
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "canonical.obj")
            output = os.path.join(directory, "public.obj")
            write_triangle_with_normals(source)
            with open(os.path.join(directory, "mesh.mtl"), "w") as material:
                material.write("newmtl atlas\nmap_Kd texture.jpg\n")
            with open(os.path.join(directory, "texture.jpg"), "wb") as texture:
                texture.write(b"texture-bytes")

            result = export_georeferenced_mesh(source, output, contract)
            with open(output) as mesh:
                lines = mesh.readlines()
            with open(os.path.join(directory, "mesh.mtl")) as material:
                material_text = material.read()
            with open(os.path.join(directory, "texture.jpg"), "rb") as texture:
                texture_bytes = texture.read()

        vertices = np.asarray(
            [
                [float(value) for value in line.split()[1:4]]
                for line in lines
                if line.startswith("v ")
            ]
        )
        normals = np.asarray(
            [
                [float(value) for value in line.split()[1:4]]
                for line in lines
                if line.startswith("vn ")
            ]
        )
        self.assertEqual((result.vertex_count, result.face_count), (3, 1))
        np.testing.assert_allclose(
            vertices,
            contract.transform_points(
                [[0.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 50.0]],
                apply_storage_offset=True,
            ),
            atol=0.0001,
            rtol=0.0,
        )
        self.assertEqual(len(normals), 3)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
        source_vertices = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 50.0]]
        )
        for point, normal in zip(source_vertices, normals):
            for tangent in ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
                delta = np.asarray(tangent) * 0.01
                transformed_tangent = (
                    contract.transform_points([point + delta])[0]
                    - contract.transform_points([point - delta])[0]
                )
                transformed_tangent /= np.linalg.norm(transformed_tangent)
                self.assertLess(
                    abs(float(np.dot(normal, transformed_tangent))), 1e-6
                )
        face = next(line for line in lines if line.startswith("f ")).split()[1:]
        self.assertEqual(
            ["/".join(token.split("/")[:2]) for token in face],
            ["1/1", "2/2", "3/3"],
        )
        self.assertEqual([token.split("/")[2] for token in face], ["1", "2", "3"])
        for expected in (
            "mtllib mesh.mtl\n",
            "vt 0 0\n",
            "vt 1 0\n",
            "vt 0 1\n",
            "usemtl atlas\n",
        ):
            self.assertIn(expected, lines)
        self.assertEqual(material_text, "newmtl atlas\nmap_Kd texture.jpg\n")
        self.assertEqual(texture_bytes, b"texture-bytes")


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
