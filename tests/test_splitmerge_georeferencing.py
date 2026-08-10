import os
import json
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from opendm.georeferencing import (
    CoordinateContractError,
    ManifestError,
    coordinate_contract_metadata,
    persist_split_merge_derivatives,
    NormalizedControlSet,
    SharedControlObservation,
    SharedOutputObservation,
    TopocentricAnchor,
    resolve_coordinate_contract,
    resolve_output_selection,
)

try:
    from opendm import types
    from stages.splitmerge import ODMMergeStage, ODMSplitStage
    from stages.odm_georeferencing import resolve_project_output_selection
except ImportError:
    types = ODMMergeStage = ODMSplitStage = resolve_project_output_selection = None


@unittest.skipIf(ODMMergeStage is None, "ODX split/merge dependencies are not installed")
class TestSplitMergeGeoreferencing(unittest.TestCase):
    def setUp(self):
        self.controls = NormalizedControlSet.from_coordinates(
            [(0.0, 0.0, 0.0)], vertical_control=True
        )

    def fixture(self, directory):
        tree = types.ODM_Tree(directory)
        os.makedirs(tree.odm_georeferencing, exist_ok=True)
        os.makedirs(tree.submodels_path, exist_ok=True)
        anchors = (
            TopocentricAnchor(-23.2067, -45.7650, 635.0),
            TopocentricAnchor(-23.2050, -45.7620, 642.0),
        )
        selection = resolve_output_selection(
            anchors[0], self.controls,
            storage_offset=(421000.0, 7433000.0),
        )
        selection.persist(tree.path(
            "odm_georeferencing", "output_selection.json"
        ))
        contracts = []
        point_clouds = []
        for index, anchor in enumerate(anchors):
            project = os.path.join(tree.submodels_path, "submodel_{:04d}".format(index))
            georeferencing = os.path.join(project, "odm_georeferencing")
            os.makedirs(georeferencing)
            selection.persist(os.path.join(georeferencing, "output_selection.json"))
            contract = resolve_coordinate_contract(
                anchor, self.controls, output_selection=selection
            )
            contract.persist(os.path.join(georeferencing, "coordinate_contract.json"))
            point_cloud = os.path.join(georeferencing, "odm_georeferenced_model.laz")
            open(point_cloud, "wb").close()
            report = os.path.join(project, "odm_report")
            os.makedirs(report)
            published_position = [421500.0 + index * 0.5, 7433500.0, 650.0]
            with open(os.path.join(report, "shots.geojson"), "w") as shots:
                json.dump({
                    "type": "FeatureCollection",
                    "coordinate_contract": coordinate_contract_metadata(contract),
                    "features": [{
                        "type": "Feature",
                        "properties": {
                            "filename": "shared.jpg",
                            "translation": published_position,
                        },
                        "geometry": {"type": "Point", "coordinates": [0.0, 0.0, 0.0]},
                    }],
                }, shots)
            contracts.append(contract)
            point_clouds.append(point_cloud)
            persist_split_merge_derivatives(
                project, contract,
                shared_control_observations=(SharedControlObservation(
                    "shared.jpg", -45.7635, -23.2058, 650.0
                ),),
                shared_output_observations=(SharedOutputObservation(
                    "shared.jpg",
                    tuple(contract.inverse_points([published_position])[0]),
                    tuple(published_position),
                ),),
            )
        args = SimpleNamespace(
            merge="pointcloud", dsm=False, dtm=False,
            rerun=None, rerun_all=False, rerun_from=None,
        )
        outputs = {
            "tree": tree, "large": True,
            "reconstruction": SimpleNamespace(),
        }
        return args, outputs, contracts, point_clouds

    def test_merge_accepts_distinct_anchors_and_uses_only_public_derivatives(self):
        with tempfile.TemporaryDirectory() as directory:
            args, outputs, _, point_clouds = self.fixture(directory)
            tree = outputs["tree"]

            def merge(inputs, output, rerun=False):
                self.assertEqual(inputs, point_clouds)
                self.assertTrue(all("odm_georeferenced_model.laz" in path for path in inputs))
                self.assertTrue(all("opensfm" not in path for path in inputs))
                open(output, "wb").close()

            stage = ODMMergeStage("merge", args)
            with mock.patch("stages.splitmerge.point_cloud.merge", side_effect=merge), mock.patch(
                "stages.splitmerge.point_cloud.post_point_cloud_steps"
            ), mock.patch(
                "stages.splitmerge.merge_cameras",
                side_effect=lambda _, output: open(output, "w").close(),
            ):
                stage.process(args, outputs)

            self.assertTrue(os.path.isfile(tree.odm_georeferencing_model_laz))
            with open(os.path.join(tree.odm_report, "shots.geojson")) as shots:
                self.assertEqual(
                    [feature["properties"]["filename"] for feature in json.load(shots)["features"]],
                    ["shared.jpg"],
                )

    def test_merge_rejects_incompatible_contract_before_artifact_work(self):
        with tempfile.TemporaryDirectory() as directory:
            args, outputs, contracts, _ = self.fixture(directory)
            manifest = os.path.join(
                outputs["tree"].submodels_path, "submodel_0001",
                "odm_georeferencing", "coordinate_contract.json",
            )
            replace(
                contracts[1], storage_offset=(1.0, 2.0, 0.0)
            ).persist(manifest)
            stage = ODMMergeStage("merge", args)

            with mock.patch("stages.splitmerge.point_cloud.merge") as merge:
                with self.assertRaisesRegex(CoordinateContractError, "storage offset"):
                    stage.process(args, outputs)
                merge.assert_not_called()

    def test_merge_rejects_a_submodel_without_published_derivative_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            args, outputs, contracts, _ = self.fixture(directory)
            project = os.path.join(
                outputs["tree"].submodels_path, "submodel_0002"
            )
            georeferencing = os.path.join(project, "odm_georeferencing")
            os.makedirs(georeferencing)
            contracts[0].persist(os.path.join(
                georeferencing, "coordinate_contract.json"
            ))
            stage = ODMMergeStage("merge", args)

            with mock.patch("stages.splitmerge.point_cloud.merge") as merge:
                with self.assertRaisesRegex(
                    ManifestError,
                    "rerun the submodel to publish compatible merge derivatives",
                ):
                    stage.process(args, outputs)
                merge.assert_not_called()

    def test_merge_rejects_disagreeing_published_overlap_geometry_before_artifact_work(self):
        with tempfile.TemporaryDirectory() as directory:
            args, outputs, contracts, _ = self.fixture(directory)
            shots_path = os.path.join(
                outputs["tree"].submodels_path, "submodel_0001",
                "odm_report", "shots.geojson",
            )
            with open(shots_path) as shots_file:
                shots = json.load(shots_file)
            shifted = shots["features"][0]["properties"]["translation"]
            shifted[0] += 2.0
            with open(shots_path, "w") as shots_file:
                json.dump(shots, shots_file)
            project = os.path.join(
                outputs["tree"].submodels_path, "submodel_0001"
            )
            persist_split_merge_derivatives(
                project, contracts[1],
                shared_output_observations=(SharedOutputObservation(
                    "shared.jpg",
                    tuple(contracts[1].inverse_points([shifted])[0]),
                    tuple(shifted),
                ),),
            )

            stage = ODMMergeStage("merge", args)
            with mock.patch("stages.splitmerge.point_cloud.merge") as merge:
                with self.assertRaisesRegex(
                    CoordinateContractError, "product tolerance"
                ):
                    stage.process(args, outputs)
                merge.assert_not_called()

    def test_parent_rerun_loads_persisted_selection_without_resolving_again(self):
        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            os.makedirs(tree.odm_georeferencing)
            selection = resolve_output_selection(
                TopocentricAnchor(-23.2067, -45.7650, 635.0), self.controls,
                storage_offset=(421000.0, 7433000.0),
            )
            selection.persist(tree.path(
                "odm_georeferencing", "output_selection.json"
            ))

            with mock.patch(
                "stages.odm_georeferencing.resolve_output_selection",
                side_effect=AssertionError("rerun must not re-resolve"),
            ) as resolve:
                loaded = resolve_project_output_selection(
                    tree, SimpleNamespace(), SimpleNamespace()
                )

            self.assertEqual(loaded, selection)
            resolve.assert_not_called()

    def test_local_split_marks_fresh_reconstruction_before_first_toolchain_dispatch(self):
        class Args(SimpleNamespace):
            def __contains__(self, key):
                return hasattr(self, key)

        with tempfile.TemporaryDirectory() as directory:
            tree = types.ODM_Tree(directory)
            os.makedirs(tree.opensfm)
            submodel_opensfm = os.path.join(
                tree.submodels_path, "submodel_0000", "opensfm"
            )
            args = Args(
                project_path=directory, split=1, split_overlap=50,
                sm_cluster=None, max_concurrency=2, rolling_shutter=False,
                rolling_shutter_readout=0, gps_accuracy=10, sm_no_align=True,
                sfm_algorithm="incremental", sfm_no_partial=False, primary_band=None,
                rerun=None, rerun_all=False, rerun_from=None,
            )
            reconstruction = SimpleNamespace(
                photos=[SimpleNamespace(filename="a.jpg"), SimpleNamespace(filename="b.jpg")],
                multi_camera=[], gcp=None,
                has_geotagged_photos=lambda: True,
            )

            class FakeOSFMContext:
                def __init__(self, path):
                    self.root = path

                def path(self, *parts):
                    return os.path.join(self.root, *parts)

                def setup(self, *args, **kwargs):
                    pass

                def photos_to_metadata(self, *args, **kwargs):
                    pass

                def feature_matching(self, *args, **kwargs):
                    pass

                def run(self, action):
                    os.makedirs(submodel_opensfm, exist_ok=True)

                def create_tracks(self, *args, **kwargs):
                    pass

                def reconstruct(self, *args, **kwargs):
                    pass

                def touch(self, path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    open(path, "a").close()

                def name(self):
                    return "submodel_0000"

            def dispatch(*args, **kwargs):
                self.assertTrue(os.path.isfile(os.path.join(
                    submodel_opensfm, "fresh_reconstruction.marker"
                )))

            metadata = SimpleNamespace(
                get_submodel_paths=lambda: [submodel_opensfm]
            )
            outputs = {"tree": tree, "reconstruction": reconstruction}
            with mock.patch(
                "stages.splitmerge.OSFMContext", FakeOSFMContext
            ), mock.patch(
                "stages.splitmerge.metadataset.MetaDataSet", return_value=metadata
            ), mock.patch(
                "stages.splitmerge.resolve_project_output_selection"
            ), mock.patch(
                "stages.splitmerge._propagate_output_selection"
            ), mock.patch(
                "stages.splitmerge.system.run", side_effect=dispatch
            ):
                ODMSplitStage("split", args).process(args, outputs)

            self.assertTrue(outputs["large"])


if __name__ == "__main__":
    unittest.main()
