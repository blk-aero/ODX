import json
import os
import tempfile
import unittest

import numpy as np
from pyproj import CRS, Transformer

from opendm.georeferencing import (
    CoordinateContractError,
    ManifestError,
    TopocentricAnchor,
    VerticalReference,
    load_coordinate_contract,
    resolve_coordinate_contract,
)


class TestCoordinateContract(unittest.TestCase):
    def setUp(self):
        self.anchor = TopocentricAnchor(-23.206694678, -45.764977891, 635.5485)

    def test_transforms_points_with_the_exact_persisted_operation(self):
        contract = resolve_coordinate_contract(
            self.anchor, storage_offset=(421000.0, 7433000.0)
        )
        points = np.asarray([[0.0, 0.0, 0.0], [100000.0, -40000.0, 250.0]])

        actual = contract.transform_points(points)
        topocentric_to_geographic = Transformer.from_pipeline(
            "+proj=pipeline "
            "+step +inv +proj=topocentric +ellps=WGS84 "
            "+lat_0=-23.206694678 +lon_0=-45.764977891 +h_0=635.5485 "
            "+step +inv +proj=cart +ellps=WGS84"
        )
        geographic = topocentric_to_geographic.transform(*points.T)
        expected = np.column_stack(
            Transformer.from_crs(
                CRS.from_epsg(4979), contract.output_crs, always_xy=True
            ).transform(*geographic)
        )

        np.testing.assert_allclose(actual, expected, atol=0.0001, rtol=0.0)
        stored = contract.transform_points(points, apply_storage_offset=True)
        np.testing.assert_allclose(
            stored,
            actual - [421000.0, 7433000.0, 0.0],
            atol=0.0,
            rtol=0.0,
        )

    def test_selects_utm_or_ups_from_the_anchor(self):
        self.assertEqual(
            resolve_coordinate_contract(self.anchor).output_crs.to_epsg(), 32723
        )
        self.assertEqual(
            resolve_coordinate_contract(TopocentricAnchor(85.0, 10.0, 12.0))
            .output_crs.to_epsg(),
            5041,
        )
        self.assertEqual(
            resolve_coordinate_contract(TopocentricAnchor(-81.0, 10.0, 12.0))
            .output_crs.to_epsg(),
            5042,
        )

    def test_persists_and_reloads_the_immutable_contract(self):
        contract = resolve_coordinate_contract(
            self.anchor, vertical_reference=VerticalReference.UNREFERENCED
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "coordinate_contract.json")
            contract.persist(path)

            loaded = load_coordinate_contract(path)

        self.assertEqual(loaded, contract)
        self.assertEqual(loaded.vertical_reference, VerticalReference.UNREFERENCED)
        self.assertEqual(
            loaded.transform_points([(1000.0, 2000.0, 25.0)])[0, 2], 25.0
        )

    def test_missing_or_changed_manifest_requires_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "coordinate_contract.json")
            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)

            resolve_coordinate_contract(self.anchor).persist(path)
            with open(path) as manifest:
                payload = json.load(manifest)
            payload["operation"] = "invalid"
            with open(path, "w") as manifest:
                json.dump(payload, manifest)

            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)

    def test_rejects_invalid_storage_and_coordinate_values(self):
        with self.assertRaises(CoordinateContractError):
            resolve_coordinate_contract(self.anchor, storage_offset=(1.0, 2.0, 3.0))

        contract = resolve_coordinate_contract(self.anchor)
        with self.assertRaises(CoordinateContractError):
            contract.transform_points([(float("nan"), 0.0, 0.0)])
        with self.assertRaises(CoordinateContractError):
            contract.transform_points([(700000.0, 0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
