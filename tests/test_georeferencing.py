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
    anchor = TopocentricAnchor(-23.206694678, -45.764977891, 635.5485)

    def test_transforms_a_non_affine_extent_exactly(self):
        contract = resolve_coordinate_contract(self.anchor)
        points = np.asarray([[0.0, 0.0, 0.0], [100000.0, -40000.0, 250.0]])

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

        np.testing.assert_allclose(
            contract.transform_points(points), expected, atol=0.0001, rtol=0.0
        )
        self.assertEqual(contract.output_crs.to_epsg(), 32723)
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
        for point in ((float("nan"), 0.0, 0.0), (700000.0, 0.0, 0.0)):
            with self.subTest(point=point):
                with self.assertRaises(CoordinateContractError):
                    contract.transform_points([point])

    def test_reloads_or_rejects_the_persisted_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "coordinate_contract.json")
            contract = resolve_coordinate_contract(
                self.anchor, vertical_reference=VerticalReference.UNREFERENCED
            )

            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)

            contract.persist(path)
            loaded = load_coordinate_contract(path)
            self.assertEqual(loaded, contract)
            self.assertEqual(loaded.vertical_reference, VerticalReference.UNREFERENCED)
            self.assertEqual(
                loaded.transform_points([(1000.0, 2000.0, 25.0)])[0, 2], 25.0
            )

            with open(path) as manifest:
                payload = json.load(manifest)
            payload["operation"] = "invalid"
            with open(path, "w") as manifest:
                json.dump(payload, manifest)

            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)


if __name__ == "__main__":
    unittest.main()
