import json
import os
import tempfile
import unittest

import numpy as np
from pyproj import CRS, Transformer

from opendm.georeferencing import (
    ManifestError,
    TopocentricAnchor,
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

    def test_reloads_or_rejects_the_persisted_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "coordinate_contract.json")
            contract = resolve_coordinate_contract(self.anchor)

            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)

            contract.persist(path)
            self.assertEqual(load_coordinate_contract(path), contract)

            with open(path) as manifest:
                payload = json.load(manifest)
            payload["operation"] = "invalid"
            with open(path, "w") as manifest:
                json.dump(payload, manifest)

            with self.assertRaisesRegex(ManifestError, "rerun from reconstruction"):
                load_coordinate_contract(path)


if __name__ == "__main__":
    unittest.main()
