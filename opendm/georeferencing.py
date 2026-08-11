"""Exact, persisted coordinate contracts at ODX's public geometry boundary."""

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
import os
import tempfile
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from pyproj import CRS, Transformer


MANIFEST_SCHEMA = "odx-coordinate-contract-v1"


class GeoreferencingError(RuntimeError):
    """A georeferencing failure with enough context to diagnose the artifact."""

    def __init__(
        self,
        message: str,
        *,
        artifact: str = "coordinate contract",
        operation: str = "resolve",
        coordinate: Optional[Sequence[float]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        details = [artifact, operation, message]
        if coordinate is not None:
            details.append("coordinate sample={!r}".format(tuple(coordinate)))
        if cause is not None:
            details.append("{}: {}".format(type(cause).__name__, cause))
        super().__init__(": ".join(details))
        self.artifact = artifact
        self.operation = operation
        self.coordinate = tuple(coordinate) if coordinate is not None else None
        self.cause = cause


class CoordinateContractError(GeoreferencingError):
    pass


class ManifestError(GeoreferencingError):
    pass


class VerticalReference(str, Enum):
    WGS84_ELLIPSOIDAL = "wgs84_ellipsoidal"
    UNREFERENCED = "unreferenced"


@dataclass(frozen=True)
class TopocentricAnchor:
    """The WGS 84 geographic anchor for ODX's canonical ENU frame."""

    latitude: float
    longitude: float
    ellipsoidal_height: float

    def __post_init__(self) -> None:
        try:
            values = tuple(
                float(value)
                for value in (self.latitude, self.longitude, self.ellipsoidal_height)
            )
        except (TypeError, ValueError) as error:
            raise CoordinateContractError(
                "topocentric anchor must be numeric", cause=error
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise CoordinateContractError(
                "non-finite topocentric anchor", coordinate=values
            )
        if not -90.0 <= values[0] <= 90.0 or not -180.0 <= values[1] <= 180.0:
            raise CoordinateContractError(
                "invalid topocentric anchor", coordinate=values
            )
        object.__setattr__(self, "latitude", values[0])
        object.__setattr__(self, "longitude", values[1])
        object.__setattr__(self, "ellipsoidal_height", values[2])


@dataclass(frozen=True)
class CoordinateContract:
    """The immutable exact mapping from canonical ENU to public geometry."""

    anchor: TopocentricAnchor
    output_crs_wkt: str
    operation: str
    storage_offset: Tuple[float, float]
    vertical_reference: VerticalReference
    schema: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage_offset", _xy_offset(self.storage_offset))
        self.validate()

    @property
    def output_crs(self) -> CRS:
        try:
            return CRS.from_wkt(self.output_crs_wkt)
        except Exception as error:
            raise CoordinateContractError(
                "invalid persisted output CRS",
                operation="validate contract",
                cause=error,
            ) from error

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise CoordinateContractError("unsupported coordinate-contract schema")
        if not isinstance(self.anchor, TopocentricAnchor):
            raise CoordinateContractError("invalid topocentric anchor")
        if not isinstance(self.vertical_reference, VerticalReference):
            raise CoordinateContractError("invalid vertical-reference state")

        output_crs = self.output_crs
        expected_crs = _output_crs_for_anchor(self.anchor)
        if output_crs.to_epsg() != expected_crs.to_epsg():
            raise CoordinateContractError(
                "output CRS does not match the automatic anchor selection",
                operation="validate contract",
            )
        expected_operation = _resolved_operation(self.anchor, output_crs)
        if self.operation != expected_operation:
            raise CoordinateContractError(
                "operation does not match the persisted anchor and output CRS",
                operation="validate contract",
            )
        self._transformer()

    def transform_points(
        self,
        coordinates: Iterable[Sequence[float]],
        *,
        apply_storage_offset: bool = False,
    ) -> np.ndarray:
        """Transform canonical ENU points in one vectorized exact operation.

        Absolute formats leave ``apply_storage_offset`` false. Mesh formats use
        it to retain ODX's existing XY-only local storage convention.
        """
        points = _as_points(coordinates, "forward point transformation")
        self._validate_operation_area(points)
        try:
            transformed = _transform(self._transformer(), points)
        except GeoreferencingError:
            raise
        except Exception as error:
            raise CoordinateContractError(
                "coordinate operation failed",
                operation="forward point transformation",
                coordinate=points[0],
                cause=error,
            ) from error
        if self.vertical_reference == VerticalReference.UNREFERENCED:
            transformed[:, 2] = points[:, 2]
        if apply_storage_offset:
            transformed[:, :2] -= _xy_offset(self.storage_offset)
        if not np.all(np.isfinite(transformed)):
            raise CoordinateContractError(
                "non-finite transformation result",
                operation="forward point transformation",
                coordinate=points[0],
            )
        return transformed

    def persist(self, path: str) -> None:
        self.validate()
        directory = os.path.dirname(path) or "."
        temporary = None
        try:
            os.makedirs(directory, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".coordinate-contract-", dir=directory
            )
            with os.fdopen(descriptor, "w") as manifest:
                json.dump(_manifest_payload(self), manifest, indent=2, sort_keys=True)
                manifest.write("\n")
            os.replace(temporary, path)
            temporary = None
        except Exception as error:
            raise ManifestError(
                "could not persist coordinate contract",
                artifact=path,
                operation="persist manifest",
                cause=error,
            ) from error
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _transformer(self) -> Transformer:
        try:
            transformer = Transformer.from_pipeline(self.operation)
        except Exception as error:
            raise CoordinateContractError(
                "persisted coordinate operation is unavailable",
                operation="load operation",
                cause=error,
            ) from error
        if transformer.definition != self.operation:
            raise CoordinateContractError(
                "persisted coordinate operation did not reload deterministically",
                operation="load operation",
            )
        return transformer

    def _validate_operation_area(self, points: np.ndarray) -> None:
        area = self.output_crs.area_of_use
        if area is None:
            raise CoordinateContractError(
                "selected output CRS has no area of use",
                operation="validate transformation area",
            )
        try:
            geographic = _transform(
                Transformer.from_pipeline(
                    _topocentric_to_geographic_operation(self.anchor)
                ),
                points,
            )
        except Exception as error:
            raise CoordinateContractError(
                "could not validate transformation area",
                operation="validate transformation area",
                coordinate=points[0],
                cause=error,
            ) from error
        if not np.all(np.isfinite(geographic)):
            raise CoordinateContractError(
                "non-finite coordinate while validating transformation area",
                operation="validate transformation area",
                coordinate=points[0],
            )
        west, south, east, north = area.bounds
        longitude = geographic[:, 0]
        latitude = geographic[:, 1]
        longitude_ok = (
            (west <= longitude) & (longitude <= east)
            if west <= east
            else (longitude >= west) | (longitude <= east)
        )
        outside = np.flatnonzero(
            ~(longitude_ok & (south <= latitude) & (latitude <= north))
        )
        if outside.size:
            raise CoordinateContractError(
                "coordinate falls outside the selected output CRS area",
                operation="validate transformation area",
                coordinate=points[int(outside[0])],
            )


def resolve_coordinate_contract(
    anchor: TopocentricAnchor,
    *,
    storage_offset: Sequence[float] = (0.0, 0.0),
    vertical_reference: VerticalReference = VerticalReference.WGS84_ELLIPSOIDAL,
) -> CoordinateContract:
    """Resolve the one automatic UTM/UPS operation for an ordinary project."""
    if not isinstance(anchor, TopocentricAnchor):
        raise CoordinateContractError("topocentric anchor is required")
    if not isinstance(vertical_reference, VerticalReference):
        raise CoordinateContractError("invalid vertical-reference state")
    offset = _xy_offset(storage_offset)
    output_crs = _output_crs_for_anchor(anchor)
    contract = CoordinateContract(
        anchor=anchor,
        output_crs_wkt=output_crs.to_wkt(),
        operation=_resolved_operation(anchor, output_crs),
        storage_offset=offset,
        vertical_reference=vertical_reference,
    )
    return contract


def load_coordinate_contract(path: str) -> CoordinateContract:
    """Reload one validated contract; partial or legacy jobs must be rebuilt."""
    rerun = "rerun from reconstruction to create a compatible coordinate contract"
    try:
        with open(path) as manifest:
            payload = json.load(manifest)
        expected_fields = {
            "anchor",
            "operation",
            "output_crs_wkt",
            "schema",
            "storage_offset",
            "vertical_reference",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError("incompatible coordinate-contract manifest")
        contract = CoordinateContract(
            anchor=TopocentricAnchor(**payload["anchor"]),
            output_crs_wkt=payload["output_crs_wkt"],
            operation=payload["operation"],
            storage_offset=tuple(payload["storage_offset"]),
            vertical_reference=VerticalReference(payload["vertical_reference"]),
            schema=payload["schema"],
        )
        return contract
    except Exception as error:
        raise ManifestError(
            rerun, artifact=path, operation="load manifest", cause=error
        ) from error


def _as_points(coordinates: Iterable[Sequence[float]], operation: str) -> np.ndarray:
    try:
        points = np.asarray(coordinates, dtype=np.float64)
    except Exception as error:
        raise CoordinateContractError(
            "coordinates must be numeric", operation=operation, cause=error
        ) from error
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise CoordinateContractError(
            "coordinates must be a non-empty Nx3 array in ENU order",
            operation=operation,
        )
    bad = np.argwhere(~np.isfinite(points))
    if bad.size:
        raise CoordinateContractError(
            "non-finite transformation input",
            operation=operation,
            coordinate=points[int(bad[0, 0])],
        )
    return points


def _transform(transformer: Transformer, points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.asarray([transformer.transform(*points[0], errcheck=True)])
    return np.column_stack(transformer.transform(*points.T, errcheck=True))


def _xy_offset(values: Sequence[float]) -> Tuple[float, float]:
    try:
        offset = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise CoordinateContractError(
            "storage offset must be numeric", cause=error
        ) from error
    if len(offset) != 2 or not all(math.isfinite(value) for value in offset):
        raise CoordinateContractError("storage offset must be finite and XY-only")
    return offset


def _output_crs_for_anchor(anchor: TopocentricAnchor) -> CRS:
    if anchor.latitude > 84.0:
        return CRS.from_epsg(5041)
    if anchor.latitude < -80.0:
        return CRS.from_epsg(5042)
    zone = min(
        60, max(1, int(math.floor((anchor.longitude + 180.0) / 6.0)) + 1)
    )
    return CRS.from_epsg((32600 if anchor.latitude >= 0.0 else 32700) + zone)


def _topocentric_to_geographic_operation(anchor: TopocentricAnchor) -> str:
    return (
        "+proj=pipeline "
        "+step +inv +proj=topocentric +ellps=WGS84 "
        "+lat_0={:.15g} +lon_0={:.15g} +h_0={:.15g} "
        "+step +inv +proj=cart +ellps=WGS84"
    ).format(anchor.latitude, anchor.longitude, anchor.ellipsoidal_height)


def _resolved_operation(anchor: TopocentricAnchor, output_crs: CRS) -> str:
    epsg = output_crs.to_epsg()
    if 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760:
        projection = "+step +proj=utm +zone={} +ellps=WGS84".format(epsg % 100)
        if epsg >= 32701:
            projection += " +south"
    elif epsg in (5041, 5042):
        latitude = 90 if epsg == 5041 else -90
        projection = (
            "+step +proj=stere +lat_0={0} +lat_ts={0} +lon_0=0 "
            "+k=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84"
        ).format(latitude)
    else:
        raise CoordinateContractError("unsupported automatic output CRS")
    try:
        return Transformer.from_pipeline(
            _topocentric_to_geographic_operation(anchor) + " " + projection
        ).definition
    except Exception as error:
        raise CoordinateContractError(
            "could not resolve the exact coordinate operation", cause=error
        ) from error


def _manifest_payload(contract: CoordinateContract) -> dict:
    return {
        "schema": contract.schema,
        "anchor": asdict(contract.anchor),
        "output_crs_wkt": contract.output_crs_wkt,
        "operation": contract.operation,
        "storage_offset": list(contract.storage_offset),
        "vertical_reference": contract.vertical_reference.value,
    }
