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


class PointCloudExportError(GeoreferencingError):
    pass


class MeshExportError(GeoreferencingError):
    pass


class ReconstructionError(GeoreferencingError):
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


@dataclass(frozen=True)
class PointCloudExportResult:
    point_count: int
    scale: Tuple[float, float, float]
    offset: Tuple[float, float, float]


@dataclass(frozen=True)
class MeshExportResult:
    vertex_count: int
    face_count: int


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


def load_topocentric_anchor(path: str) -> TopocentricAnchor:
    try:
        with open(path) as reference_file:
            reference = json.load(reference_file)
        return TopocentricAnchor(
            reference["latitude"], reference["longitude"], reference["altitude"]
        )
    except Exception as error:
        raise CoordinateContractError(
            "could not read the persisted topocentric anchor",
            artifact=path,
            operation="resolve stage contract",
            cause=error,
        ) from error


def resolve_stage_coordinate_contract(
    manifest_path: str,
    reference_path: str,
    storage_offset: Sequence[float],
    *,
    fresh_reconstruction: bool = False,
    vertical_reference: VerticalReference,
) -> CoordinateContract:
    """Create a fresh contract or reload the exact persisted contract."""
    expected = resolve_coordinate_contract(
        load_topocentric_anchor(reference_path),
        storage_offset=storage_offset,
        vertical_reference=vertical_reference,
    )
    if os.path.exists(manifest_path):
        try:
            persisted = load_coordinate_contract(manifest_path)
            if persisted == expected:
                return persisted
            if not fresh_reconstruction:
                raise ManifestError(
                    "coordinate contract does not match reconstructed project metadata; "
                    "rerun from reconstruction",
                    artifact=manifest_path,
                    operation="validate project contract",
                )
        except ManifestError:
            if not fresh_reconstruction:
                raise
    elif not fresh_reconstruction:
        raise ManifestError(
            "coordinate contract is missing; rerun from reconstruction",
            artifact=manifest_path,
            operation="stage reload",
        )
    expected.persist(manifest_path)
    return expected


PUBLIC_LAZ_DIMENSIONS = (
    "X",
    "Y",
    "Z",
    "Intensity",
    "ReturnNumber",
    "NumberOfReturns",
    "ScanDirectionFlag",
    "EdgeOfFlightLine",
    "Classification",
    "Synthetic",
    "KeyPoint",
    "Withheld",
    "Overlap",
    "ScanAngleRank",
    "UserData",
    "PointSourceId",
    "GpsTime",
    "Red",
    "Green",
    "Blue",
)


def _public_point_cloud_batch(
    source: np.ndarray, contract: CoordinateContract
) -> np.ndarray:
    names = source.dtype.names or ()
    if not {"X", "Y", "Z"}.issubset(names):
        raise PointCloudExportError(
            "point dimensions X, Y, and Z are required",
            artifact="georeferenced point cloud",
            operation="transform point batch",
        )
    output_names = tuple(name for name in PUBLIC_LAZ_DIMENSIONS if name in names)
    output = np.empty(
        source.shape,
        dtype=[
            (name, np.float64 if name in ("X", "Y", "Z") else source.dtype[name])
            for name in output_names
        ],
    )
    for name in output_names:
        output[name] = source[name]
    transformed = contract.transform_points(
        np.column_stack((source["X"], source["Y"], source["Z"]))
    )
    for index, name in enumerate(("X", "Y", "Z")):
        output[name] = transformed[:, index]
    return output


def _las_encoding(
    lower: np.ndarray, upper: np.ndarray, spacing: Optional[float]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if spacing is None:
        spacing = 0.01
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise PointCloudExportError(
            "point spacing must be finite and positive",
            artifact="georeferenced point cloud",
            operation="select LAS encoding",
        )
    scale_value = min(pow(10.0, round(math.log10(spacing))) / 10.0, 0.001)
    scale = np.full(3, scale_value)
    offset = (lower + upper) / 2.0
    encoded = np.vstack(((lower - offset) / scale, (upper - offset) / scale))
    int32 = np.iinfo(np.int32)
    if np.any(encoded < int32.min) or np.any(encoded > int32.max):
        raise PointCloudExportError(
            "selected LAS scale cannot encode coordinates as signed 32-bit integers",
            artifact="georeferenced point cloud",
            operation="select LAS encoding",
            coordinate=lower,
        )
    return tuple(scale), tuple(offset)


def export_georeferenced_point_cloud(
    source_path: str,
    output_path: str,
    contract: CoordinateContract,
    *,
    spacing: Optional[float] = None,
    chunk_size: int = 250000,
    vlrs: Optional[Sequence[dict]] = None,
) -> PointCloudExportResult:
    """Stream canonical points through the contract and publish the stock LAZ."""
    if chunk_size <= 0:
        raise PointCloudExportError(
            "chunk size must be positive",
            artifact=output_path,
            operation="stream point cloud",
        )
    try:
        import pdal
    except Exception as error:
        raise PointCloudExportError(
            "PDAL Python bindings are unavailable",
            artifact=output_path,
            operation="stream point cloud",
            cause=error,
        ) from error

    reader_spec = json.dumps(
        [source_path, {"type": "filters.ferry", "dimensions": "views=>UserData"}]
    )

    def batches():
        pipeline = pdal.Pipeline(reader_spec)
        if not pipeline.streamable:
            raise PointCloudExportError(
                "canonical point-cloud reader is not streamable",
                artifact=output_path,
                operation="stream point cloud",
            )
        return pipeline.iterator(chunk_size=chunk_size)

    point_count = 0
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    output_dtype = None
    temporary = None
    try:
        for source in batches():
            if len(source) == 0:
                continue
            transformed = _public_point_cloud_batch(source, contract)
            coordinates = np.column_stack(
                (transformed["X"], transformed["Y"], transformed["Z"])
            )
            lower = np.minimum(lower, np.min(coordinates, axis=0))
            upper = np.maximum(upper, np.max(coordinates, axis=0))
            point_count += len(transformed)
            if output_dtype is None:
                output_dtype = transformed.dtype
            elif transformed.dtype != output_dtype:
                raise PointCloudExportError(
                    "point dimensions changed between streamed batches",
                    artifact=output_path,
                    operation="scan point cloud",
                )
        if point_count == 0:
            raise PointCloudExportError(
                "canonical point cloud contains no points",
                artifact=output_path,
                operation="scan point cloud",
            )

        scale, offset = _las_encoding(lower, upper, spacing)
        output_directory = os.path.dirname(output_path) or "."
        os.makedirs(output_directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=os.path.basename(output_path) + ".",
            suffix=".tmp.laz",
            dir=output_directory,
        )
        os.close(descriptor)
        os.unlink(temporary)
        writer = {
            "type": "writers.las",
            "filename": temporary,
            "compression": "laszip",
            "a_srs": contract.output_crs_wkt,
            "minor_version": 2,
            "dataformat_id": 3,
            "scale_x": scale[0],
            "scale_y": scale[1],
            "scale_z": scale[2],
            "offset_x": offset[0],
            "offset_y": offset[1],
            "offset_z": offset[2],
        }
        if vlrs is not None:
            writer["vlrs"] = vlrs
        source_batches = iter(batches())
        buffer = np.empty(chunk_size, dtype=output_dtype)
        written_count = 0

        def load_next_batch():
            nonlocal written_count
            for source in source_batches:
                if len(source):
                    transformed = _public_point_cloud_batch(source, contract)
                    if transformed.dtype != output_dtype or len(transformed) > len(buffer):
                        raise PointCloudExportError(
                            "streamed point layout changed during export",
                            artifact=output_path,
                            operation="write LAZ",
                        )
                    buffer[: len(transformed)] = transformed
                    written_count += len(transformed)
                    return len(transformed)
            return 0

        pipeline = pdal.Pipeline(
            json.dumps([writer]), arrays=[buffer], stream_handlers=[load_next_batch]
        )
        if not pipeline.streamable:
            raise PointCloudExportError(
                "LAZ writer is not streamable",
                artifact=output_path,
                operation="write LAZ",
            )
        reported_count = pipeline.execute_streaming(chunk_size=chunk_size)
        if written_count != point_count or reported_count != point_count:
            raise PointCloudExportError(
                "point count changed during LAZ serialization",
                artifact=output_path,
                operation="write LAZ",
            )
        os.replace(temporary, output_path)
        temporary = None
        return PointCloudExportResult(point_count, scale, offset)
    except GeoreferencingError:
        raise
    except Exception as error:
        raise PointCloudExportError(
            "point-cloud export failed",
            artifact=output_path,
            operation="write exact LAZ",
            cause=error,
        ) from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _obj_index(value: str, count: int, kind: str) -> int:
    try:
        index = int(value)
    except ValueError as error:
        raise MeshExportError(
            "invalid OBJ {} index".format(kind),
            artifact="georeferenced mesh",
            operation="parse OBJ",
            cause=error,
        ) from error
    resolved = index - 1 if index > 0 else count + index
    if index == 0 or resolved < 0 or resolved >= count:
        raise MeshExportError(
            "OBJ {} index is out of range".format(kind),
            artifact="georeferenced mesh",
            operation="parse OBJ",
        )
    return resolved


def _transform_normals(
    contract: CoordinateContract,
    coordinates: Iterable[Sequence[float]],
    normals: Iterable[Sequence[float]],
) -> np.ndarray:
    points = _as_points(coordinates, "normal transformation")
    vectors = _as_points(normals, "normal transformation")
    if len(points) != len(vectors):
        raise CoordinateContractError(
            "point and normal counts differ", operation="normal transformation"
        )
    epsilon = 0.001
    offsets = np.eye(3) * epsilon
    plus = np.concatenate([points + offset for offset in offsets])
    minus = np.concatenate([points - offset for offset in offsets])
    differences = (
        contract.transform_points(plus) - contract.transform_points(minus)
    ) / (2.0 * epsilon)
    count = len(points)
    jacobians = np.stack(
        [differences[index * count : (index + 1) * count] for index in range(3)],
        axis=2,
    )
    try:
        transformed = np.linalg.solve(
            np.swapaxes(jacobians, 1, 2), vectors[..., np.newaxis]
        )[..., 0]
    except Exception as error:
        raise CoordinateContractError(
            "normal transformation failed",
            operation="normal transformation",
            coordinate=points[0],
            cause=error,
        ) from error
    lengths = np.linalg.norm(transformed, axis=1)
    if not np.all(np.isfinite(transformed)) or np.any(lengths == 0.0):
        raise CoordinateContractError(
            "invalid transformed normal",
            operation="normal transformation",
            coordinate=points[0],
        )
    return transformed / lengths[:, None]


def export_georeferenced_mesh(
    source_path: str, output_path: str, contract: CoordinateContract
) -> MeshExportResult:
    """Atomically transform OBJ vertices while preserving its visual structure."""
    temporary = None
    try:
        with open(source_path) as source:
            lines = source.readlines()
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
        if normals.size == 0:
            normals = np.empty((0, 3), dtype=np.float64)
        face_count = sum(line.startswith("f ") for line in lines)
        if len(vertices) == 0 or face_count == 0:
            raise MeshExportError(
                "canonical mesh must contain vertices and faces",
                artifact=output_path,
                operation="transform OBJ",
            )
        transformed = contract.transform_points(
            vertices, apply_storage_offset=True
        )

        normal_pairs = {}
        parsed_faces = {}
        for line_index, line in enumerate(lines):
            if not line.startswith("f "):
                continue
            parsed = []
            for token in line.split()[1:]:
                fields = token.split("/")
                vertex_index = _obj_index(fields[0], len(vertices), "vertex")
                normal_index = None
                if len(fields) >= 3 and fields[2]:
                    normal_index = _obj_index(fields[2], len(normals), "normal")
                    pair = (normal_index, vertex_index)
                    if pair not in normal_pairs:
                        normal_pairs[pair] = len(normal_pairs) + 1
                parsed.append((fields, vertex_index, normal_index))
            parsed_faces[line_index] = parsed

        pairs = list(normal_pairs)
        if pairs:
            transformed_normals = _transform_normals(
                contract,
                vertices[[vertex for _, vertex in pairs]],
                normals[[normal for normal, _ in pairs]],
            )
        else:
            transformed_normals = np.empty((0, 3), dtype=np.float64)

        rewritten_faces = {}
        for line_index, parsed in parsed_faces.items():
            if not any(normal_index is not None for _, _, normal_index in parsed):
                continue
            tokens = []
            for fields, vertex_index, normal_index in parsed:
                if normal_index is not None:
                    fields[2] = str(normal_pairs[(normal_index, vertex_index)])
                tokens.append("/".join(fields))
            rewritten_faces[line_index] = "f " + " ".join(tokens) + "\n"

        output_directory = os.path.dirname(output_path) or "."
        os.makedirs(output_directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=os.path.basename(output_path) + ".",
            suffix=".tmp.obj",
            dir=output_directory,
        )
        with os.fdopen(descriptor, "w") as output:
            vertex_index = 0
            emitted_normals = False
            for line_index, line in enumerate(lines):
                if line.startswith("v "):
                    output.write(
                        "v {:.17g} {:.17g} {:.17g}\n".format(
                            *transformed[vertex_index]
                        )
                    )
                    vertex_index += 1
                elif line.startswith("vn "):
                    if not emitted_normals:
                        for normal in transformed_normals:
                            output.write(
                                "vn {:.17g} {:.17g} {:.17g}\n".format(*normal)
                            )
                        emitted_normals = True
                elif line_index in rewritten_faces:
                    output.write(rewritten_faces[line_index])
                else:
                    output.write(line)
        os.replace(temporary, output_path)
        temporary = None
        return MeshExportResult(len(vertices), face_count)
    except GeoreferencingError:
        raise
    except Exception as error:
        raise MeshExportError(
            "mesh export failed",
            artifact=output_path,
            operation="write exact OBJ",
            cause=error,
        ) from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def validate_canonical_reconstruction(path: str) -> None:
    try:
        with open(path) as reconstruction_file:
            reconstruction = json.load(reconstruction_file)
        if not isinstance(reconstruction, list) or not reconstruction:
            raise ValueError("OpenSfM reconstruction must be a nonempty list")
    except Exception as error:
        raise ReconstructionError(
            "canonical topocentric reconstruction is missing or incompatible; "
            "rerun from reconstruction",
            artifact=path,
            operation="validate canonical reconstruction",
            cause=error,
        ) from error


def _atomic_copy(source_path: str, output_path: str) -> None:
    temporary = None
    try:
        output_directory = os.path.dirname(output_path) or "."
        os.makedirs(output_directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=os.path.basename(output_path) + ".", dir=output_directory
        )
        with open(source_path, "rb") as source, os.fdopen(descriptor, "wb") as output:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        os.replace(temporary, output_path)
        temporary = None
    except Exception as error:
        raise ReconstructionError(
            "canonical reconstruction file operation failed",
            artifact=output_path,
            operation="copy canonical reconstruction",
            cause=error,
        ) from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def preserve_canonical_reconstruction(active_path: str, canonical_path: str) -> None:
    validate_canonical_reconstruction(active_path)
    _atomic_copy(active_path, canonical_path)


def materialize_canonical_reconstruction(active_path: str, canonical_path: str) -> None:
    validate_canonical_reconstruction(canonical_path)
    _atomic_copy(canonical_path, active_path)


def publish_legacy_reconstruction_compatibility(
    active_path: str,
    canonical_path: str,
    generated_path: str,
    stock_export,
) -> None:
    """Late-publish stock OpenSfM's affine view from canonical input only."""
    materialize_canonical_reconstruction(active_path, canonical_path)
    try:
        os.unlink(generated_path)
    except FileNotFoundError:
        pass
    try:
        stock_export()
        try:
            with open(generated_path) as reconstruction_file:
                generated = json.load(reconstruction_file)
            if not isinstance(generated, list) or not generated:
                raise ValueError("OpenSfM reconstruction must be a nonempty list")
        except Exception as validation_error:
            raise ReconstructionError(
                "stock OpenSfM did not produce a compatible reconstruction",
                artifact=generated_path,
                operation="publish legacy reconstruction",
                cause=validation_error,
            ) from validation_error
        os.replace(generated_path, active_path)
    except Exception as error:
        try:
            os.unlink(generated_path)
        except FileNotFoundError:
            pass
        if isinstance(error, GeoreferencingError):
            raise
        raise ReconstructionError(
            "stock OpenSfM compatibility export failed",
            artifact=generated_path,
            operation="publish legacy reconstruction",
            cause=error,
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
