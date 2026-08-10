"""Exact, persisted coordinate contracts for ODX export boundaries."""

from dataclasses import asdict, dataclass, replace
from enum import Enum
import glob
import json
import math
import os
import platform
import tempfile
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pyproj
from pyproj import CRS, Transformer
from pyproj.enums import TransformDirection
from pyproj.transformer import TransformerGroup


MANIFEST_SCHEMA = "odx-coordinate-contract-v1"
OUTPUT_SELECTION_SCHEMA = "odx-output-selection-v1"
SPLIT_MERGE_DERIVATIVES_SCHEMA = "odx-split-merge-derivatives-v1"
WGS84_GEOGRAPHIC_3D = CRS.from_epsg(4979)
WGS84_ECEF = CRS.from_epsg(4978)


def _persist_json_manifest(path: str, payload: dict, description: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w") as manifest:
            json.dump(payload, manifest, indent=2, sort_keys=True)
            manifest.write("\n")
        os.replace(temporary, path)
    except Exception as error:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise ManifestError(
            "could not persist {}".format(description), artifact=path, cause=error
        ) from error


def _load_json_manifest(path: str, message: str, operation: str, decode):
    try:
        with open(path) as manifest:
            return decode(json.load(manifest))
    except Exception as error:
        raise ManifestError(
            message, artifact=path, operation=operation, cause=error
        ) from error


class GeoreferencingError(RuntimeError):
    """A georeferencing failure with export diagnostic context."""

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


class PointCloudExportError(CoordinateContractError):
    pass


class MeshExportError(CoordinateContractError):
    pass


class RasterExportError(CoordinateContractError):
    pass


class LegacyCompatibilityError(CoordinateContractError):
    pass


class VerticalReference(str, Enum):
    WGS84_ELLIPSOIDAL = "wgs84_ellipsoidal"
    UNREFERENCED = "unreferenced"


class ControlSource(str, Enum):
    GCP = "gcp"
    GPS = "gps"


@dataclass(frozen=True)
class ControlProvenance:
    source: ControlSource
    input_crs_wkt: str
    operation: str
    grids: Tuple[str, ...]
    accuracy: float
    axis_order: Tuple[str, str, str]
    height_assumptions: Tuple[str, ...]
    warnings: Tuple[str, ...]
    versions: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class NormalizedControlSet:
    coordinates: Tuple[Tuple[float, float, float], ...]
    vertical_control: bool
    provenance: Tuple[ControlProvenance, ...] = ()

    @classmethod
    def from_coordinates(
        cls,
        coordinates: Iterable[Sequence[float]],
        *,
        vertical_control: bool = True,
        provenance: Sequence[ControlProvenance] = (),
    ):
        points = _as_points(coordinates, "normalized control set")
        return cls(
            tuple(tuple(float(value) for value in point) for point in points),
            bool(vertical_control),
            tuple(provenance),
        )

    @property
    def height_assumptions(self) -> Tuple[str, ...]:
        assumptions = tuple(
            assumption
            for item in self.provenance
            for assumption in item.height_assumptions
        )
        if assumptions:
            return assumptions
        if not self.vertical_control:
            return ("relative Z retained without a height datum claim",)
        return ()


@dataclass(frozen=True)
class TopocentricAnchor:
    latitude: float
    longitude: float
    ellipsoidal_height: float

    def __post_init__(self) -> None:
        values = (self.latitude, self.longitude, self.ellipsoidal_height)
        if not all(math.isfinite(value) for value in values):
            raise CoordinateContractError("non-finite topocentric anchor", coordinate=values)
        if not -90.0 <= self.latitude <= 90.0:
            raise CoordinateContractError("invalid anchor latitude", coordinate=values)
        if not -180.0 <= self.longitude <= 180.0:
            raise CoordinateContractError("invalid anchor longitude", coordinate=values)


@dataclass(frozen=True)
class SharedControlObservation:
    identifier: str
    longitude: float
    latitude: float
    height: float

    def __post_init__(self) -> None:
        if not self.identifier:
            raise CoordinateContractError("control observation identifier is empty")
        values = (self.longitude, self.latitude, self.height)
        if not all(math.isfinite(value) for value in values):
            raise CoordinateContractError(
                "non-finite shared control observation", coordinate=values
            )
        if not -180.0 <= self.longitude <= 180.0 or not -90.0 <= self.latitude <= 90.0:
            raise CoordinateContractError(
                "invalid shared control observation", coordinate=values
            )


@dataclass(frozen=True)
class SharedOutputObservation:
    """One published overlap position and its canonical topocentric source."""

    identifier: str
    topocentric: Tuple[float, float, float]
    output: Tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.identifier:
            raise CoordinateContractError("output observation identifier is empty")
        for label, coordinates in (
            ("topocentric", self.topocentric), ("output", self.output)
        ):
            if len(coordinates) != 3 or not all(
                math.isfinite(value) for value in coordinates
            ):
                raise CoordinateContractError(
                    "invalid shared {} observation".format(label),
                    coordinate=coordinates,
                )


@dataclass(frozen=True)
class OutputSelection:
    """Anchor-independent output choices shared by split submodels."""

    output_crs_wkt: str
    storage_offset: Tuple[float, float, float]
    vertical_reference: VerticalReference
    geometry_unit: str
    height_assumptions: Tuple[str, ...]
    warnings: Tuple[str, ...]
    schema: str = OUTPUT_SELECTION_SCHEMA

    @property
    def output_crs(self) -> CRS:
        return CRS.from_wkt(self.output_crs_wkt)

    def validate(self) -> None:
        if self.schema != OUTPUT_SELECTION_SCHEMA:
            raise ValueError("unsupported output-selection schema")
        _validate_static_crs(self.output_crs)
        if not self.output_crs.is_projected:
            raise ValueError("output selection must use a projected CRS")
        if self.geometry_unit != self.output_crs.axis_info[0].unit_name:
            raise ValueError("output unit does not match the output CRS")
        if self.vertical_reference not in tuple(VerticalReference):
            raise ValueError("invalid vertical-reference status")
        if len(self.storage_offset) != 3 or self.storage_offset[2] != 0.0:
            raise ValueError("invalid XY-only storage offset")
        if not all(math.isfinite(value) for value in self.storage_offset):
            raise ValueError("non-finite storage offset")

    def to_dict(self) -> dict:
        return asdict(self)

    def persist(self, path: str) -> None:
        self.validate()
        _persist_json_manifest(path, self.to_dict(), "output selection")


@dataclass(frozen=True)
class LasEncoding:
    """Explicit LAS integer encoding which decodes to absolute coordinates."""

    scale: Tuple[float, float, float]
    offset: Tuple[float, float, float]


@dataclass(frozen=True)
class PointCloudExportResult:
    point_count: int
    dimensions: Tuple[str, ...]
    encoding: LasEncoding
    bounds: Optional["DecodedCoordinateBounds"] = None


@dataclass(frozen=True)
class MeshExportResult:
    vertex_count: int
    normal_count: int
    face_count: int


@dataclass(frozen=True)
class RasterExportResult:
    source_bounds: Tuple[float, float, float, float]
    maximum_transformation_error: float


@dataclass(frozen=True)
class DecodedCoordinateBounds:
    minimum: np.ndarray
    maximum: np.ndarray


@dataclass(frozen=True)
class CameraPoseExport:
    centres: np.ndarray
    world_to_camera_rotations: np.ndarray


def transform_camera_poses(coordinates, world_to_camera_rotations, contract) -> CameraPoseExport:
    """Transform OpenSfM camera centres and rigid poses into local output frames."""
    centres = _as_points(coordinates, "camera centre transformation")
    try:
        rotations = np.asarray(world_to_camera_rotations, dtype=np.float64)
    except Exception as error:
        raise CoordinateContractError(
            "camera rotations are not numeric",
            operation="camera pose transformation", cause=error,
        ) from error
    if rotations.shape != (len(centres), 3, 3):
        raise CoordinateContractError(
            "camera rotations must be Nx3x3",
            operation="camera pose transformation",
        )
    if not np.all(np.isfinite(rotations)):
        raise CoordinateContractError(
            "non-finite camera rotation", operation="camera pose transformation",
            coordinate=rotations[0].flat[:3],
        )
    camera_to_world = np.swapaxes(rotations, 1, 2)
    transformed_frames = contract.transform_camera_frames(centres, camera_to_world)
    return CameraPoseExport(
        contract.transform_points(centres, apply_storage_offset=False),
        np.swapaxes(transformed_frames, 1, 2),
    )


def decoded_coordinate_bounds(coordinates, *, tolerance: float = 0.0) -> DecodedCoordinateBounds:
    """Return tight bounds from every decoded coordinate, including representation tolerance."""
    points = _as_points(coordinates, "decoded artifact bounds")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise CoordinateContractError(
            "bounds tolerance must be finite and non-negative",
            operation="derive decoded artifact bounds",
        )
    return DecodedCoordinateBounds(
        np.min(points, axis=0) - tolerance,
        np.max(points, axis=0) + tolerance,
    )


def coordinate_contract_operations(contract: "CoordinateContract") -> list:
    """Return the deterministic CRS-then-alignment execution provenance."""
    operations = [{
        "order": 1,
        "kind": "crs",
        "definition": contract.operation,
    }]
    if contract.post_crs_alignment is not None:
        operations.append({
            "order": 2,
            "kind": "post_crs_alignment",
            "matrix": list(contract.post_crs_alignment),
        })
    return operations


def coordinate_contract_metadata(contract: "CoordinateContract") -> dict:
    """Return the shared public provenance embedded by boundary and report adapters."""
    return {
        "schema": contract.schema,
        "crs_wkt": contract.output_crs_wkt,
        "operation": contract.operation,
        "storage_offset": list(contract.storage_offset),
        "vertical_reference": contract.vertical_reference.value,
        "height_assumptions": list(contract.height_assumptions),
        "warnings": list(contract.warnings),
        "operations": coordinate_contract_operations(contract),
        "post_crs_alignment": (
            list(contract.post_crs_alignment)
            if contract.post_crs_alignment is not None
            else None
        ),
    }


def transform_output_boundary(
    boundary,
    contract: "CoordinateContract",
    *,
    finest_resolution: float,
    serialization_bound: float,
    maximum_depth: int = 32,
) -> np.ndarray:
    """Exactly invert and resolution-adapt an output-CRS boundary."""
    try:
        source = np.asarray(boundary, dtype=np.float64)
    except Exception as error:
        raise CoordinateContractError(
            "boundary is not numeric", artifact="boundary",
            operation="inverse boundary transformation", cause=error,
        ) from error
    if source.ndim != 2 or source.shape[1] not in (2, 3) or len(source) < 2:
        raise CoordinateContractError(
            "boundary must contain at least two 2D or 3D coordinates",
            artifact="boundary", operation="inverse boundary transformation",
        )
    if not np.all(np.isfinite(source)):
        raise CoordinateContractError(
            "boundary contains non-finite coordinates", artifact="boundary",
            operation="inverse boundary transformation", coordinate=source[0],
        )
    tolerances = (finest_resolution, serialization_bound)
    if not all(math.isfinite(value) and value >= 0.0 for value in tolerances):
        raise CoordinateContractError(
            "boundary tolerances must be finite and non-negative",
            artifact="boundary", operation="adaptive subdivision",
        )
    if maximum_depth < 1:
        raise CoordinateContractError(
            "maximum subdivision depth must be positive",
            artifact="boundary", operation="adaptive subdivision",
        )

    if source.shape[1] == 2:
        reference_height = contract.transform_points(
            [[0.0, 0.0, 0.0]], apply_storage_offset=False
        )[0, 2]
        source = np.column_stack((source, np.full(len(source), reference_height)))
    threshold = max(finest_resolution * 0.25, serialization_bound)
    if threshold == 0.0:
        raise CoordinateContractError(
            "boundary subdivision tolerance must be positive",
            artifact="boundary", operation="adaptive subdivision",
        )

    def subdivide(start, end, transformed_start, transformed_end, depth):
        midpoint = start + (end - start) * 0.5
        transformed_midpoint = contract.inverse_points([midpoint])[0]
        chord_midpoint = transformed_start + (transformed_end - transformed_start) * 0.5
        deviation = float(np.linalg.norm(
            transformed_midpoint[:2] - chord_midpoint[:2]
        ))
        if deviation <= threshold:
            return [transformed_end]
        if depth >= maximum_depth:
            raise CoordinateContractError(
                "adaptive subdivision did not converge",
                artifact="boundary", operation="adaptive subdivision",
                coordinate=midpoint,
            )
        return (
            subdivide(start, midpoint, transformed_start, transformed_midpoint, depth + 1)
            + subdivide(midpoint, end, transformed_midpoint, transformed_end, depth + 1)
        )

    transformed_vertices = contract.inverse_points(source)
    result = [transformed_vertices[0]]
    for index in range(len(source) - 1):
        result.extend(subdivide(
            source[index], source[index + 1],
            transformed_vertices[index], transformed_vertices[index + 1], 0,
        ))
    return np.asarray(result, dtype=np.float64)


def _raster_coordinate_operation(contract: "CoordinateContract") -> str:
    """Return the persisted operation plus the post-CRS XY alignment, if any."""
    if contract.post_crs_alignment is None:
        return contract.operation
    matrix = np.asarray(contract.post_crs_alignment, dtype=np.float64).reshape(4, 4)
    if contract.vertical_reference == VerticalReference.UNREFERENCED:
        # Orthophoto pixels lie on topocentric Z=0. The coordinate contract
        # deliberately preserves that relative Z before applying alignment.
        matrix = matrix.copy()
        matrix[0, 2] = 0.0
        matrix[1, 2] = 0.0
    return contract.operation + (
        " +step +proj=affine"
        " +s11={:.17g} +s12={:.17g} +s13={:.17g}"
        " +s21={:.17g} +s22={:.17g} +s23={:.17g}"
        " +s31={:.17g} +s32={:.17g} +s33={:.17g}"
        " +xoff={:.17g} +yoff={:.17g} +zoff={:.17g}"
    ).format(
        matrix[0, 0], matrix[0, 1], matrix[0, 2],
        matrix[1, 0], matrix[1, 1], matrix[1, 2],
        matrix[2, 0], matrix[2, 1], matrix[2, 2],
        matrix[0, 3], matrix[1, 3], matrix[2, 3],
    )


def export_georeferenced_raster(
    source_path: str,
    corners_path: str,
    output_path: str,
    contract: "CoordinateContract",
    *,
    resolution: float,
    creation_options=None,
    resampling: str = "near",
    memory_limit_mb: Optional[int] = None,
    gdal_module=None,
) -> RasterExportResult:
    """Atomically warp a topocentric raster through the exact coordinate operation."""
    try:
        with open(corners_path, "r") as corner_file:
            values = [float(value) for value in corner_file.read().split()]
    except Exception as error:
        raise RasterExportError(
            "cannot read topocentric corner control",
            artifact=output_path,
            operation="read raster corner control",
            cause=error,
        ) from error
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise RasterExportError(
            "invalid topocentric corner control",
            artifact=output_path,
            operation="read raster corner control",
            coordinate=values[:3] if values else None,
        )
    xmin, ymin, xmax, ymax = values
    if xmin >= xmax or ymin >= ymax:
        raise RasterExportError(
            "empty topocentric raster extent",
            artifact=output_path,
            operation="read raster corner control",
            coordinate=(xmin, ymin, 0.0),
        )
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise RasterExportError(
            "output resolution must be finite and positive",
            artifact=output_path,
            operation="configure exact raster warp",
        )
    if gdal_module is None:
        try:
            from osgeo import gdal as gdal_module
        except Exception as error:
            raise RasterExportError(
                "GDAL Python bindings are unavailable",
                artifact=output_path,
                operation="configure exact raster warp",
                cause=error,
            ) from error

    output_directory = os.path.dirname(output_path) or "."
    os.makedirs(output_directory, exist_ok=True)
    descriptor, working_vrt = tempfile.mkstemp(
        prefix=os.path.basename(output_path) + ".", suffix=".topocentric.vrt",
        dir=output_directory,
    )
    os.close(descriptor)
    os.unlink(working_vrt)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(output_path) + ".", suffix=".tmp.tif",
        dir=output_directory,
    )
    os.close(descriptor)
    os.unlink(temporary)
    try:
        # A geocentric source declaration gives GDAL a metre-based 3D source
        # domain. coordinateOperation below replaces its CRS-to-CRS operation;
        # the raster coordinates are the topocentric X/Y values from the corner
        # file and Z=0, exactly as consumed by the persisted PROJ pipeline.
        translate_options = gdal_module.TranslateOptions(
            format="VRT",
            outputBounds=[xmin, ymax, xmax, ymin],
            outputSRS="EPSG:4978",
        )
        topocentric = gdal_module.Translate(
            working_vrt, source_path, options=translate_options
        )
        if topocentric is None:
            raise RasterExportError(
                "GDAL could not assign topocentric raster control",
                artifact=output_path,
                operation="assign raster corner control",
            )
        topocentric.FlushCache()
        coordinate_operation = _raster_coordinate_operation(contract)
        transformer_options = [
            "SRC_SRS=EPSG:4978",
            "DST_SRS={}".format(contract.output_crs_wkt),
            "COORDINATE_OPERATION={}".format(coordinate_operation),
        ]
        transformer = gdal_module.Transformer(
            topocentric, None, transformer_options
        )
        if transformer is None:
            raise RasterExportError(
                "GDAL could not construct the exact raster transformer",
                artifact=output_path,
                operation="validate exact raster mapping",
            )
        fractions = np.linspace(0.0, 1.0, 5)
        pixel_samples = [
            (topocentric.RasterXSize * column,
             topocentric.RasterYSize * row, 0.0)
            for row in fractions for column in fractions
        ]
        transformed_samples, successful = transformer.TransformPoints(
            False, pixel_samples
        )
        if not all(successful):
            raise RasterExportError(
                "GDAL rejected an exact raster mapping sample",
                artifact=output_path,
                operation="validate exact raster mapping",
            )
        topocentric_samples = np.asarray([
            (xmin + (xmax - xmin) * column,
             ymax - (ymax - ymin) * row, 0.0)
            for row in fractions for column in fractions
        ])
        reference_samples = contract.transform_points(
            topocentric_samples, apply_storage_offset=False
        )
        transformation_errors = np.linalg.norm(
            np.asarray(transformed_samples, dtype=np.float64)[:, :2]
            - reference_samples[:, :2],
            axis=1,
        )
        maximum_transformation_error = float(np.max(transformation_errors))
        if maximum_transformation_error > 0.0001:
            worst = int(np.argmax(transformation_errors))
            raise RasterExportError(
                "GDAL mapping differs from the coordinate contract by {:.17g} m".format(
                    maximum_transformation_error
                ),
                artifact=output_path,
                operation="validate exact raster mapping",
                coordinate=topocentric_samples[worst],
            )
        options = {
            "format": "GTiff",
            "srcSRS": "EPSG:4978",
            "dstSRS": contract.output_crs_wkt,
            "coordinateOperation": coordinate_operation,
            "errorThreshold": 0.0,
            "xRes": resolution,
            "yRes": resolution,
            "resampleAlg": resampling,
            "multithread": True,
            "creationOptions": [
                "{}={}".format(key, value)
                for key, value in (creation_options or {}).items()
            ],
        }
        if memory_limit_mb is not None:
            # ChunkAndWarpMulti overlaps two chunk working sets.
            options["warpMemoryLimit"] = memory_limit_mb * 1024 * 1024 / 2
        warp_options = gdal_module.WarpOptions(**options)
        warped = gdal_module.Warp(temporary, topocentric, options=warp_options)
        if warped is None:
            raise RasterExportError(
                "GDAL exact raster warp returned no dataset",
                artifact=output_path,
                operation="warp exact raster",
            )
        warped.FlushCache()
        warped = None
        topocentric = None
        os.replace(temporary, output_path)
        temporary = None
        return RasterExportResult(
            (xmin, ymin, xmax, ymax),
            maximum_transformation_error=maximum_transformation_error,
        )
    except GeoreferencingError:
        raise
    except Exception as error:
        raise RasterExportError(
            "raster export failed",
            artifact=output_path,
            operation="warp exact raster",
            coordinate=(xmin, ymin, 0.0),
            cause=error,
        ) from error
    finally:
        for path in (working_vrt, temporary):
            if path is not None:
                try:
                    os.unlink(path)
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


def export_georeferenced_mesh(
    source_path: str,
    output_path: str,
    contract: "CoordinateContract",
    *,
    apply_storage_offset: bool = True,
) -> MeshExportResult:
    """Atomically write an exact OBJ derivative of a canonical topocentric mesh."""
    temporary = None
    try:
        with open(source_path, "r") as source:
            lines = source.readlines()

        vertices = np.asarray([
            [float(value) for value in line.split()[1:4]]
            for line in lines if line.startswith("v ")
        ], dtype=np.float64)
        normals = np.asarray([
            [float(value) for value in line.split()[1:4]]
            for line in lines if line.startswith("vn ")
        ], dtype=np.float64)
        if not len(vertices):
            raise MeshExportError(
                "canonical mesh contains no vertices",
                artifact=output_path,
                operation="transform OBJ",
            )
        if normals.size == 0:
            normals = np.empty((0, 3), dtype=np.float64)

        face_records = []
        normal_vertices = [[] for _ in range(len(normals))]
        seen_normal_vertices = set()
        face_count = 0
        for line_index, line in enumerate(lines):
            if not line.startswith("f "):
                continue
            face_count += 1
            tokens = line.split()[1:]
            parsed = []
            for token in tokens:
                fields = token.split("/")
                vertex_index = _obj_index(fields[0], len(vertices), "vertex")
                normal_index = None
                if len(fields) >= 3 and fields[2]:
                    normal_index = _obj_index(fields[2], len(normals), "normal")
                    if (normal_index, vertex_index) not in seen_normal_vertices:
                        normal_vertices[normal_index].append(vertex_index)
                        seen_normal_vertices.add((normal_index, vertex_index))
                parsed.append((fields, vertex_index, normal_index))
            face_records.append((line_index, parsed))

        transformed_vertices = contract.transform_points(
            vertices, apply_storage_offset=apply_storage_offset
        )

        normal_pairs = []
        normal_pair_indexes = {}
        for normal_index, vertex_indexes in enumerate(normal_vertices):
            if not vertex_indexes:
                vertex_indexes = [0]
            for vertex_index in vertex_indexes:
                normal_pair_indexes[(normal_index, vertex_index)] = len(normal_pairs) + 1
                normal_pairs.append((normal_index, vertex_index))
        if normal_pairs:
            normal_points = vertices[[vertex for _, vertex in normal_pairs]]
            source_normals = normals[[normal for normal, _ in normal_pairs]]
            transformed_normals = contract.transform_normals(normal_points, source_normals)
        else:
            transformed_normals = np.empty((0, 3), dtype=np.float64)

        rewritten_faces = {}
        for line_index, parsed in face_records:
            tokens = []
            for fields, vertex_index, normal_index in parsed:
                if normal_index is not None:
                    fields[2] = str(normal_pair_indexes[(normal_index, vertex_index)])
                tokens.append("/".join(fields))
            rewritten_faces[line_index] = "f " + " ".join(tokens) + "\n"

        output_directory = os.path.dirname(output_path) or "."
        os.makedirs(output_directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=os.path.basename(output_path) + ".", suffix=".tmp.obj",
            dir=output_directory,
        )
        with os.fdopen(descriptor, "w") as output:
            vertex_index = 0
            emitted_normals = False
            for line_index, line in enumerate(lines):
                if line.startswith("v "):
                    output.write("v {:.17g} {:.17g} {:.17g}\n".format(
                        *transformed_vertices[vertex_index]
                    ))
                    vertex_index += 1
                elif line.startswith("vn "):
                    if not emitted_normals:
                        for normal in transformed_normals:
                            output.write("vn {:.17g} {:.17g} {:.17g}\n".format(*normal))
                        emitted_normals = True
                elif line_index in rewritten_faces:
                    output.write(rewritten_faces[line_index])
                else:
                    output.write(line)
        os.replace(temporary, output_path)
        temporary = None
        return MeshExportResult(len(vertices), len(transformed_normals), face_count)
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


def select_las_encoding(
    coordinates: Iterable[Sequence[float]],
    spacing: Optional[float],
    *,
    output_unit_to_metre: float = 1.0,
) -> LasEncoding:
    """Select ODX's adaptive scale and a deterministic signed-int32-safe offset."""
    points = _as_points(coordinates, "select LAS encoding")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise PointCloudExportError(
            "point spacing must be finite and positive",
            artifact="georeferenced point cloud",
            operation="select LAS encoding",
        )
    if not math.isfinite(output_unit_to_metre) or output_unit_to_metre <= 0.0:
        raise PointCloudExportError(
            "output linear unit must be finite and positive",
            artifact="georeferenced point cloud",
            operation="select LAS encoding",
        )
    spacing_scale_metres = pow(10.0, round(math.log10(spacing))) / 10.0
    scale_value = min(spacing_scale_metres, 0.001) / output_unit_to_metre
    scale = np.full(3, scale_value, dtype=np.float64)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    offset = (lower + upper) / 2.0
    encoded_lower = np.rint((lower - offset) / scale)
    encoded_upper = np.rint((upper - offset) / scale)
    int32 = np.iinfo(np.int32)
    if np.any(encoded_lower < int32.min) or np.any(encoded_upper > int32.max):
        raise PointCloudExportError(
            "selected LAS scale cannot encode coordinates as signed 32-bit integers",
            artifact="georeferenced point cloud",
            operation="select LAS encoding",
            coordinate=lower,
        )
    return LasEncoding(tuple(scale), tuple(offset))


def _point_cloud_dtype(dtype: np.dtype) -> np.dtype:
    coordinate_dimensions = {"X", "Y", "Z"}
    return np.dtype([
        (name, np.float64 if name in coordinate_dimensions else dtype.fields[name][0])
        for name in dtype.names or ()
    ])


def transform_point_cloud_batch(
    contract: "CoordinateContract",
    points: np.ndarray,
    *,
    expected_count: Optional[int] = None,
) -> np.ndarray:
    """Transform one structured point batch without changing its dimensions or count."""
    source = np.asarray(points)
    names = set(source.dtype.names or ())
    if not {"X", "Y", "Z"}.issubset(names):
        raise PointCloudExportError(
            "point dimensions X, Y, and Z are required",
            artifact="georeferenced point cloud",
            operation="transform point batch",
        )
    if expected_count is not None and len(source) != expected_count:
        raise PointCloudExportError(
            "point count changed (expected {}, received {})".format(expected_count, len(source)),
            artifact="georeferenced point cloud",
            operation="transform point batch",
        )
    normal_names = ("NormalX", "NormalY", "NormalZ")
    present_normals = tuple(name for name in normal_names if name in names)
    if present_normals and len(present_normals) != 3:
        raise PointCloudExportError(
            "point normals require NormalX, NormalY, and NormalZ",
            artifact="georeferenced point cloud",
            operation="transform point batch",
        )
    source_coordinates = np.column_stack((source["X"], source["Y"], source["Z"]))
    try:
        transformed_coordinates = contract.transform_points(
            source_coordinates, apply_storage_offset=False
        )
        transformed_normals = None
        if present_normals:
            transformed_normals = contract.transform_normals(
                source_coordinates,
                np.column_stack(tuple(source[name] for name in normal_names)),
            )
    except GeoreferencingError as error:
        raise PointCloudExportError(
            "exact point transformation failed",
            artifact="georeferenced point cloud",
            operation="transform point batch",
            coordinate=source_coordinates[0] if len(source_coordinates) else None,
            cause=error,
        ) from error
    transformed = np.empty(source.shape, dtype=_point_cloud_dtype(source.dtype))
    for name in source.dtype.names:
        transformed[name] = source[name]
    for index, name in enumerate(("X", "Y", "Z")):
        transformed[name] = transformed_coordinates[:, index]
    if transformed_normals is not None:
        for index, name in enumerate(normal_names):
            transformed[name] = transformed_normals[:, index]
    if len(transformed) != len(source):
        raise PointCloudExportError(
            "point count changed during transformation",
            artifact="georeferenced point cloud",
            operation="transform point batch",
        )
    return transformed


PUBLIC_LAZ_DIMENSIONS = (
    "X", "Y", "Z", "Intensity", "ReturnNumber", "NumberOfReturns",
    "ScanDirectionFlag", "EdgeOfFlightLine", "Classification", "Synthetic",
    "KeyPoint", "Withheld", "Overlap", "ScanAngleRank", "UserData",
    "PointSourceId", "GpsTime", "Red", "Green", "Blue",
)


def public_point_cloud_batch(points: np.ndarray) -> np.ndarray:
    """Select source fields that belong to stock ODX's public LAZ contract."""
    source = np.asarray(points)
    names = tuple(
        name for name in PUBLIC_LAZ_DIMENSIONS if name in (source.dtype.names or ())
    )
    public = np.empty(
        source.shape, dtype=[(name, source.dtype[name]) for name in names]
    )
    for name in names:
        public[name] = source[name]
    return public


def export_georeferenced_point_cloud(
    source_path: str,
    output_path: str,
    contract: "CoordinateContract",
    *,
    spacing: float,
    chunk_size: int = 250000,
    vlrs: Optional[Sequence[dict]] = None,
    pdal_module=None,
) -> PointCloudExportResult:
    """Stream a canonical point cloud twice and atomically publish an exact LAZ."""
    if chunk_size <= 0:
        raise PointCloudExportError(
            "chunk size must be positive",
            artifact=output_path,
            operation="stream point cloud",
        )
    if pdal_module is None:
        try:
            import pdal as pdal_module
        except Exception as error:
            raise PointCloudExportError(
                "PDAL Python bindings are unavailable",
                artifact=output_path,
                operation="stream point cloud",
                cause=error,
            ) from error

    reader_spec = json.dumps([
        source_path,
        {"type": "filters.ferry", "dimensions": "views=>UserData"},
    ])

    def batches():
        pipeline = pdal_module.Pipeline(reader_spec)
        if not pipeline.streamable:
            raise PointCloudExportError(
                "canonical point-cloud reader is not streamable",
                artifact=output_path,
                operation="stream point cloud",
            )
        return pipeline.iterator(chunk_size=chunk_size)

    point_count = 0
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    output_dtype = None
    dimensions = None
    temporary = None
    try:
        for source in batches():
            if len(source) == 0:
                continue
            transformed = transform_point_cloud_batch(
                contract, public_point_cloud_batch(source)
            )
            coordinates = np.column_stack(
                (transformed["X"], transformed["Y"], transformed["Z"])
            )
            lower = np.minimum(lower, np.min(coordinates, axis=0))
            upper = np.maximum(upper, np.max(coordinates, axis=0))
            point_count += len(transformed)
            if output_dtype is None:
                output_dtype = transformed.dtype
                dimensions = tuple(transformed.dtype.names)
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

        encoding = select_las_encoding(
            np.vstack((lower, upper)),
            spacing if spacing is not None else 0.01,
            output_unit_to_metre=contract.output_crs.axis_info[0].unit_conversion_factor,
        )
        writer = {
            "type": "writers.las",
            "filename": None,
            "compression": "laszip",
            "a_srs": contract.output_crs_wkt,
            "minor_version": 4,
            "dataformat_id": 3,
            "enhanced_srs_vlrs": True,
            "scale_x": encoding.scale[0],
            "scale_y": encoding.scale[1],
            "scale_z": encoding.scale[2],
            "offset_x": encoding.offset[0],
            "offset_y": encoding.offset[1],
            "offset_z": encoding.offset[2],
        }
        if vlrs is not None:
            writer["vlrs"] = vlrs

        output_directory = os.path.dirname(output_path) or "."
        os.makedirs(output_directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=os.path.basename(output_path) + ".", suffix=".tmp.laz",
            dir=output_directory,
        )
        os.close(descriptor)
        os.unlink(temporary)
        writer["filename"] = temporary
        source_batches = iter(batches())
        buffer = np.empty(chunk_size, dtype=output_dtype)
        written_count = 0

        def load_next_batch():
            nonlocal written_count
            while True:
                try:
                    source = next(source_batches)
                except StopIteration:
                    return 0
                if len(source):
                    break
            transformed = transform_point_cloud_batch(
                contract, public_point_cloud_batch(source)
            )
            if transformed.dtype != output_dtype or len(transformed) > len(buffer):
                raise PointCloudExportError(
                    "streamed point layout changed during export",
                    artifact=output_path,
                    operation="write LAZ",
                )
            buffer[:len(transformed)] = transformed
            written_count += len(transformed)
            return len(transformed)

        pipeline = pdal_module.Pipeline(
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
                "point count changed during LAZ serialization (source {}, transformed {}, written {})".format(
                    point_count, written_count, reported_count
                ),
                artifact=output_path,
                operation="write LAZ",
            )
        os.replace(temporary, output_path)
        serialization_tolerance = max(encoding.scale) * 0.5
        bounds = decoded_coordinate_bounds(
            np.vstack((lower, upper)), tolerance=serialization_tolerance
        )
        return PointCloudExportResult(point_count, dimensions, encoding, bounds)
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


def _as_points(coordinates: Iterable[Sequence[float]], operation: str) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise CoordinateContractError(
            "coordinates must have explicit (easting, northing, height) order",
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


def _is_dynamic(crs: CRS) -> bool:
    candidates = [crs] + list(crs.sub_crs_list)
    return any(
        candidate.datum is not None
        and "dynamic" in (candidate.datum.type_name or "").lower()
        for candidate in candidates
    )


def _validate_static_crs(crs: CRS) -> None:
    if _is_dynamic(crs):
        raise CoordinateContractError("dynamic or epoch-dependent CRS is unsupported")
    if len(crs.axis_info) < 2:
        raise CoordinateContractError("invalid axis definition")
    for axis in crs.axis_info[:2]:
        if not axis.unit_name or not math.isfinite(axis.unit_conversion_factor):
            raise CoordinateContractError("invalid axis or unit definition")


def _requires_coordinate_epoch(transformer: Transformer) -> bool:
    definition = (transformer.definition or "").lower()
    return any(marker in definition for marker in ("t_epoch", "deformation", "velocitygrid"))


def _topocentric_to_geographic_operation(anchor: TopocentricAnchor) -> str:
    return (
        "+proj=pipeline "
        "+step +inv +proj=topocentric +ellps=WGS84 "
        "+lat_0={:.15g} +lon_0={:.15g} +h_0={:.15g} "
        "+step +inv +proj=cart +ellps=WGS84"
    ).format(anchor.latitude, anchor.longitude, anchor.ellipsoidal_height)


def _output_crs_for_anchor(anchor: TopocentricAnchor) -> CRS:
    if anchor.latitude > 84.0:
        return CRS.from_epsg(5041)
    if anchor.latitude < -80.0:
        return CRS.from_epsg(5042)
    zone = int(math.floor((anchor.longitude + 180.0) / 6.0)) + 1
    zone = min(60, max(1, zone))
    return CRS.from_epsg((32600 if anchor.latitude >= 0.0 else 32700) + zone)


def _coordinate_operation(anchor: TopocentricAnchor, output_crs: CRS) -> str:
    operation = _topocentric_to_geographic_operation(anchor)
    epsg = output_crs.to_epsg()
    if 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760:
        suffix = "+step +proj=utm +zone={} +ellps=WGS84".format(epsg % 100)
        if 32701 <= epsg <= 32760:
            suffix += " +south"
    elif epsg in (5041, 5042):
        latitude = 90 if epsg == 5041 else -90
        suffix = (
            "+step +proj=stere +lat_0={0} +lat_ts={0} +lon_0=0 "
            "+k=0.994 +x_0=2000000 +y_0=2000000 +ellps=WGS84"
        ).format(latitude)
    else:
        raise CoordinateContractError("unsupported automatic output CRS")
    return operation + " " + suffix


def _area_contains_lon_lat(area: Sequence[float], longitude: float, latitude: float) -> bool:
    west, south, east, north = area
    longitude_inside = west <= longitude <= east if west <= east else longitude >= west or longitude <= east
    return longitude_inside and south <= latitude <= north


def _area_coverage(area: Sequence[float], longitude, latitude):
    inside = [
        _area_contains_lon_lat(area, x, y)
        for x, y in zip(longitude, latitude)
    ]
    return any(inside), all(inside)


def _area_status(
    anchor: TopocentricAnchor, output_crs: CRS, points: np.ndarray
) -> Tuple[Tuple[float, float, float, float], Tuple[str, ...]]:
    area = output_crs.area_of_use
    if area is None:
        raise CoordinateContractError("selected operation has no declared area of use")
    bounds = tuple(float(value) for value in area.bounds)
    geographic = Transformer.from_pipeline(_topocentric_to_geographic_operation(anchor))
    lon, lat, _ = geographic.transform(points[:, 0], points[:, 1], points[:, 2])
    any_inside, all_inside = _area_coverage(bounds, lon, lat)
    if not any_inside:
        raise CoordinateContractError(
            "dataset is wholly outside selected operation area",
            coordinate=points[0],
        )
    warnings = ()
    if not all_inside:
        warnings = ("dataset crosses selected CRS or operation area boundary",)
    return bounds, warnings


@dataclass(frozen=True)
class CoordinateContract:
    anchor: TopocentricAnchor
    output_crs_wkt: str
    operation: str
    storage_offset: Tuple[float, float, float]
    vertical_reference: VerticalReference
    operation_area: Tuple[float, float, float, float]
    grids: Tuple[str, ...]
    stated_accuracy: float
    axis_order: Tuple[str, str, str]
    geometry_unit: str
    height_assumptions: Tuple[str, ...]
    warnings: Tuple[str, ...]
    versions: Tuple[Tuple[str, str], ...]
    control_provenance: Tuple[ControlProvenance, ...] = ()
    post_crs_alignment: Optional[Tuple[float, ...]] = None
    schema: str = MANIFEST_SCHEMA

    @property
    def output_crs(self) -> CRS:
        return CRS.from_wkt(self.output_crs_wkt)

    def _transformer(self) -> Transformer:
        try:
            transformer = Transformer.from_pipeline(self.operation)
        except Exception as error:
            raise CoordinateContractError(
                "persisted PROJ operation is unavailable",
                operation="load operation",
                cause=error,
            ) from error
        if transformer.definition != self.operation:
            raise CoordinateContractError(
                "persisted PROJ operation did not reload deterministically",
                operation="load operation",
            )
        return transformer

    def transform_points(
        self, coordinates: Iterable[Sequence[float]], *, apply_storage_offset: bool = True
    ) -> np.ndarray:
        points = _as_points(coordinates, "forward point transformation")
        try:
            transformed = np.column_stack(self._transformer().transform(*points.T, errcheck=True))
            if self.vertical_reference == VerticalReference.UNREFERENCED:
                transformed[:, 2] = points[:, 2] / self.output_crs.axis_info[0].unit_conversion_factor
            transformed = self._apply_alignment(transformed)
            if apply_storage_offset:
                transformed = transformed - np.asarray(self.storage_offset)
        except GeoreferencingError:
            raise
        except Exception as error:
            raise CoordinateContractError(
                "coordinate operation failed",
                operation="forward point transformation",
                coordinate=points[0],
                cause=error,
            ) from error
        return self._finite_result(transformed, points[0], "forward point transformation")

    def inverse_points(
        self, coordinates: Iterable[Sequence[float]], *, storage_offset_applied: bool = False
    ) -> np.ndarray:
        points = _as_points(coordinates, "inverse point transformation")
        absolute = points + np.asarray(self.storage_offset) if storage_offset_applied else points.copy()
        absolute = self._remove_alignment(absolute)
        try:
            if self.vertical_reference == VerticalReference.UNREFERENCED:
                relative_z = absolute[:, 2] * self.output_crs.axis_info[0].unit_conversion_factor
                seed = absolute.copy()
                seed[:, 2] = self.anchor.ellipsoidal_height + relative_z
                transformed = np.column_stack(
                    self._transformer().transform(
                        *seed.T, direction=TransformDirection.INVERSE, errcheck=True
                    )
                )
                transformed[:, 2] = relative_z
                transformed = self._solve_unreferenced_inverse(absolute[:, :2], transformed)
            else:
                transformed = np.column_stack(
                    self._transformer().transform(
                        *absolute.T, direction=TransformDirection.INVERSE, errcheck=True
                    )
                )
        except Exception as error:
            raise CoordinateContractError(
                "coordinate operation failed",
                operation="inverse point transformation",
                coordinate=points[0],
                cause=error,
            ) from error
        return self._finite_result(transformed, points[0], "inverse point transformation")

    def _solve_unreferenced_inverse(self, target_xy: np.ndarray, estimate: np.ndarray) -> np.ndarray:
        epsilon = 0.001
        transformer = self._transformer()
        for _ in range(6):
            projected = np.column_stack(transformer.transform(*estimate.T, errcheck=True))[:, :2]
            residual = target_xy - projected
            if np.max(np.abs(residual)) <= 1e-7:
                return estimate
            columns = []
            for axis in (0, 1):
                shifted = estimate.copy()
                shifted[:, axis] += epsilon
                shifted_xy = np.column_stack(
                    transformer.transform(*shifted.T, errcheck=True)
                )[:, :2]
                columns.append((shifted_xy - projected) / epsilon)
            jacobians = np.stack(columns, axis=2)
            corrections = np.stack(
                [np.linalg.solve(jacobian, error) for jacobian, error in zip(jacobians, residual)]
            )
            estimate[:, :2] += corrections
        raise CoordinateContractError(
            "unreferenced inverse did not converge",
            operation="inverse point transformation",
            coordinate=target_xy[0],
        )

    def _jacobians(self, coordinates: Iterable[Sequence[float]]) -> np.ndarray:
        points = _as_points(coordinates, "local Jacobian")
        epsilon = 0.001
        offsets = np.eye(3) * epsilon
        plus = np.concatenate([points + offset for offset in offsets])
        minus = np.concatenate([points - offset for offset in offsets])
        differences = (
            self.transform_points(plus, apply_storage_offset=False)
            - self.transform_points(minus, apply_storage_offset=False)
        ) / (2.0 * epsilon)
        count = len(points)
        return np.stack([differences[index * count:(index + 1) * count] for index in range(3)], axis=2)

    def transform_tangents(self, coordinates, tangents) -> np.ndarray:
        vectors = _as_points(tangents, "tangent transformation")
        jacobians = self._jacobians(coordinates)
        if len(vectors) != len(jacobians):
            raise CoordinateContractError("point and tangent counts differ", operation="tangent transformation")
        return np.einsum("nij,nj->ni", jacobians, vectors)

    def transform_normals(self, coordinates, normals) -> np.ndarray:
        vectors = _as_points(normals, "normal transformation")
        jacobians = self._jacobians(coordinates)
        if len(vectors) != len(jacobians):
            raise CoordinateContractError("point and normal counts differ", operation="normal transformation")
        try:
            result = np.linalg.solve(
                np.swapaxes(jacobians, 1, 2), vectors[..., np.newaxis]
            )[..., 0]
        except Exception as error:
            raise CoordinateContractError(
                "normal transformation failed",
                operation="normal transformation",
                coordinate=vectors[0],
                cause=error,
            ) from error
        lengths = np.linalg.norm(result, axis=1)
        if np.any(lengths == 0.0):
            raise CoordinateContractError("zero-length transformed normal", operation="normal transformation")
        return result / lengths[:, None]

    def transform_camera_frames(self, coordinates, frames) -> np.ndarray:
        jacobians = self._jacobians(coordinates)
        try:
            source = np.asarray(frames, dtype=np.float64)
        except Exception as error:
            raise CoordinateContractError(
                "camera frames are not numeric",
                operation="camera frame transformation",
                cause=error,
            ) from error
        if source.shape != (len(jacobians), 3, 3):
            raise CoordinateContractError("camera frames must be Nx3x3", operation="camera frame transformation")
        if not np.all(np.isfinite(source)):
            raise CoordinateContractError(
                "non-finite camera frame",
                operation="camera frame transformation",
                coordinate=source[0].flat[:3],
            )
        result = []
        try:
            for jacobian, frame in zip(jacobians, source):
                u, _, vt = np.linalg.svd(jacobian @ frame)
                rotation = u @ vt
                if np.linalg.det(rotation) < 0.0:
                    u[:, -1] *= -1.0
                    rotation = u @ vt
                result.append(rotation)
        except Exception as error:
            raise CoordinateContractError(
                "camera-frame orthonormalization failed",
                operation="camera frame transformation",
                coordinate=source[0].flat[:3],
                cause=error,
            ) from error
        transformed = np.asarray(result)
        return self._finite_result(
            transformed, source[0].flat[:3], "camera frame transformation"
        )

    def with_post_crs_alignment(self, matrix: Sequence[Sequence[float]]):
        if self.post_crs_alignment is not None:
            raise CoordinateContractError(
                "post-CRS alignment is already defined",
                operation="derive aligned contract",
            )
        alignment = np.asarray(matrix, dtype=np.float64)
        if alignment.shape != (4, 4) or not np.all(np.isfinite(alignment)):
            raise CoordinateContractError("post-CRS alignment must be a finite 4x4 matrix")
        if not np.allclose(alignment[3], [0.0, 0.0, 0.0, 1.0]):
            raise CoordinateContractError("post-CRS alignment must be affine")
        try:
            np.linalg.inv(alignment[:3, :3])
        except Exception as error:
            raise CoordinateContractError(
                "post-CRS alignment must be invertible",
                operation="derive aligned contract",
                cause=error,
            ) from error
        return replace(self, post_crs_alignment=tuple(float(value) for value in alignment.flat))

    def without_post_crs_alignment(self):
        return replace(self, post_crs_alignment=None)

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ValueError("unsupported manifest schema")
        _validate_static_crs(self.output_crs)
        expected = Transformer.from_pipeline(
            _coordinate_operation(self.anchor, self.output_crs)
        ).definition
        if self.operation != expected:
            raise ValueError("operation does not match the persisted anchor and output CRS")
        if self.axis_order != ("easting", "northing", "height"):
            raise ValueError("invalid application axis order")
        if self.geometry_unit != self.output_crs.axis_info[0].unit_name:
            raise ValueError("output unit does not match the output CRS")
        if self.vertical_reference not in tuple(VerticalReference):
            raise ValueError("invalid vertical-reference status")
        if len(self.storage_offset) != 3 or self.storage_offset[2] != 0.0:
            raise ValueError("invalid XY-only storage offset")
        if not all(math.isfinite(value) for value in self.storage_offset):
            raise ValueError("non-finite storage offset")
        if self.post_crs_alignment is not None:
            alignment = np.asarray(self.post_crs_alignment).reshape(4, 4)
            if not np.all(np.isfinite(alignment)) or not np.allclose(
                alignment[3], [0.0, 0.0, 0.0, 1.0]
            ):
                raise ValueError("invalid post-CRS alignment")
            np.linalg.inv(alignment[:3, :3])
        self._transformer()

    def _apply_alignment(self, points: np.ndarray) -> np.ndarray:
        if self.post_crs_alignment is None:
            return points
        matrix = np.asarray(self.post_crs_alignment).reshape(4, 4)
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    def _remove_alignment(self, points: np.ndarray) -> np.ndarray:
        if self.post_crs_alignment is None:
            return points
        matrix = np.asarray(self.post_crs_alignment).reshape(4, 4)
        try:
            inverse = np.linalg.inv(matrix[:3, :3])
        except Exception as error:
            raise CoordinateContractError(
                "persisted post-CRS alignment is not invertible",
                operation="inverse point transformation",
                coordinate=points[0],
                cause=error,
            ) from error
        return (points - matrix[:3, 3]) @ inverse.T

    @staticmethod
    def _finite_result(result: np.ndarray, sample, operation: str) -> np.ndarray:
        if not np.all(np.isfinite(result)):
            raise CoordinateContractError(
                "non-finite transformation result", operation=operation, coordinate=sample
            )
        return result

    def persist(self, path: str) -> None:
        payload = asdict(self)
        payload["operations"] = coordinate_contract_operations(self)
        _persist_json_manifest(path, payload, "manifest")


def _atomic_copy(source: str, destination: str, operation: str) -> None:
    directory = os.path.dirname(destination) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = destination + ".tmp"
    try:
        with open(source, "rb") as source_file, open(temporary, "wb") as output_file:
            while True:
                block = source_file.read(1024 * 1024)
                if not block:
                    break
                output_file.write(block)
        os.replace(temporary, destination)
    except Exception as error:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise LegacyCompatibilityError(
            "canonical reconstruction file operation failed",
            artifact=destination,
            operation=operation,
            cause=error,
        ) from error


def preserve_canonical_reconstruction(active_path: str, canonical_path: str) -> None:
    """Atomically snapshot the topocentric OpenSfM identity after reconstruction."""
    if not os.path.isfile(active_path):
        raise LegacyCompatibilityError(
            "topocentric reconstruction is missing; rerun from reconstruction",
            artifact=active_path,
            operation="preserve canonical reconstruction",
        )
    _atomic_copy(active_path, canonical_path, "preserve canonical reconstruction")


def canonical_reconstruction_path(tree, georeferenced: bool) -> str:
    """Select the authoritative reconstruction for geometry and its statistics."""
    return (
        tree.opensfm_topocentric_reconstruction
        if georeferenced
        else tree.opensfm_reconstruction
    )


def materialize_canonical_reconstruction(active_path: str, canonical_path: str) -> None:
    """Restore canonical geometry at the filename expected by stock OpenSfM."""
    if not os.path.isfile(canonical_path):
        raise LegacyCompatibilityError(
            "canonical topocentric reconstruction is missing; rerun from reconstruction",
            artifact=canonical_path,
            operation="materialize canonical reconstruction",
        )
    _atomic_copy(canonical_path, active_path, "materialize canonical reconstruction")


def publish_legacy_reconstruction_compatibility(
    active_path: str,
    canonical_path: str,
    generated_path: str,
    provenance_path: str,
    contract: "CoordinateContract",
    stock_export,
) -> None:
    """Publish stock OpenSfM's late affine view without exposing it to geometry work."""
    materialize_canonical_reconstruction(active_path, canonical_path)
    for stale_path in (generated_path, provenance_path):
        try:
            os.unlink(stale_path)
        except FileNotFoundError:
            pass

    try:
        stock_export()
    except Exception as error:
        try:
            os.unlink(generated_path)
        except FileNotFoundError:
            pass
        raise LegacyCompatibilityError(
            "stock OpenSfM compatibility export failed",
            artifact=generated_path,
            operation="publish legacy reconstruction",
            cause=error,
        ) from error
    if not os.path.isfile(generated_path):
        raise LegacyCompatibilityError(
            "stock OpenSfM did not produce the compatibility reconstruction",
            artifact=generated_path,
            operation="publish legacy reconstruction",
        )

    os.replace(generated_path, active_path)
    _persist_json_manifest(
        provenance_path,
        {
            "schema": "odx-reconstruction-compatibility-v1",
            "representation": "affine_approximation",
            "exact": False,
            "source": os.path.basename(canonical_path),
            "artifact": os.path.basename(active_path),
            "operation": "stock OpenSfM export_geocoords",
            "output_crs": contract.output_crs_wkt,
            "xy_offset": list(contract.storage_offset[:2]),
            "consumption": "legacy external compatibility only",
        },
        "legacy reconstruction provenance",
    )


def resolve_coordinate_contract(
    anchor: TopocentricAnchor,
    controls: NormalizedControlSet,
    *,
    storage_offset: Sequence[float] = (0.0, 0.0),
    output_selection: Optional[OutputSelection] = None,
) -> CoordinateContract:
    points = _as_points(controls.coordinates, "resolve")
    if output_selection is None:
        output_crs = _output_crs_for_anchor(anchor)
        selected_offset = storage_offset
        vertical_reference = (
            VerticalReference.WGS84_ELLIPSOIDAL
            if controls.vertical_control
            else VerticalReference.UNREFERENCED
        )
        height_assumptions = controls.height_assumptions
        selection_warnings = ()
    else:
        try:
            output_selection.validate()
        except Exception as error:
            raise CoordinateContractError(
                "invalid shared output selection", operation="combine output selection",
                cause=error,
            ) from error
        output_crs = output_selection.output_crs
        selected_offset = output_selection.storage_offset
        vertical_reference = output_selection.vertical_reference
        height_assumptions = output_selection.height_assumptions
        selection_warnings = output_selection.warnings
    _validate_static_crs(output_crs)
    operation = _coordinate_operation(anchor, output_crs)
    try:
        transformer = Transformer.from_pipeline(operation)
        transformed = np.column_stack(transformer.transform(*points.T, errcheck=True))
    except Exception as error:
        raise CoordinateContractError(
            "could not resolve strict three-dimensional PROJ operation",
            coordinate=points[0],
            cause=error,
        ) from error
    if not np.all(np.isfinite(transformed)):
        raise CoordinateContractError("non-finite transformation result", coordinate=points[0])
    area, warnings = _area_status(anchor, output_crs, points)
    offset = tuple(float(value) for value in selected_offset)
    if len(offset) == 2:
        offset += (0.0,)
    if len(offset) != 3 or not all(math.isfinite(value) for value in offset) or offset[2] != 0.0:
        raise CoordinateContractError("storage offset must be finite and XY-only")
    unit = output_crs.axis_info[0].unit_name
    if not unit or output_crs.axis_info[0].unit_conversion_factor <= 0.0:
        raise CoordinateContractError("invalid output linear unit")
    return CoordinateContract(
        anchor=anchor,
        output_crs_wkt=output_crs.to_wkt(),
        operation=transformer.definition,
        storage_offset=offset,
        vertical_reference=vertical_reference,
        operation_area=area,
        grids=(),
        stated_accuracy=float(transformer.accuracy),
        axis_order=("easting", "northing", "height"),
        geometry_unit=unit,
        height_assumptions=height_assumptions,
        warnings=tuple(dict.fromkeys(selection_warnings + warnings)),
        versions=(
            ("python", platform.python_version()),
            ("pyproj", pyproj.__version__),
            ("proj", pyproj.proj_version_str),
            ("numpy", np.__version__),
        ),
        control_provenance=controls.provenance,
    )


def resolve_output_selection(
    anchor: TopocentricAnchor,
    controls: NormalizedControlSet,
    *,
    storage_offset: Sequence[float] = (0.0, 0.0),
) -> OutputSelection:
    """Resolve the parent choices that must not vary between split submodels."""
    contract = resolve_coordinate_contract(
        anchor, controls, storage_offset=storage_offset
    )
    return OutputSelection(
        output_crs_wkt=contract.output_crs_wkt,
        storage_offset=contract.storage_offset,
        vertical_reference=contract.vertical_reference,
        geometry_unit=contract.geometry_unit,
        height_assumptions=contract.height_assumptions,
        warnings=contract.warnings,
    )


def load_output_selection(path: str) -> OutputSelection:
    rerun = "rerun split preparation to create a compatible output selection"
    def decode(payload):
        payload["storage_offset"] = tuple(payload["storage_offset"])
        payload["height_assumptions"] = tuple(payload["height_assumptions"])
        payload["warnings"] = tuple(payload["warnings"])
        payload["vertical_reference"] = VerticalReference(payload["vertical_reference"])
        selection = OutputSelection(**payload)
        selection.validate()
        return selection
    return _load_json_manifest(path, rerun, "validate output selection", decode)


def validate_split_merge(
    output_selection: OutputSelection,
    contracts: Sequence[CoordinateContract],
    *,
    overlap_samples=(),
    product_tolerance: float = 0.0,
) -> None:
    """Reject incompatible submodel derivatives before parent artifact merging."""
    try:
        output_selection.validate()
    except Exception as error:
        raise CoordinateContractError(
            "invalid parent output selection", artifact="split merge",
            operation="premerge validation", cause=error,
        ) from error
    if not contracts:
        raise CoordinateContractError(
            "no submodel coordinate contracts", artifact="split merge",
            operation="premerge validation",
        )
    if not math.isfinite(product_tolerance) or product_tolerance < 0.0:
        raise CoordinateContractError(
            "product tolerance must be finite and non-negative",
            artifact="split merge", operation="premerge validation",
        )
    expected_alignment = contracts[0].post_crs_alignment
    for index, contract in enumerate(contracts):
        try:
            contract.validate()
        except Exception as error:
            raise CoordinateContractError(
                "submodel {} coordinate operation is invalid".format(index),
                artifact="split merge", operation="premerge validation", cause=error,
            ) from error
        comparisons = (
            ("output selection", contract.output_crs_wkt, output_selection.output_crs_wkt),
            ("output unit", contract.geometry_unit, output_selection.geometry_unit),
            ("storage offset", contract.storage_offset, output_selection.storage_offset),
            ("vertical-reference state", contract.vertical_reference, output_selection.vertical_reference),
            ("vertical defaulting", contract.height_assumptions, output_selection.height_assumptions),
        )
        for label, actual, expected in comparisons:
            if actual != expected:
                raise CoordinateContractError(
                    "submodel {} has incompatible {}".format(index, label),
                    artifact="split merge", operation="premerge validation",
                )
        if contract.post_crs_alignment != expected_alignment:
            raise CoordinateContractError(
                "submodel {} has incompatible post-CRS alignment".format(index),
                artifact="split merge", operation="premerge validation",
            )
    allowed_error = product_tolerance
    for first_index, first_coordinates, second_index, second_coordinates in overlap_samples:
        if not 0 <= first_index < len(contracts) or not 0 <= second_index < len(contracts):
            raise CoordinateContractError(
                "overlap references an unknown submodel", artifact="split merge",
                operation="overlap validation",
            )
        first = _as_points(first_coordinates, "split overlap")
        second = _as_points(second_coordinates, "split overlap")
        if first.shape != second.shape:
            raise CoordinateContractError(
                "overlap sample counts differ", artifact="split merge",
                operation="overlap validation",
            )
        errors = np.linalg.norm(first - second, axis=1)
        if np.any(errors > allowed_error):
            sample = int(np.argmax(errors))
            raise CoordinateContractError(
                "overlap coordinates exceed the product tolerance",
                artifact="split merge", operation="overlap validation",
                coordinate=first[sample],
            )


def transform_shared_control_observations(
    contract: CoordinateContract,
    control_observations: Sequence[SharedControlObservation],
) -> Tuple[dict, ...]:
    """Transform identified Earth-referenced control observations through one anchor."""
    if not control_observations:
        return ()
    identifiers = [observation.identifier for observation in control_observations]
    geographic = [
        (observation.longitude, observation.latitude, observation.height)
        for observation in control_observations
    ]
    to_topocentric = Transformer.from_pipeline(
        _topocentric_to_geographic_operation(contract.anchor)
    )
    local = np.column_stack(to_topocentric.transform(
        *np.asarray(geographic).T,
        direction=TransformDirection.INVERSE,
        errcheck=True,
    ))
    output = contract.transform_points(local, apply_storage_offset=False)
    if contract.vertical_reference == VerticalReference.UNREFERENCED:
        output[:, 2] = 0.0
    return tuple({
        "id": identifier,
        "geographic": list(source),
        "output": list(transformed),
    } for identifier, source, transformed in zip(identifiers, geographic, output))


def validate_shared_output_observations(
    contract: CoordinateContract,
    observations: Sequence[SharedOutputObservation],
    *,
    transformation_tolerance: float = 0.0001,
    representation_tolerance: float = 1e-9,
) -> None:
    """Validate published positions independently from product disagreement."""
    tolerances = (transformation_tolerance, representation_tolerance)
    if not all(math.isfinite(value) and value >= 0.0 for value in tolerances):
        raise CoordinateContractError(
            "output tolerances must be finite and non-negative",
            artifact="split merge", operation="published overlap validation",
        )
    if not observations:
        return
    expected = contract.transform_points(
        [observation.topocentric for observation in observations],
        apply_storage_offset=False,
    )
    published = _as_points(
        [observation.output for observation in observations],
        "published split overlap",
    )
    errors = np.linalg.norm(expected - published, axis=1)
    allowed_error = transformation_tolerance + representation_tolerance
    if np.any(errors > allowed_error):
        sample = int(np.argmax(errors))
        raise CoordinateContractError(
            "published overlap exceeds transformation and representation tolerances",
            artifact="split merge", operation="published overlap validation",
            coordinate=published[sample],
        )


def split_merge_derivative_paths(project_path: str) -> Tuple[str, ...]:
    """Enumerate only public artifact families eligible for parent merging."""
    fixed = (
        "cameras.json",
        os.path.join("odm_georeferencing", "odm_georeferenced_model.laz"),
        os.path.join("odm_georeferencing", "odm_georeferenced_model.bounds.geojson"),
        os.path.join("odm_georeferencing", "odm_georeferenced_model.bounds.gpkg"),
        os.path.join("odm_georeferencing", "odm_georeferenced_model.extent.geojson"),
        os.path.join("odm_georeferencing", "odm_georeferenced_model.extent.gpkg"),
        os.path.join("odm_orthophoto", "odm_orthophoto.tif"),
        os.path.join("odm_orthophoto", "odm_orthophoto_feathered.tif"),
        os.path.join("odm_orthophoto", "odm_orthophoto_cut.tif"),
        os.path.join("odm_dem", "dsm.tif"),
        os.path.join("odm_dem", "dtm.tif"),
        os.path.join("odm_report", "shots.geojson"),
        "orthophoto_tiles",
        "3d_tiles",
        "entwine_pointcloud",
    )
    paths = {
        path for path in fixed
        if os.path.exists(os.path.join(project_path, path))
    }
    for pattern in (
        os.path.join("odm_texturing*", "**", "odm_textured_model_geo.obj"),
        os.path.join("odm_texturing*", "**", "odm_textured_model_geo.glb"),
    ):
        for path in glob.glob(os.path.join(project_path, pattern), recursive=True):
            paths.add(os.path.relpath(path, project_path))
    return tuple(sorted(paths))


def persist_split_merge_derivatives(
    project_path: str, contract: CoordinateContract, *,
    shared_control_observations: Sequence[SharedControlObservation] = (),
    shared_output_observations: Sequence[SharedOutputObservation] = (),
) -> str:
    """Tie every mergeable submodel derivative to its immutable contract."""
    path = os.path.join(
        project_path, "odm_georeferencing", "merge_derivatives.json"
    )
    validate_shared_output_observations(contract, shared_output_observations)
    _persist_json_manifest(path, {
        "schema": SPLIT_MERGE_DERIVATIVES_SCHEMA,
        "coordinate_contract": coordinate_contract_metadata(contract),
        "artifacts": list(split_merge_derivative_paths(project_path)),
        "shared_control_observations": list(transform_shared_control_observations(
            contract, shared_control_observations
        )),
        "shared_output_observations": [asdict(observation) for observation in (
            shared_output_observations
        )],
    }, "split merge derivative metadata")
    return path


def load_split_merge_derivatives(
    project_path: str, contract: CoordinateContract
):
    """Validate merge inputs were published under the loaded contract."""
    path = os.path.join(
        project_path, "odm_georeferencing", "merge_derivatives.json"
    )
    def decode(payload):
        if payload.get("schema") != SPLIT_MERGE_DERIVATIVES_SCHEMA:
            raise ValueError("unsupported split merge derivative schema")
        if payload.get("coordinate_contract") != coordinate_contract_metadata(contract):
            raise ValueError("derivative metadata differs from its coordinate contract")
        persisted = tuple(payload.get("artifacts", ()))
        if persisted != split_merge_derivative_paths(project_path):
            raise ValueError("mergeable derivative set changed after publication")
        control_observations = tuple(payload.get("shared_control_observations", ()))
        source_observations = tuple(
            SharedControlObservation(item["id"], *item["geographic"])
            for item in control_observations
        )
        expected_observations = transform_shared_control_observations(
            contract, source_observations
        )
        for actual, expected in zip(control_observations, expected_observations):
            if actual["id"] != expected["id"] or not np.allclose(
                actual["output"], expected["output"], atol=0.0001, rtol=0.0
            ):
                raise ValueError("control observation differs from its coordinate contract")
        if len(control_observations) != len(expected_observations):
            raise ValueError("invalid shared control-observation metadata")
        output_observations = tuple(
            SharedOutputObservation(
                item["identifier"], tuple(item["topocentric"]), tuple(item["output"])
            )
            for item in payload.get("shared_output_observations", ())
        )
        validate_shared_output_observations(contract, output_observations)
        return persisted, control_observations, output_observations
    return _load_json_manifest(
        path, "rerun the submodel to publish compatible merge derivatives",
        "validate merge derivatives", decode,
    )


def load_coordinate_contract(path: str) -> CoordinateContract:
    rerun = "rerun from reconstruction to create a compatible coordinate contract"
    try:
        with open(path) as manifest:
            payload = json.load(manifest)
    except Exception as error:
        raise ManifestError(rerun, artifact=path, operation="load manifest", cause=error) from error
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(rerun, artifact=path, operation="validate manifest")
    try:
        persisted_operations = payload.pop("operations", None)
        payload["anchor"] = TopocentricAnchor(**payload["anchor"])
        for field in (
            "storage_offset", "operation_area", "grids", "axis_order",
            "height_assumptions", "warnings", "versions",
        ):
            payload[field] = tuple(payload[field])
        payload["versions"] = tuple(tuple(item) for item in payload["versions"])
        payload["vertical_reference"] = VerticalReference(payload["vertical_reference"])
        payload["control_provenance"] = tuple(
            ControlProvenance(
                source=ControlSource(item["source"]),
                input_crs_wkt=item["input_crs_wkt"],
                operation=item["operation"],
                grids=tuple(item["grids"]),
                accuracy=float(item["accuracy"]),
                axis_order=tuple(item["axis_order"]),
                height_assumptions=tuple(item["height_assumptions"]),
                warnings=tuple(item["warnings"]),
                versions=tuple(tuple(version) for version in item["versions"]),
            )
            for item in payload["control_provenance"]
        )
        if payload.get("post_crs_alignment") is not None:
            payload["post_crs_alignment"] = tuple(payload["post_crs_alignment"])
        contract = CoordinateContract(**payload)
        contract.validate()
        if (
            persisted_operations is not None
            and persisted_operations != coordinate_contract_operations(contract)
        ):
            raise ValueError("coordinate operation order or parameters changed")
        if dict(contract.versions).get("proj") != pyproj.proj_version_str:
            raise ValueError("PROJ runtime version changed")
        _validate_static_crs(contract.output_crs)
        return contract
    except Exception as error:
        raise ManifestError(rerun, artifact=path, operation="validate manifest", cause=error) from error


def load_topocentric_anchor(reference_path: str) -> TopocentricAnchor:
    try:
        with open(reference_path) as reference_file:
            reference = json.load(reference_file)
        return TopocentricAnchor(
            float(reference["latitude"]),
            float(reference["longitude"]),
            float(reference["altitude"]),
        )
    except Exception as error:
        raise CoordinateContractError(
            "could not read the persisted topocentric anchor",
            artifact=reference_path,
            operation="resolve stage contract",
            cause=error,
        ) from error


def resolve_stage_coordinate_contract(
    manifest_path: str,
    reference_path: str,
    storage_offset: Sequence[float],
    *,
    controls: NormalizedControlSet,
    fresh_reconstruction: bool = False,
    output_selection: Optional[OutputSelection] = None,
) -> CoordinateContract:
    """Create once or reload the immutable job contract at the stage seam."""
    loaded = None
    if os.path.exists(manifest_path):
        try:
            loaded = load_coordinate_contract(manifest_path)
        except ManifestError:
            if not fresh_reconstruction:
                raise
    if loaded is None and not fresh_reconstruction:
        raise ManifestError(
            "coordinate contract is missing; rerun from reconstruction",
            artifact=manifest_path,
            operation="stage reload",
        )
    anchor = load_topocentric_anchor(reference_path)
    expected = resolve_coordinate_contract(
        anchor,
        controls,
        storage_offset=storage_offset,
        output_selection=output_selection,
    )
    if loaded is not None:
        comparable_loaded = replace(
            loaded, post_crs_alignment=None, versions=expected.versions
        )
        if comparable_loaded == expected:
            return loaded
        if not fresh_reconstruction:
            raise ManifestError(
                "coordinate contract does not match reconstructed project metadata; rerun from reconstruction",
                artifact=manifest_path,
                operation="validate project contract",
            )
    expected.persist(manifest_path)
    return expected


def normalize_control_observations(
    coordinates: Iterable[Sequence[float]],
    input_crs,
    anchor: TopocentricAnchor,
    *,
    declared_vertical: bool,
    source: ControlSource = ControlSource.GCP,
):
    points = _as_points(coordinates, "normalize control")
    try:
        crs = CRS.from_user_input(input_crs)
    except Exception as error:
        raise CoordinateContractError(
            "invalid input CRS definition",
            operation="normalize control",
            coordinate=points[0],
            cause=error,
        ) from error
    _validate_static_crs(crs)
    assumptions = []
    if declared_vertical and len(crs.axis_info) < 3:
        raise CoordinateContractError("declared vertical control requires a three-dimensional CRS")
    normalized_input = points.copy()
    source = ControlSource(source)
    if source == ControlSource.GPS:
        assumptions.append("GPS altitude is WGS 84 ellipsoidal height in metres")
    elif declared_vertical:
        assumptions.append("declared vertical CRS component honored during control normalization")
    else:
        if crs.is_projected:
            normalized_input[:, 2] *= crs.axis_info[0].unit_conversion_factor
            assumptions.append("undeclared GCP height uses the input horizontal CRS linear unit as ellipsoidal height")
        else:
            assumptions.append("undeclared GCP height uses legacy ellipsoidal metres for a geographic input CRS")
    try:
        group = TransformerGroup(crs, WGS84_ECEF, always_xy=True, allow_ballpark=False)
        if not group.best_available or not group.transformers:
            missing = [grid.short_name for operation in group.unavailable_operations for grid in operation.grids if not grid.available]
            if not missing and not group.transformers:
                raise CoordinateContractError(
                    "only ballpark coordinate operations are available"
                )
            raise CoordinateContractError(
                "required transformation grids are unavailable: {}".format(", ".join(sorted(set(missing))) or "unknown")
            )
        transformer = Transformer.from_crs(
            crs, WGS84_ECEF, always_xy=True, allow_ballpark=False, only_best=True
        )
        if _requires_coordinate_epoch(transformer):
            raise CoordinateContractError(
                "epoch-dependent coordinate operation is unsupported",
                operation="normalize control",
            )
        ecef = np.column_stack(transformer.transform(*normalized_input.T, errcheck=True))
        geographic = Transformer.from_crs(WGS84_ECEF, WGS84_GEOGRAPHIC_3D, always_xy=True)
        lon, lat, _ = geographic.transform(*ecef.T, errcheck=True)
        warnings = ()
        operation_area = transformer.area_of_use or crs.area_of_use
        if operation_area is not None:
            bounds = operation_area.bounds
            any_inside, all_inside = _area_coverage(bounds, lon, lat)
            if not any_inside:
                raise CoordinateContractError(
                    "control dataset is wholly outside the input operation area",
                    operation="normalize control",
                    coordinate=points[0],
                )
            if not all_inside:
                warnings = ("control dataset crosses the input operation area boundary",)
        topocentric = Transformer.from_pipeline(
            "+proj=topocentric +ellps=WGS84 +lat_0={:.15g} +lon_0={:.15g} +h_0={:.15g}".format(
                anchor.latitude, anchor.longitude, anchor.ellipsoidal_height
            )
        )
        result = np.column_stack(topocentric.transform(*ecef.T, errcheck=True))
    except GeoreferencingError:
        raise
    except Exception as error:
        raise CoordinateContractError(
            "control normalization failed",
            operation="normalize control",
            coordinate=points[0],
            cause=error,
        ) from error
    if not np.all(np.isfinite(result)):
        raise CoordinateContractError(
            "non-finite transformation result",
            operation="normalize control",
            coordinate=points[0],
        )
    provenance = ControlProvenance(
        input_crs_wkt=crs.to_wkt(),
        source=source,
        operation=transformer.definition,
        grids=tuple(
            grid.short_name
            for operation in transformer.operations
            for grid in operation.grids
        ),
        accuracy=transformer.accuracy,
        axis_order=("easting", "northing", "height"),
        height_assumptions=tuple(assumptions),
        warnings=warnings,
        versions=(
            ("pyproj", pyproj.__version__),
            ("proj", pyproj.proj_version_str),
        ),
    )
    return NormalizedControlSet.from_coordinates(
        result,
        vertical_control=True,
        provenance=(provenance,),
    )
