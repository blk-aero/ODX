import os
import shutil
import struct
import pipes
import fiona
import fiona.crs
import json
import zipfile
import numpy as np
from collections import OrderedDict
from pyproj import CRS, Transformer

from opendm import io
from opendm import log
from opendm import types
from opendm import system
from opendm import context
from opendm.cropper import Cropper
from opendm import point_cloud
from opendm.gcp import GCPFile
from opendm.georeferencing import (
    ControlSource,
    coordinate_contract_metadata,
    MeshExportError,
    NormalizedControlSet,
    export_georeferenced_point_cloud,
    export_georeferenced_mesh,
    load_topocentric_anchor,
    load_output_selection,
    normalize_control_observations,
    materialize_canonical_reconstruction,
    publish_legacy_reconstruction_compatibility,
    resolve_stage_coordinate_contract,
    resolve_output_selection,
)
from opendm.multispectral import get_primary_band_name
from opendm.osfm import OSFMContext
from opendm.boundary import (
    as_polygon,
    export_to_bounds_files,
    write_coordinate_contract_tags,
)
from opendm.align import compute_alignment_matrix, transform_point_cloud, transform_obj
from opendm.utils import np_from_json, np_to_json


def _export_ground_control_points(tree, reconstruction, contract):
    gcp_export_file = tree.path("odm_georeferencing", "ground_control_points.gpkg")
    gcp_gml_export_file = tree.path("odm_georeferencing", "ground_control_points.gml")
    gcp_geojson_export_file = tree.path("odm_georeferencing", "ground_control_points.geojson")
    gcp_geojson_zip_export_file = tree.path("odm_georeferencing", "ground_control_points.zip")
    gcps = OSFMContext(tree.opensfm).ground_control_points(
        reconstruction.georef.proj4(), coordinate_contract=contract
    )
    if not gcps:
        log.WARNING("GCPs could not be loaded for writing to %s" % gcp_export_file)
        return

    gcp_schema = {
        'geometry': 'Point',
        'properties': OrderedDict([
            ('id', 'str'),
            ('observations_count', 'int'),
            ('observations_list', 'str'),
            ('error_x', 'float'),
            ('error_y', 'float'),
            ('error_z', 'float'),
        ])
    }
    with fiona.open(gcp_export_file, 'w', driver="GPKG",
                    crs_wkt=contract.output_crs_wkt, schema=gcp_schema) as output:
        for gcp in gcps:
            output.write({
                'geometry': {'type': 'Point', 'coordinates': gcp['coordinates']},
                'properties': OrderedDict([
                    ('id', gcp['id']),
                    ('observations_count', len(gcp['observations'])),
                    ('observations_list', ",".join(
                        observation['shot_id'] for observation in gcp['observations']
                    )),
                    ('error_x', gcp['error'][0]),
                    ('error_y', gcp['error'][1]),
                    ('error_z', gcp['error'][2]),
                ])
            })
        write_coordinate_contract_tags(
            output, coordinate_contract_metadata(contract)
        )

    try:
        with fiona.open(gcp_export_file, 'r') as source:
            with fiona.open(gcp_gml_export_file, 'w', driver='GML',
                            crs_wkt=source.crs_wkt, schema=source.schema) as output:
                for feature in source:
                    output.write(feature)
                write_coordinate_contract_tags(
                    output, coordinate_contract_metadata(contract)
                )
    except Exception as error:
        log.WARNING("Cannot generate ground control points GML file: %s" % error)

    geographic = Transformer.from_crs(
        contract.output_crs, CRS.from_epsg(4979), always_xy=True
    )
    geojson = {
        'type': 'FeatureCollection',
        'features': [],
        'coordinate_contract': coordinate_contract_metadata(contract),
    }
    for gcp in gcps:
        properties = gcp.copy()
        del properties['coordinates']
        geojson['features'].append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': list(geographic.transform(*gcp['coordinates'])),
            },
            'properties': properties,
        })
    with open(gcp_geojson_export_file, 'w') as output:
        output.write(json.dumps(geojson, indent=4))
    with zipfile.ZipFile(gcp_geojson_zip_export_file, 'w',
                         compression=zipfile.ZIP_LZMA) as archive:
        archive.write(
            gcp_geojson_export_file,
            arcname=os.path.basename(gcp_geojson_export_file),
        )


def _project_controls(tree, reconstruction, args, anchor):
    controls = NormalizedControlSet.from_coordinates(
        [(0.0, 0.0, 0.0)], vertical_control=False
    )
    observations = []
    input_crs = None
    control_source = None
    declared_vertical = False
    source_gcp = GCPFile(tree.odm_georeferencing_gcp)
    if source_gcp.exists() and not args.use_exif:
        observations = sorted({
            (entry.x, entry.y, entry.z) for entry in source_gcp.iter_entries()
        })
        if observations:
            input_crs = source_gcp.srs
            declared_vertical = len(input_crs.axis_info) >= 3
            control_source = ControlSource.GCP
    else:
        observations = sorted({
            (photo.longitude, photo.latitude, photo.altitude)
            for photo in reconstruction.photos
            if None not in (
                getattr(photo, "longitude", None),
                getattr(photo, "latitude", None),
                getattr(photo, "altitude", None),
            )
        })
        if observations:
            input_crs = CRS.from_epsg(4979)
            declared_vertical = True
            control_source = ControlSource.GPS
    if observations:
        controls = normalize_control_observations(
            observations, input_crs, anchor,
            declared_vertical=declared_vertical, source=control_source,
        )
    return controls


def resolve_project_output_selection(tree, reconstruction, args):
    """Create once or reload the generated parent split metadata."""
    selection_path = tree.path("odm_georeferencing", "output_selection.json")
    if os.path.isfile(selection_path):
        return load_output_selection(selection_path)
    reference_path = tree.path("opensfm", "reference_lla.json")
    anchor = load_topocentric_anchor(reference_path)
    controls = _project_controls(tree, reconstruction, args, anchor)
    selection = resolve_output_selection(
        anchor, controls, storage_offset=reconstruction.get_proj_offset()
    )
    selection.persist(selection_path)
    return selection


def resolve_project_coordinate_contract(tree, reconstruction, args):
    """Resolve the persisted job contract shared by working and export stages."""
    reference_path = tree.path("opensfm", "reference_lla.json")
    anchor = load_topocentric_anchor(reference_path)
    controls = _project_controls(tree, reconstruction, args, anchor)
    selection_path = tree.path("odm_georeferencing", "output_selection.json")
    output_selection = (
        load_output_selection(selection_path)
        if os.path.isfile(selection_path)
        else None
    )
    fresh_marker = tree.path("opensfm", "fresh_reconstruction.marker")
    return resolve_stage_coordinate_contract(
        tree.path("odm_georeferencing", "coordinate_contract.json"),
        reference_path,
        reconstruction.get_proj_offset(),
        controls=controls,
        fresh_reconstruction=os.path.isfile(fresh_marker),
        output_selection=output_selection,
    )


def _resolve_alignment_matrix(tree, rerun, input_laz, prepare_input=None):
    """Load or compute registration through ODX's existing alignment workflow."""
    stats_dir = tree.path("opensfm", "stats", "codem")
    if os.path.exists(stats_dir) and rerun:
        shutil.rmtree(stats_dir)
    matrix_path = tree.odm_georeferencing_alignment_matrix
    if io.file_exists(matrix_path) and not rerun:
        with open(matrix_path, "r") as matrix_file:
            alignment_matrix = np_from_json(matrix_file.read())
        log.INFO("Reusing alignment matrix: %s" % alignment_matrix)
        return alignment_matrix
    if io.file_exists(matrix_path):
        os.unlink(matrix_path)
    if prepare_input is not None:
        prepare_input()
    return compute_alignment_matrix(input_laz, tree.odm_align_file, stats_dir)


class ODMGeoreferencingStage(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        if reconstruction.is_georeferenced():
            coordinate_contract_path = tree.path(
                "odm_georeferencing", "coordinate_contract.json"
            )
            fresh_reconstruction_marker = tree.path("opensfm", "fresh_reconstruction.marker")
            outputs["coordinate_contract"] = resolve_project_coordinate_contract(
                tree, reconstruction, args
            )
            if os.path.isfile(fresh_reconstruction_marker):
                os.unlink(fresh_reconstruction_marker)
            if (
                io.file_exists(tree.opensfm_reconstruction)
                or io.file_exists(tree.opensfm_topocentric_reconstruction)
            ):
                materialize_canonical_reconstruction(
                    tree.opensfm_reconstruction,
                    tree.opensfm_topocentric_reconstruction,
                )

        # Export GCP information if available

        gcp_export_file = tree.path("odm_georeferencing", "ground_control_points.gpkg")
        gcp_geojson_zip_export_file = tree.path("odm_georeferencing", "ground_control_points.zip")
        gcp_exported = False
        unaligned_model = io.related_file_path(tree.odm_georeferencing_model_laz, postfix="_unaligned")
        if os.path.isfile(unaligned_model) and self.rerun():
            os.unlink(unaligned_model)

        def textured_model_paths():
            for texturing in [tree.odm_texturing, tree.odm_25dtexturing]:
                if reconstruction.multi_camera:
                    primary = get_primary_band_name(reconstruction.multi_camera, args.primary_band)
                    subdirectories = [
                        "" if band['name'] == primary else band['name'].lower()
                        for band in reconstruction.multi_camera
                    ]
                else:
                    subdirectories = [""]
                for subdirectory in subdirectories:
                    yield os.path.join(
                        texturing, subdirectory, tree.odm_textured_model_obj
                    )

        def publish_bounds(contract, point_cloud_result):
            if contract.post_crs_alignment is not None and outputs.get(
                "topocentric_boundary"
            ):
                # The boundary selected canonical geometry before registration;
                # publish that same geometry through the finalized aligned contract.
                topocentric = np.asarray(
                    outputs["topocentric_boundary"], dtype=np.float64
                )
                topocentric_3d = np.column_stack((
                    topocentric[:, :2], np.zeros(len(topocentric))
                ))
                outputs["boundary"] = contract.transform_points(
                    topocentric_3d, apply_storage_offset=False
                )[:, :2].tolist()

            bounds = getattr(point_cloud_result, "bounds", None)
            decoded_extent = None
            if bounds is not None:
                minimum, maximum = bounds.minimum, bounds.maximum
                decoded_extent = [
                    [minimum[0], minimum[1]],
                    [maximum[0], minimum[1]],
                    [maximum[0], maximum[1]],
                    [minimum[0], maximum[1]],
                    [minimum[0], minimum[1]],
                ]
                extent_base, _ = os.path.splitext(
                    tree.odm_georeferencing_model_laz
                )
                export_to_bounds_files(
                    decoded_extent,
                    contract.output_crs.to_proj4(),
                    extent_base + ".extent.geojson",
                    extent_base + ".extent.gpkg",
                    coordinate_contract=coordinate_contract_metadata(contract),
                )

            if args.crop > 0:
                log.INFO("Calculating cropping area and generating bounds shapefile from point cloud")
                cropper = Cropper(tree.odm_georeferencing, 'odm_georeferenced_model')
                decimation_step = 4 if args.fast_orthophoto else 40
                if not args.fast_orthophoto:
                    decimation_step *= int(len(reconstruction.photos) / 1000) + 1
                    decimation_step = min(decimation_step, 95)
                try:
                    bounds_gpkg = cropper.create_bounds_gpkg(
                        tree.odm_georeferencing_model_laz,
                        args.crop,
                        decimation_step=decimation_step,
                    )
                    if (
                        isinstance(bounds_gpkg, (str, bytes, os.PathLike))
                        and os.path.isfile(bounds_gpkg)
                    ):
                        metadata = coordinate_contract_metadata(contract)
                        bounds_geojson = os.path.splitext(bounds_gpkg)[0] + ".geojson"
                        with open(bounds_geojson, "r") as source:
                            collection = json.load(source)
                        collection["coordinate_contract"] = metadata
                        with open(bounds_geojson, "w") as output:
                            output.write(json.dumps(collection))
                        with fiona.open(bounds_gpkg, "a") as output:
                            write_coordinate_contract_tags(output, metadata)
                    return
                except Exception:
                    log.WARNING("Cannot calculate crop bounds! We will skip cropping")
                    args.crop = 0

            boundary = outputs.get("boundary") or decoded_extent
            if boundary is not None:
                log.INFO("Publishing output-CRS boundary and decoded bounds")
                bounds_base, _ = os.path.splitext(tree.odm_georeferencing_model_laz)
                export_to_bounds_files(
                    boundary,
                    contract.output_crs.to_proj4(),
                    bounds_base + ".bounds.geojson",
                    bounds_base + ".bounds.gpkg",
                    coordinate_contract=coordinate_contract_metadata(contract),
                )

        if not io.file_exists(tree.odm_georeferencing_model_laz) or self.rerun():
            cmd = f'pdal translate -i "{tree.filtered_point_cloud}" -o \"{tree.odm_georeferencing_model_laz}\"'
            stages = ["ferry"]
            params = [
                '--filters.ferry.dimensions="views => UserData"',
                '--writers.las.minor_version=2',
                '--writers.las.dataformat_id=3',
            ]

            if reconstruction.is_georeferenced():
                log.INFO("Georeferencing point cloud")

                # Establish appropriate las scale for export
                point_spacing = None
                filtered_point_cloud_stats = tree.path("odm_filterpoints", "point_cloud_stats.json")

                if os.path.isfile(filtered_point_cloud_stats):
                    try:
                        with open(filtered_point_cloud_stats, 'r') as stats:
                             las_stats = json.load(stats)
                             point_spacing = las_stats['spacing']
                             log.INFO("LAS scale uses the minimum of the spacing-derived value and 0.001 metre")
                    except Exception as e:
                        log.WARNING("Cannot read point_cloud_stats.json. Using default LAS scale: 0.001")
                else:
                    log.INFO("No point_cloud_stats.json found. Using default LAS scale: 0.001")

                point_cloud_contract = outputs["coordinate_contract"].without_post_crs_alignment()
                persisted_alignment = outputs["coordinate_contract"].post_crs_alignment
                alignment_requested = (
                    tree.odm_align_file is not None or persisted_alignment is not None
                )
                aligned_contract = outputs["coordinate_contract"]
                alignment_matrix = None
                alignment_input = io.related_file_path(
                    tree.odm_georeferencing_model_laz, postfix="_alignment_input"
                )

                if alignment_requested:
                    try:
                        if persisted_alignment is not None:
                            alignment_matrix = np.asarray(persisted_alignment).reshape(4, 4)
                            log.INFO("Reusing persisted alignment matrix: %s" % alignment_matrix)
                        else:
                            def prepare_alignment_input():
                                export_georeferenced_point_cloud(
                                    tree.filtered_point_cloud,
                                    alignment_input,
                                    point_cloud_contract,
                                    spacing=point_spacing,
                                )

                            alignment_matrix = _resolve_alignment_matrix(
                                tree, self.rerun(), alignment_input,
                                prepare_input=prepare_alignment_input,
                            )
                        if alignment_matrix is not None:
                            aligned_contract = (
                                outputs["coordinate_contract"]
                                if persisted_alignment is not None
                                else point_cloud_contract.with_post_crs_alignment(alignment_matrix)
                            )
                            log.INFO("Alignment matrix: %s" % alignment_matrix)
                    except Exception as error:
                        alignment_matrix = None
                        aligned_contract = point_cloud_contract
                        log.WARNING("Cannot compute alignment matrix: %s" % error)
                    finally:
                        if os.path.isfile(alignment_input):
                            os.unlink(alignment_input)

                    if alignment_matrix is None:
                        log.WARNING("Alignment to %s will be skipped." % tree.odm_align_file)

                outputs["coordinate_contract"] = aligned_contract
                if reconstruction.has_gcp():
                    _export_ground_control_points(tree, reconstruction, aligned_contract)
                    gcp_exported = True

                point_cloud_vlrs = None
                if reconstruction.has_gcp() and io.file_exists(gcp_geojson_zip_export_file):
                    if os.path.getsize(gcp_geojson_zip_export_file) <= 65535:
                        log.INFO("Embedding GCP info in point cloud")
                        point_cloud_vlrs = [{
                            "filename": gcp_geojson_zip_export_file.replace(os.sep, "/"),
                            "user_id": "ODX",
                            "record_id": 2,
                            "description": "Ground Control Points (zip)",
                        }]
                    else:
                        log.WARNING("Cannot embed GCP info in point cloud, %s is too large" % gcp_geojson_zip_export_file)

                if alignment_matrix is not None:
                    export_georeferenced_point_cloud(
                        tree.filtered_point_cloud, unaligned_model,
                        point_cloud_contract, spacing=point_spacing,
                        vlrs=point_cloud_vlrs,
                    )
                elif os.path.isfile(unaligned_model):
                    os.unlink(unaligned_model)

                result = export_georeferenced_point_cloud(
                    tree.filtered_point_cloud,
                    tree.odm_georeferencing_model_laz,
                    aligned_contract,
                    spacing=point_spacing,
                    vlrs=point_cloud_vlrs,
                )
                log.INFO(
                    "Exported %s exact points with LAS scale %s and offset %s"
                    % (result.point_count, result.encoding.scale, result.encoding.offset)
                )
                outputs["decoded_georeferenced_bounds"] = getattr(result, "bounds", None)
                publish_bounds(aligned_contract, result)

                if alignment_matrix is not None:
                    aligned_contract.persist(coordinate_contract_path)
                    with open(tree.odm_georeferencing_alignment_matrix, "w") as f:
                        f.write(np_to_json(alignment_matrix))
                elif io.file_exists(tree.odm_georeferencing_alignment_matrix):
                    os.unlink(tree.odm_georeferencing_alignment_matrix)
                self.update_progress(50)
            else:
                log.INFO("Converting point cloud (non-georeferenced)")
                system.run(cmd + ' ' + ' '.join(stages) + ' ' + ' '.join(params))

            if not reconstruction.is_georeferenced() and tree.odm_align_file is not None:
                try:
                    alignment_matrix = _resolve_alignment_matrix(
                        tree, self.rerun(), tree.odm_georeferencing_model_laz
                    )
                except Exception as error:
                    alignment_matrix = None
                    log.WARNING("Cannot compute alignment matrix: %s" % error)

                if alignment_matrix is not None:
                    log.INFO("Alignment matrix: %s" % alignment_matrix)

                    # Preserve legacy affine alignment for non-georeferenced jobs.
                    if os.path.isfile(unaligned_model):
                        os.rename(unaligned_model, tree.odm_georeferencing_model_laz)
                    os.rename(tree.odm_georeferencing_model_laz, unaligned_model)

                    try:
                        transform_point_cloud(unaligned_model, alignment_matrix, tree.odm_georeferencing_model_laz)
                        log.INFO("Transformed %s" % tree.odm_georeferencing_model_laz)
                    except Exception as e:
                        log.WARNING("Cannot transform point cloud: %s" % str(e))
                        os.rename(unaligned_model, tree.odm_georeferencing_model_laz)

                    def transform_textured_model(obj):
                        if os.path.isfile(obj):
                            unaligned_obj = io.related_file_path(obj, postfix="_unaligned")
                            if os.path.isfile(unaligned_obj):
                                os.rename(unaligned_obj, obj)
                            os.rename(obj, unaligned_obj)
                            try:
                                transform_obj(unaligned_obj, alignment_matrix, [reconstruction.georef.utm_east_offset, reconstruction.georef.utm_north_offset], obj)
                                log.INFO("Transformed %s" % obj)
                            except Exception as e:
                                log.WARNING("Cannot transform textured model: %s" % str(e))
                                os.rename(unaligned_obj, obj)

                    for obj in textured_model_paths():
                        transform_textured_model(obj)

                    with open(tree.odm_georeferencing_alignment_matrix, "w") as f:
                        f.write(np_to_json(alignment_matrix))
                else:
                    log.WARNING("Alignment to %s will be skipped." % tree.odm_align_file)
            elif (
                not reconstruction.is_georeferenced()
                and io.file_exists(tree.odm_georeferencing_alignment_matrix)
            ):
                os.unlink(tree.odm_georeferencing_alignment_matrix)

            point_cloud.post_point_cloud_steps(args, tree, self.rerun())
        else:
            log.WARNING('Found a valid georeferenced model in: %s'
                            % tree.odm_georeferencing_model_laz)

        if not gcp_exported and reconstruction.has_gcp() and (
            not io.file_exists(gcp_export_file) or self.rerun()
        ):
            _export_ground_control_points(
                tree, reconstruction, outputs["coordinate_contract"]
            )

        if reconstruction.is_georeferenced():
            for public_obj in textured_model_paths():
                topocentric_obj = io.related_file_path(
                    public_obj, postfix="_topocentric"
                )
                try:
                    if os.path.isfile(topocentric_obj):
                        result = export_georeferenced_mesh(
                            topocentric_obj, public_obj, outputs["coordinate_contract"]
                        )
                        log.INFO(
                            "Exported exact textured mesh %s (%s vertices, %s faces)"
                            % (public_obj, result.vertex_count, result.face_count)
                        )
                    elif os.path.isfile(public_obj):
                        raise MeshExportError(
                            "canonical topocentric mesh is missing; rerun from mvs_texturing",
                            artifact=public_obj,
                            operation="select canonical mesh",
                        )
                except MeshExportError as error:
                    # Textured models are optional today. Do not publish the
                    # topocentric working file as an approximate substitute.
                    if os.path.isfile(public_obj):
                        os.unlink(public_obj)
                    log.WARNING("Cannot export exact textured model: %s" % error)

            if (
                io.file_exists(tree.opensfm_reconstruction)
                or io.file_exists(tree.opensfm_topocentric_reconstruction)
            ):
                contract = outputs["coordinate_contract"]
                projection = contract.output_crs.to_proj4()
                offset_x, offset_y = contract.storage_offset[:2]
                octx = OSFMContext(tree.opensfm)

                def stock_export():
                    octx.run(
                        'export_geocoords --reconstruction --proj "%s" '
                        '--offset-x %s --offset-y %s'
                        % (projection, offset_x, offset_y)
                    )

                publish_legacy_reconstruction_compatibility(
                    tree.opensfm_reconstruction,
                    tree.opensfm_topocentric_reconstruction,
                    tree.opensfm_geocoords_reconstruction,
                    tree.opensfm_compatibility_provenance,
                    contract,
                    stock_export,
                )

        if args.optimize_disk_space and io.file_exists(tree.odm_georeferencing_model_laz) and io.file_exists(tree.filtered_point_cloud):
            os.remove(tree.filtered_point_cloud)
