import os
import math
import shutil
import struct
import pipes
import fiona
import fiona.crs
import json
import zipfile
from collections import OrderedDict
from pyproj import CRS

from opendm import io
from opendm import log
from opendm import types
from opendm import system
from opendm import context
from opendm import location
from opendm.cropper import Cropper
from opendm import point_cloud
from opendm.georeferencing import (
    VerticalReference,
    export_georeferenced_mesh,
    export_georeferenced_point_cloud,
    materialize_canonical_reconstruction,
    publish_legacy_reconstruction_compatibility,
    resolve_stage_coordinate_contract,
)
from opendm.multispectral import get_photos_by_band, get_primary_band_name
from opendm.osfm import OSFMContext, is_submodel
from opendm.boundary import as_polygon, export_to_bounds_files
from opendm.align import compute_alignment_matrix, transform_point_cloud, transform_obj
from opendm.utils import np_to_json


def _vertical_reference(reconstruction, args):
    has_gcp = reconstruction.has_gcp()
    if has_gcp and any(
        not entry.is_checkpoint() and not math.isnan(entry.z)
        for entry in reconstruction.gcp.iter_entries()
    ):
        return VerticalReference.WGS84_ELLIPSOIDAL

    if not has_gcp or getattr(args, "force_gps", False):
        photos = (
            get_photos_by_band(reconstruction.multi_camera, args.primary_band)
            if reconstruction.multi_camera
            else reconstruction.photos
        )
        if photos and all(photo.altitude is not None for photo in photos):
            zero_altitudes = sum(photo.altitude == 0 for photo in photos)
            if zero_altitudes / len(photos) <= 0.05:
                return VerticalReference.WGS84_ELLIPSOIDAL

    return VerticalReference.UNREFERENCED


class ODMGeoreferencingStage(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        coordinate_contract_path = tree.path(
            "odm_georeferencing", "coordinate_contract.json"
        )
        if reconstruction.is_georeferenced() and (
            tree.odm_align_file is None and not is_submodel(tree.opensfm)
        ):
            try:
                materialize_canonical_reconstruction(
                    tree.opensfm_reconstruction,
                    tree.opensfm_topocentric_reconstruction,
                )
                outputs["coordinate_contract"] = resolve_stage_coordinate_contract(
                    coordinate_contract_path,
                    tree.path("opensfm", "reference_lla.json"),
                    reconstruction.get_proj_offset(),
                    fresh_reconstruction=bool(outputs.get("fresh_reconstruction")),
                    vertical_reference=_vertical_reference(reconstruction, args),
                )
            except Exception as error:
                log.WARNING("Exact georeferencing unavailable; using stock export: %s" % error)

        def textured_model_paths():
            if reconstruction.multi_camera:
                primary = get_primary_band_name(
                    reconstruction.multi_camera, args.primary_band
                )
                bands = [
                    (band['name'] == primary, band['name'].lower())
                    for band in reconstruction.multi_camera
                ]
            else:
                bands = [(True, "")]
            for is_primary, subdirectory in bands:
                if not args.skip_3dmodel and (is_primary or args.use_3dmesh):
                    yield os.path.join(
                        tree.odm_texturing,
                        subdirectory,
                        tree.odm_textured_model_obj,
                    )
                if not args.use_3dmesh:
                    yield os.path.join(
                        tree.odm_25dtexturing,
                        subdirectory,
                        tree.odm_textured_model_obj,
                    )

        # Export GCP information if available

        gcp_export_file = tree.path("odm_georeferencing", "ground_control_points.gpkg")
        gcp_gml_export_file = tree.path("odm_georeferencing", "ground_control_points.gml")
        gcp_geojson_export_file = tree.path("odm_georeferencing", "ground_control_points.geojson")
        gcp_geojson_zip_export_file = tree.path("odm_georeferencing", "ground_control_points.zip")
        unaligned_model = io.related_file_path(tree.odm_georeferencing_model_laz, postfix="_unaligned")
        if os.path.isfile(unaligned_model) and self.rerun():
            os.unlink(unaligned_model)

        if reconstruction.has_gcp() and (not io.file_exists(gcp_export_file) or self.rerun()):
            octx = OSFMContext(tree.opensfm)
            gcps = octx.ground_control_points(reconstruction.georef.proj4())

            if len(gcps):
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

                # Write GeoPackage
                with fiona.open(gcp_export_file, 'w', driver="GPKG",
                                crs=fiona.crs.from_string(reconstruction.georef.proj4()),
                                schema=gcp_schema) as f:
                    for gcp in gcps:
                        f.write({
                            'geometry': {
                                'type': 'Point',
                                'coordinates': gcp['coordinates'],
                            },
                            'properties': OrderedDict([
                                ('id', gcp['id']),
                                ('observations_count', len(gcp['observations'])),
                                ('observations_list', ",".join([obs['shot_id'] for obs in gcp['observations']])),
                                ('error_x', gcp['error'][0]),
                                ('error_y', gcp['error'][1]),
                                ('error_z', gcp['error'][2]),
                            ])
                        })

                # Write GML
                try:
                    with fiona.open(gcp_export_file, 'r') as src:
                        with fiona.open(gcp_gml_export_file, 'w', driver='GML', crs=src.crs, schema=src.schema) as dst:
                            for feature in src:
                                dst.write(feature)
                except Exception as e:
                    log.WARNING("Cannot generate ground control points GML file: %s" % str(e))

                # Write GeoJSON
                geojson = {
                    'type': 'FeatureCollection',
                    'features': []
                }

                from_srs = CRS.from_proj4(reconstruction.georef.proj4())
                to_srs = CRS.from_epsg(4326)
                transformer = location.transformer(from_srs, to_srs)

                for gcp in gcps:
                    properties = gcp.copy()
                    del properties['coordinates']

                    geojson['features'].append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': transformer.TransformPoint(*gcp['coordinates']),
                        },
                        'properties': properties
                    })

                with open(gcp_geojson_export_file, 'w') as f:
                    f.write(json.dumps(geojson, indent=4))
                
                with zipfile.ZipFile(gcp_geojson_zip_export_file, 'w', compression=zipfile.ZIP_LZMA) as f:
                    f.write(gcp_geojson_export_file, arcname=os.path.basename(gcp_geojson_export_file))

            else:
                log.WARNING("GCPs could not be loaded for writing to %s" % gcp_export_file)

        if (
            not io.file_exists(tree.odm_georeferencing_model_laz)
            or self.rerun()
            or outputs.get("fresh_reconstruction")
        ):
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

                exact_exported = False
                contract = outputs.get("coordinate_contract")
                if contract is not None:
                    try:
                        result = export_georeferenced_point_cloud(
                            tree.filtered_point_cloud,
                            tree.odm_georeferencing_model_laz,
                            contract,
                            spacing=point_spacing,
                            vlrs=point_cloud_vlrs,
                        )
                        log.INFO("Exported %s exact points" % result.point_count)
                        exact_exported = True
                    except Exception as error:
                        log.WARNING("Exact point export unavailable; using stock export: %s" % error)

                if not exact_exported:
                    utmoffset = reconstruction.georef.utm_offset()
                    las_scale = 0.001
                    if point_spacing is not None:
                        las_scale = min(pow(10, round(math.log10(point_spacing))) / 10, las_scale)
                    stages.append("transformation")
                    params += [
                        '--filters.transformation.matrix="1 0 0 %s 0 1 0 %s 0 0 1 0 0 0 0 1"' % utmoffset,
                        '--writers.las.offset_x=%s' % reconstruction.georef.utm_east_offset,
                        '--writers.las.offset_y=%s' % reconstruction.georef.utm_north_offset,
                        '--writers.las.scale_x=%s' % las_scale,
                        '--writers.las.scale_y=%s' % las_scale,
                        '--writers.las.scale_z=%s' % las_scale,
                        '--writers.las.offset_z=0',
                        '--writers.las.a_srs="%s"' % reconstruction.georef.proj4(),
                    ]
                    if point_cloud_vlrs:
                        params.append('--writers.las.vlrs="%s"' % json.dumps(point_cloud_vlrs))
                    system.run(cmd + ' ' + ' '.join(stages) + ' ' + ' '.join(params))

                self.update_progress(50)

                if args.crop > 0:
                    log.INFO("Calculating cropping area and generating bounds shapefile from point cloud")
                    cropper = Cropper(tree.odm_georeferencing, 'odm_georeferenced_model')

                    if args.fast_orthophoto:
                        decimation_step = 4
                    else:
                        decimation_step = 40

                    # More aggressive decimation for large datasets
                    if not args.fast_orthophoto:
                        decimation_step *= int(len(reconstruction.photos) / 1000) + 1
                        decimation_step = min(decimation_step, 95)

                    try:
                        cropper.create_bounds_gpkg(tree.odm_georeferencing_model_laz, args.crop,
                                                    decimation_step=decimation_step)
                    except:
                        log.WARNING("Cannot calculate crop bounds! We will skip cropping")
                        args.crop = 0

                if 'boundary' in outputs and args.crop == 0:
                    log.INFO("Using boundary JSON as cropping area")

                    bounds_base, _ = os.path.splitext(tree.odm_georeferencing_model_laz)
                    bounds_json = bounds_base + ".bounds.geojson"
                    bounds_gpkg = bounds_base + ".bounds.gpkg"
                    export_to_bounds_files(outputs['boundary'], reconstruction.get_proj_srs(), bounds_json, bounds_gpkg)
            else:
                log.INFO("Converting point cloud (non-georeferenced)")
                system.run(cmd + ' ' + ' '.join(stages) + ' ' + ' '.join(params))


            stats_dir = tree.path("opensfm", "stats", "codem")
            if os.path.exists(stats_dir) and self.rerun():
                shutil.rmtree(stats_dir)

            if tree.odm_align_file is not None:
                alignment_file_exists = io.file_exists(tree.odm_georeferencing_alignment_matrix)

                if not alignment_file_exists or self.rerun():
                    if alignment_file_exists:
                        os.unlink(tree.odm_georeferencing_alignment_matrix)

                    a_matrix = None
                    try:
                        a_matrix = compute_alignment_matrix(tree.odm_georeferencing_model_laz, tree.odm_align_file, stats_dir)
                    except Exception as e:
                        log.WARNING("Cannot compute alignment matrix: %s" % str(e))

                    if a_matrix is not None:
                        log.INFO("Alignment matrix: %s" % a_matrix)

                        # Align point cloud
                        if os.path.isfile(unaligned_model):
                            os.rename(unaligned_model, tree.odm_georeferencing_model_laz)
                        os.rename(tree.odm_georeferencing_model_laz, unaligned_model)

                        try:
                            transform_point_cloud(unaligned_model, a_matrix, tree.odm_georeferencing_model_laz)
                            log.INFO("Transformed %s" % tree.odm_georeferencing_model_laz)
                        except Exception as e:
                            log.WARNING("Cannot transform point cloud: %s" % str(e))
                            os.rename(unaligned_model, tree.odm_georeferencing_model_laz)

                        # Align textured models
                        def transform_textured_model(obj):
                            if os.path.isfile(obj):
                                unaligned_obj = io.related_file_path(obj, postfix="_unaligned")
                                if os.path.isfile(unaligned_obj):
                                    os.rename(unaligned_obj, obj)
                                os.rename(obj, unaligned_obj)
                                try:
                                    transform_obj(unaligned_obj, a_matrix, [reconstruction.georef.utm_east_offset, reconstruction.georef.utm_north_offset], obj)
                                    log.INFO("Transformed %s" % obj)
                                except Exception as e:
                                    log.WARNING("Cannot transform textured model: %s" % str(e))
                                    os.rename(unaligned_obj, obj)

                        for texturing in [tree.odm_texturing, tree.odm_25dtexturing]:
                            if reconstruction.multi_camera:
                                primary = get_primary_band_name(reconstruction.multi_camera, args.primary_band)
                                for band in reconstruction.multi_camera:
                                    subdir = "" if band['name'] == primary else band['name'].lower()
                                    obj = os.path.join(texturing, subdir, "odm_textured_model_geo.obj")
                                    transform_textured_model(obj)
                            else:
                                obj = os.path.join(texturing, "odm_textured_model_geo.obj")
                                transform_textured_model(obj)

                        with open(tree.odm_georeferencing_alignment_matrix, "w") as f:
                            f.write(np_to_json(a_matrix))
                    else:
                        log.WARNING("Alignment to %s will be skipped." % tree.odm_align_file)
                else:
                    log.WARNING("Already computed alignment")
            elif io.file_exists(tree.odm_georeferencing_alignment_matrix):
                os.unlink(tree.odm_georeferencing_alignment_matrix)

            point_cloud.post_point_cloud_steps(args, tree, self.rerun())
        else:
            log.WARNING('Found a valid georeferenced model in: %s'
                            % tree.odm_georeferencing_model_laz)

        contract = outputs.get("coordinate_contract")
        if reconstruction.is_georeferenced() and contract is not None:
            for public_obj in textured_model_paths():
                topocentric_obj = io.related_file_path(
                    public_obj, postfix="_topocentric"
                )
                try:
                    result = export_georeferenced_mesh(
                        topocentric_obj,
                        public_obj,
                        contract,
                    )
                    log.INFO(
                        "Exported exact textured mesh %s (%s vertices, %s faces)"
                        % (public_obj, result.vertex_count, result.face_count)
                    )
                except Exception as error:
                    log.WARNING("Exact mesh export unavailable; keeping stock mesh: %s" % error)

            octx = OSFMContext(tree.opensfm)

            def stock_export():
                octx.run(
                    'export_geocoords --reconstruction --proj "%s" '
                    '--offset-x %s --offset-y %s'
                    % (
                        contract.output_crs.to_proj4(),
                        contract.storage_offset[0],
                        contract.storage_offset[1],
                    )
                )

            try:
                publish_legacy_reconstruction_compatibility(
                    tree.opensfm_reconstruction,
                    tree.opensfm_topocentric_reconstruction,
                    tree.opensfm_geocoords_reconstruction,
                    stock_export,
                )
            except Exception as error:
                log.WARNING("Could not publish stock reconstruction compatibility: %s" % error)
        elif reconstruction.is_georeferenced() and not outputs.get("stock_georeferenced_reconstruction"):
            octx = OSFMContext(tree.opensfm)
            try:
                octx.run(
                    'export_geocoords --reconstruction --proj "%s" '
                    '--offset-x %s --offset-y %s'
                    % (
                        reconstruction.georef.proj4(),
                        reconstruction.georef.utm_east_offset,
                        reconstruction.georef.utm_north_offset,
                    )
                )
                shutil.move(
                    tree.opensfm_geocoords_reconstruction,
                    tree.opensfm_reconstruction,
                )
            except Exception as error:
                log.WARNING("Could not restore stock reconstruction: %s" % error)

        if args.optimize_disk_space and io.file_exists(tree.odm_georeferencing_model_laz) and io.file_exists(tree.filtered_point_cloud):
            os.remove(tree.filtered_point_cloud)
