import os

from opendm import io
from opendm import log
from opendm import system
from opendm import context
from opendm import types
from opendm import gsd
from opendm import orthophoto
from opendm.osfm import is_submodel
from opendm.concurrency import get_max_memory_mb
from opendm.cutline import compute_cutline
from opendm.utils import double_quote
from opendm import pseudogeo
from opendm.multispectral import get_primary_band_name
from opendm.georeferencing import (
    CoordinateContractError,
    load_coordinate_contract,
    validate_canonical_reconstruction,
)


def _mesh_has_geometry(path):
    try:
        vertices = False
        faces = False
        with open(path, encoding="utf-8", errors="replace") as source:
            for line in source:
                fields = line.split()
                if not fields:
                    continue
                vertices = vertices or fields[0] == "v" and len(fields) >= 4
                faces = faces or fields[0] == "f" and len(fields) >= 4
                if vertices and faces:
                    return True
    except OSError:
        pass
    return False


class ODMOrthoPhotoStage(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        # define paths and create working directories
        system.mkdir_p(tree.odm_orthophoto)

        if args.skip_orthophoto:
            log.WARNING("--skip-orthophoto is set, no orthophoto will be generated")
            return

        if not io.file_exists(tree.odm_orthophoto_tif) or self.rerun():

            working_reconstruction = (
                tree.opensfm_topocentric_reconstruction
                if reconstruction.is_georeferenced()
                else tree.opensfm_reconstruction
            )
            if reconstruction.is_georeferenced():
                validate_canonical_reconstruction(working_reconstruction)
            resolution = gsd.cap_resolution(args.orthophoto_resolution, working_reconstruction,
                                            ignore_gsd=args.ignore_gsd,
                                            ignore_resolution=(not reconstruction.is_georeferenced()) and args.ignore_gsd,
                                            has_gcp=reconstruction.has_gcp())

            # odm_orthophoto definitions
            kwargs = {
                'odm_ortho_bin': context.odm_orthophoto_path,
                'log': tree.odm_orthophoto_log,
                'ortho': tree.odm_orthophoto_render,
                'corners': tree.odm_orthophoto_corners,
                'res': 1.0 / (resolution/100.0),
                'bands': '',
                'depth_idx': '',
                'inpaint': '',
                'utm_offsets': '',
                'a_srs': '',
                'vars': '',
                'gdal_configs': '--config GDAL_CACHEMAX %s' % (get_max_memory_mb() * 1024 * 1024)
            }

            models = []

            if args.use_3dmesh:
                base_dir = tree.odm_texturing
            else:
                base_dir = tree.odm_25dtexturing
                
            model_file = tree.odm_textured_model_obj

            if reconstruction.multi_camera:
                for band in reconstruction.multi_camera:
                    primary = band['name'] == get_primary_band_name(reconstruction.multi_camera, args.primary_band)
                    subdir = ""
                    if not primary:
                        subdir = band['name'].lower()
                    models.append(os.path.join(base_dir, subdir, model_file))
                kwargs['bands'] = '-bands %s' % (','.join([double_quote(b['name']) for b in reconstruction.multi_camera]))

                # If a RGB band is present, 
                # use bit depth of the first non-RGB band
                depth_idx = None
                all_bands = [b['name'].lower() for b in reconstruction.multi_camera]
                for b in ['rgb', 'redgreenblue']:
                    if b in all_bands:
                        for idx in range(len(all_bands)):
                            if all_bands[idx] != b:
                                depth_idx = idx
                                break
                        break
                
                if depth_idx is not None:
                    kwargs['depth_idx'] = '-outputDepthIdx %s' % depth_idx
            else:
                models.append(os.path.join(base_dir, model_file))

                # Perform edge inpainting on georeferenced RGB datasets
                if reconstruction.is_georeferenced():
                    kwargs['inpaint'] = "-inpaintThreshold 1.0"

                # Thermal dataset with single band
                if reconstruction.photos[0].band_name.upper() == "LWIR":
                    kwargs['bands'] = '-bands lwir'

            direct_contract = None
            if reconstruction.is_georeferenced():
                contract_path = tree.path(
                    "odm_georeferencing", "coordinate_contract.json"
                )
                try:
                    direct_contract = load_coordinate_contract(contract_path)
                except Exception as error:
                    raise CoordinateContractError(
                        "direct orthophoto rendering requires a readable persisted "
                        "coordinate contract; rerun from reconstruction",
                        artifact=contract_path,
                        operation="orthophoto direct render",
                        cause=error,
                    ) from error
                stage_contract = outputs.get("coordinate_contract")
                if stage_contract is not None and stage_contract != direct_contract:
                    raise CoordinateContractError(
                        "persisted coordinate contract does not match the stage contract",
                        artifact=contract_path,
                        operation="orthophoto direct render",
                    )
                invalid_models = [
                    model for model in models if not _mesh_has_geometry(model)
                ]
                if invalid_models:
                    raise CoordinateContractError(
                        "direct orthophoto rendering requires a nonempty compatible "
                        "public georeferenced mesh: %s" % invalid_models[0],
                        artifact=invalid_models[0],
                        operation="orthophoto direct render",
                    )

            kwargs['models'] = ','.join(map(double_quote, models))

            if reconstruction.is_georeferenced():
                orthophoto_vars = orthophoto.get_orthophoto_vars(args)
                kwargs['utm_offsets'] = (
                    "-utm_north_offset %s -utm_east_offset %s"
                    % (
                        direct_contract.storage_offset[1],
                        direct_contract.storage_offset[0],
                    )
                )
                kwargs['a_srs'] = "-a_srs %s" % double_quote(
                    direct_contract.output_crs_wkt
                )
                kwargs['vars'] = ' '.join(['-co %s=%s' % (k, orthophoto_vars[k]) for k in orthophoto_vars])
                kwargs['ortho'] = tree.odm_orthophoto_tif # Render directly to final file

            # run odm_orthophoto
            log.INFO('Creating GeoTIFF')
            system.run('"{odm_ortho_bin}" -inputFiles {models} '
                       '-logFile "{log}" -outputFile "{ortho}" -resolution {res} -verbose '
                       '-outputCornerFile "{corners}" {bands} {depth_idx} {inpaint} '
                       '{utm_offsets} {a_srs} {vars} {gdal_configs} '.format(**kwargs), env_vars={'OMP_NUM_THREADS': args.max_concurrency})

            # Create georeferenced GeoTiff
            if reconstruction.is_georeferenced():
                bounds_file_path = os.path.join(tree.odm_georeferencing, 'odm_georeferenced_model.bounds.gpkg')
                    
                # Cutline computation, before cropping
                # We want to use the full orthophoto, not the cropped one.
                submodel_run = is_submodel(tree.opensfm)
                if args.orthophoto_cutline:
                    cutline_file = os.path.join(tree.odm_orthophoto, "cutline.gpkg")

                    compute_cutline(tree.odm_orthophoto_tif, 
                                    bounds_file_path,
                                    cutline_file,
                                    args.max_concurrency,
                                    scale=0.25)
                    
                    if submodel_run:
                        orthophoto.compute_mask_raster(tree.odm_orthophoto_tif, cutline_file, 
                                            os.path.join(tree.odm_orthophoto, "odm_orthophoto_cut.tif"),
                                            blend_distance=20, only_max_coords_feature=True)
                    else:
                        log.INFO("Not a submodel run, skipping mask raster generation")

                orthophoto.post_orthophoto_steps(args, bounds_file_path, tree.odm_orthophoto_tif, tree.orthophoto_tiles, resolution, 
                    reconstruction, tree, not outputs["large"])

                # Generate feathered orthophoto also
                if args.orthophoto_cutline and submodel_run:
                    orthophoto.feather_raster(tree.odm_orthophoto_tif, 
                            os.path.join(tree.odm_orthophoto, "odm_orthophoto_feathered.tif"),
                            blend_distance=20
                        )

            else:
                if io.file_exists(tree.odm_orthophoto_render):
                    pseudogeo.add_pseudo_georeferencing(tree.odm_orthophoto_render)
                    log.INFO("Renaming %s --> %s" % (tree.odm_orthophoto_render, tree.odm_orthophoto_tif))
                    os.replace(tree.odm_orthophoto_render, tree.odm_orthophoto_tif)
                else:
                    log.WARNING("Could not generate an orthophoto (it did not render)")
        else:
            log.WARNING('Found a valid orthophoto in: %s' % tree.odm_orthophoto_tif)

        if io.file_exists(tree.odm_orthophoto_render):
            os.remove(tree.odm_orthophoto_render)
