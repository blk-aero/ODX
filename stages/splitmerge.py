import os
import shutil
import json
import numpy as np
from opendm import log
from opendm.osfm import OSFMContext, get_submodel_argv, get_submodel_paths, get_all_submodel_paths, get_submodel_project_paths
from opendm import types
from opendm import io
from opendm import system
from opendm import orthophoto
from opendm.dem import utils
from opendm.dem.merge import euclidean_merge_dems
from opensfm.large import metadataset
from opendm.cropper import Cropper
from opendm.remote import LocalRemoteExecutor
from opendm.shots import merge_geojson_shots, merge_cameras
from opendm import point_cloud
from opendm.utils import double_quote, add_raster_meta_tags
from opendm.tiles.tiler import generate_dem_tiles
from opendm.cogeo import convert_to_cogeo
from opendm import multispectral
from opendm.georeferencing import (
    CoordinateContractError,
    coordinate_contract_metadata,
    load_coordinate_contract,
    load_output_selection,
    load_split_merge_derivatives,
    validate_split_merge,
)
from stages.odm_georeferencing import resolve_project_output_selection


SPLIT_OVERLAP_PRODUCT_TOLERANCE = 1.0


def _propagate_output_selection(selection_path, submodel_paths):
    for opensfm_path in submodel_paths:
        project_path = os.path.abspath(os.path.join(opensfm_path, ".."))
        destination = os.path.join(
            project_path, "odm_georeferencing", "output_selection.json"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(selection_path, destination)


def _validate_submodel_derivatives(tree):
    selection_path = tree.path("odm_georeferencing", "output_selection.json")
    selection = load_output_selection(selection_path)
    contracts = []
    project_paths = []
    for project_path in get_submodel_project_paths(tree.submodels_path):
        manifest = os.path.join(
            project_path, "odm_georeferencing", "coordinate_contract.json"
        )
        contracts.append(load_coordinate_contract(manifest))
        project_paths.append(project_path)

    validate_split_merge(selection, contracts)
    overlap_samples = []
    published_positions_by_id = {}
    covered_pairs = set()
    for contract_index, (project_path, contract) in enumerate(
        zip(project_paths, contracts)
    ):
        _, _, output_observations = load_split_merge_derivatives(
            project_path, contract
        )
        persisted_output_by_id = {
            observation.identifier: observation.output
            for observation in output_observations
        }
        shots_path = os.path.join(project_path, "odm_report", "shots.geojson")
        if not os.path.exists(shots_path):
            continue
        try:
            with open(shots_path) as shots_file:
                shots = json.load(shots_file)
            if shots.get("coordinate_contract") != coordinate_contract_metadata(contract):
                raise ValueError("report metadata differs from its coordinate contract")
            for feature in shots.get("features", ()):
                properties = feature.get("properties", {})
                identifier = properties.get("filename")
                output = properties.get("translation")
                if identifier is None or output is None:
                    continue
                if len(output) != 3:
                    raise ValueError("published camera position is not three-dimensional")
                if identifier not in persisted_output_by_id or not np.allclose(
                    output, persisted_output_by_id[identifier], atol=1e-9, rtol=0.0
                ):
                    raise ValueError(
                        "published camera position differs from derivative metadata"
                    )
                for first_index, first_output in published_positions_by_id.get(
                    identifier, ()
                ):
                    overlap_samples.append((
                        first_index, [first_output], contract_index, [output]
                    ))
                    covered_pairs.add((first_index, contract_index))
                published_positions_by_id.setdefault(identifier, []).append(
                    (contract_index, output)
                )
        except CoordinateContractError:
            raise
        except Exception as error:
            raise CoordinateContractError(
                "could not validate georeferenced overlap metadata",
                artifact=shots_path, operation="overlap validation", cause=error,
            ) from error
    missing_pairs = [
        (index, index + 1) for index in range(len(contracts) - 1)
        if (index, index + 1) not in covered_pairs
    ]
    if missing_pairs:
        raise CoordinateContractError(
            "adjacent submodels {} have no shared Earth-referenced control observation".format(
                missing_pairs
            ),
            artifact="split merge", operation="overlap validation",
        )
    validate_split_merge(
        selection, contracts, overlap_samples=overlap_samples,
        product_tolerance=SPLIT_OVERLAP_PRODUCT_TOLERANCE,
    )
    return selection, tuple(contracts)

class ODMSplitStage(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']
        photos = reconstruction.photos
        outputs['large'] = False

        image_groups_file = os.path.join(args.project_path, "image_groups.txt")
        if 'split_image_groups_is_set' in args:
            image_groups_file = os.path.abspath(args.split_image_groups)

        if io.file_exists(image_groups_file):
            outputs['large'] = True
        elif len(photos) > args.split:
            # check for availability of geotagged photos
            if reconstruction.has_geotagged_photos():
                outputs['large'] = True
            else:
                log.WARNING('Could not perform split-merge as GPS information in photos or image_groups.txt is missing.')

        if outputs['large']:
            # If we have a cluster address, we'll use a distributed workflow
            local_workflow = not bool(args.sm_cluster)

            octx = OSFMContext(tree.opensfm)
            split_done_file = octx.path("split_done.txt")

            if not io.file_exists(split_done_file) or self.rerun():
                orig_max_concurrency = args.max_concurrency
                if not local_workflow:
                    args.max_concurrency = max(1, args.max_concurrency - 1)
                    log.INFO("Setting max-concurrency to %s to better handle remote splits" % args.max_concurrency)

                log.INFO("Large dataset detected (%s photos) and split set at %s. Preparing split merge." % (len(photos), args.split))
                multiplier = (1.0 / len(reconstruction.multi_camera)) if reconstruction.multi_camera else 1.0

                config = [
                    "submodels_relpath: " + os.path.join("..", "submodels", "opensfm"),
                    "submodel_relpath_template: " + os.path.join("..", "submodels", "submodel_%04d", "opensfm"),
                    "submodel_images_relpath_template: " + os.path.join("..", "submodels", "submodel_%04d", "images"),
                    "submodel_size: %s" % max(2, int(float(args.split) * multiplier)),
                    "submodel_overlap: %s" % args.split_overlap,
                ]

                octx.setup(args, tree.dataset_raw, photos, reconstruction=reconstruction, append_config=config, rerun=self.rerun())
                octx.photos_to_metadata(photos, args.rolling_shutter, args.rolling_shutter_readout, args.gps_accuracy, self.rerun())
                resolve_project_output_selection(tree, reconstruction, args)

                self.update_progress(5)

                if local_workflow:
                    octx.feature_matching(self.rerun())

                self.update_progress(20)

                # Create submodels
                if not io.dir_exists(tree.submodels_path) or self.rerun():
                    if io.dir_exists(tree.submodels_path):
                        log.WARNING("Removing existing submodels directory: %s" % tree.submodels_path)
                        shutil.rmtree(tree.submodels_path)

                    octx.run("create_submodels")
                else:
                    log.WARNING("Submodels directory already exist at: %s" % tree.submodels_path)

                # Find paths of all submodels
                mds = metadataset.MetaDataSet(tree.opensfm)
                submodel_paths = [os.path.abspath(p) for p in mds.get_submodel_paths()]
                _propagate_output_selection(
                    tree.path("odm_georeferencing", "output_selection.json"),
                    submodel_paths,
                )

                for sp in submodel_paths:
                    sp_octx = OSFMContext(sp)
                    submodel_images_dir = os.path.abspath(sp_octx.path("..", "images"))

                    # Copy filtered GCP file if needed
                    # One in OpenSfM's directory, one in the submodel project directory
                    if reconstruction.gcp and reconstruction.gcp.exists():
                        submodel_gcp_file = os.path.abspath(sp_octx.path("..", "gcp_list.txt"))

                        if reconstruction.gcp.make_filtered_copy(submodel_gcp_file, submodel_images_dir):
                            log.INFO("Copied filtered GCP file to %s" % submodel_gcp_file)
                            io.copy(submodel_gcp_file, os.path.abspath(sp_octx.path("gcp_list.txt")))
                        else:
                            log.INFO("No GCP will be copied for %s, not enough images in the submodel are referenced by the GCP" % sp_octx.name())
                    
                    # Copy GEO file if needed (one for each submodel project directory)
                    if tree.odm_geo_file is not None and os.path.isfile(tree.odm_geo_file):
                        geo_dst_path = os.path.abspath(sp_octx.path("..", "geo.txt"))
                        io.copy(tree.odm_geo_file, geo_dst_path)
                        log.INFO("Copied GEO file to %s" % geo_dst_path)

                    # If this is a multispectral dataset,
                    # we need to link the multispectral images
                    if reconstruction.multi_camera:
                        submodel_images = os.listdir(submodel_images_dir)
                        
                        primary_band_name = multispectral.get_primary_band_name(reconstruction.multi_camera, args.primary_band)
                        _, p2s = multispectral.compute_band_maps(reconstruction.multi_camera, primary_band_name)
                        for filename in p2s:
                            if filename in submodel_images:
                                secondary_band_photos = p2s[filename]
                                for p in secondary_band_photos:
                                    system.link_file(os.path.join(tree.dataset_raw, p.filename), submodel_images_dir)

                # Reconstruct each submodel
                log.INFO("Dataset has been split into %s submodels. Reconstructing each submodel..." % len(submodel_paths))
                self.update_progress(25)

                if local_workflow:
                    for sp in submodel_paths:
                        log.INFO("Reconstructing %s" % sp)
                        local_sp_octx = OSFMContext(sp)
                        local_sp_octx.create_tracks(self.rerun())
                        local_sp_octx.reconstruct(args.sfm_algorithm, args.rolling_shutter, not args.sfm_no_partial, self.rerun())
                else:
                    lre = LocalRemoteExecutor(args.sm_cluster, args.sfm_algorithm, args.rolling_shutter, self.rerun())
                    lre.set_projects([os.path.abspath(os.path.join(p, "..")) for p in submodel_paths])
                    lre.run_reconstruction()

                self.update_progress(50)

                remove_paths = []

                # Align
                if not args.sm_no_align:
                    octx.align_reconstructions(self.rerun())

                    self.update_progress(55)

                    # Aligned reconstruction is in reconstruction.aligned.json
                    # We need to rename it to reconstruction.json
                    for sp in submodel_paths:
                        sp_octx = OSFMContext(sp)

                        aligned_recon = sp_octx.path('reconstruction.aligned.json')
                        unaligned_recon = sp_octx.path('reconstruction.unaligned.json')
                        main_recon = sp_octx.path('reconstruction.json')

                        if io.file_exists(main_recon) and io.file_exists(unaligned_recon) and not self.rerun():
                            log.INFO("Submodel %s has already been aligned." % sp_octx.name())
                            continue

                        if not io.file_exists(aligned_recon):
                            log.WARNING("Submodel %s does not have an aligned reconstruction (%s). "
                                            "This could mean that the submodel could not be reconstructed "
                                            " (are there enough features to reconstruct it?). Skipping." % (sp_octx.name(), aligned_recon))
                            remove_paths.append(sp)
                            continue

                        if io.file_exists(main_recon):
                            shutil.move(main_recon, unaligned_recon)

                        shutil.move(aligned_recon, main_recon)
                        log.INFO("%s is now %s" % (aligned_recon, main_recon))

                # Remove invalid submodels
                submodel_paths = [p for p in submodel_paths if not p in remove_paths]

                # Both local and remote split reconstruction paths bypass the
                # normal OpenSfM stage that marks a newly reconstructed model.
                # Authorize first-time contract creation only for submodels
                # that successfully reached this fresh reconstruction frontier.
                for sp in submodel_paths:
                    OSFMContext(sp).touch(
                        os.path.join(sp, "fresh_reconstruction.marker")
                    )

                # Run ODX toolchain for each submodel
                if local_workflow:
                    for sp in submodel_paths:
                        sp_octx = OSFMContext(sp)

                        log.INFO("========================")
                        log.INFO("Processing %s" % sp_octx.name()) 
                        log.INFO("========================")

                        argv = get_submodel_argv(args, tree.submodels_path, sp_octx.name())

                        # Re-run the ODX toolchain on the submodel
                        system.run(" ".join(map(double_quote, map(str, argv))), env_vars=os.environ.copy())
                else:
                    lre.set_projects([os.path.abspath(os.path.join(p, "..")) for p in submodel_paths])
                    lre.run_toolchain()

                # Restore max_concurrency value
                args.max_concurrency = orig_max_concurrency

                octx.touch(split_done_file)
            else:
                log.WARNING('Found a split done file in: %s' % split_done_file)
        else:
            log.INFO("Normal dataset, will process all at once.")
            self.progress = 0.0


class ODMMergeStage(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        if outputs['large']:
            if not os.path.exists(tree.submodels_path):
                raise system.ExitException("We reached the merge stage, but %s folder does not exist. Something must have gone wrong at an earlier stage. Check the log and fix possible problem before restarting?" % tree.submodels_path)

            _validate_submodel_derivatives(tree)
                

            # Merge point clouds
            if args.merge in ['all', 'pointcloud']:
                if not io.file_exists(tree.odm_georeferencing_model_laz) or self.rerun():
                    all_point_clouds = get_submodel_paths(tree.submodels_path, "odm_georeferencing", "odm_georeferenced_model.laz")
                    
                    try:
                        point_cloud.merge(all_point_clouds, tree.odm_georeferencing_model_laz, rerun=self.rerun())
                        point_cloud.post_point_cloud_steps(args, tree, self.rerun())
                    except Exception as e:
                        log.WARNING("Could not merge point cloud: %s (skipping)" % str(e))
                else:
                    log.WARNING("Found merged point cloud in %s" % tree.odm_georeferencing_model_laz)
                
            
            self.update_progress(25)

            # Merge crop bounds
            merged_bounds_file = os.path.join(tree.odm_georeferencing, 'odm_georeferenced_model.bounds.gpkg')
            if not io.file_exists(merged_bounds_file) or self.rerun():
                all_bounds = get_submodel_paths(tree.submodels_path, 'odm_georeferencing', 'odm_georeferenced_model.bounds.gpkg')
                log.INFO("Merging all crop bounds: %s" % all_bounds)
                if len(all_bounds) > 0:
                    # Calculate a new crop area
                    # based on the convex hull of all crop areas of all submodels
                    # (without a buffer, otherwise we are double-cropping)
                    Cropper.merge_bounds(all_bounds, merged_bounds_file, 0)
                else:
                    log.WARNING("No bounds found for any submodel.")

            # Merge orthophotos
            if args.merge in ['all', 'orthophoto']:
                if not io.dir_exists(tree.odm_orthophoto):
                    system.mkdir_p(tree.odm_orthophoto)

                if not io.file_exists(tree.odm_orthophoto_tif) or self.rerun():
                    all_orthos_and_ortho_cuts = get_all_submodel_paths(tree.submodels_path,
                        os.path.join("odm_orthophoto", "odm_orthophoto_feathered.tif"),
                        os.path.join("odm_orthophoto", "odm_orthophoto_cut.tif"),
                    )

                    if len(all_orthos_and_ortho_cuts) > 1:
                        log.INFO("Found %s submodels with valid orthophotos and cutlines" % len(all_orthos_and_ortho_cuts))
                        
                        # TODO: histogram matching via rasterio
                        # currently parts have different color tones

                        if io.file_exists(tree.odm_orthophoto_tif):
                            os.remove(tree.odm_orthophoto_tif)

                        orthophoto_vars = orthophoto.get_orthophoto_vars(args)
                        orthophoto.merge(all_orthos_and_ortho_cuts, tree.odm_orthophoto_tif, orthophoto_vars, max_workers=args.max_concurrency)
                        orthophoto.post_orthophoto_steps(args, merged_bounds_file, tree.odm_orthophoto_tif, tree.orthophoto_tiles, args.orthophoto_resolution,
                            reconstruction, tree, False)
                    elif len(all_orthos_and_ortho_cuts) == 1:
                        # Simply copy
                        log.WARNING("A single orthophoto/cutline pair was found between all submodels.")
                        shutil.copyfile(all_orthos_and_ortho_cuts[0][0], tree.odm_orthophoto_tif)
                    else:
                        log.WARNING("No orthophoto/cutline pairs were found in any of the submodels. No orthophoto will be generated.")
                else:
                    log.WARNING("Found merged orthophoto in %s" % tree.odm_orthophoto_tif)

            self.update_progress(75)

            # Merge DEMs
            def merge_dems(dem_filename, human_name):
                if not io.dir_exists(tree.path('odm_dem')):
                    system.mkdir_p(tree.path('odm_dem'))

                dem_file = tree.path("odm_dem", dem_filename)
                if not io.file_exists(dem_file) or self.rerun():
                    all_dems = get_submodel_paths(tree.submodels_path, "odm_dem", dem_filename)
                    log.INFO("Merging %ss" % human_name)
                    
                    # Merge
                    dem_vars = utils.get_dem_vars(args)
                    eu_map_source = None # Default

                    # Use DSM's euclidean map for DTMs
                    # (requires the DSM to be computed)
                    if human_name == "DTM":
                        eu_map_source = "dsm"

                    euclidean_merge_dems(all_dems, dem_file, dem_vars, euclidean_map_source=eu_map_source)

                    if io.file_exists(dem_file):
                        # Crop
                        if args.crop > 0 or args.boundary:
                            Cropper.crop(merged_bounds_file, dem_file, dem_vars, keep_original=not args.optimize_disk_space)
                        log.INFO("Created %s" % dem_file)
                        
                        if args.tiles:
                            generate_dem_tiles(dem_file, tree.path("%s_tiles" % human_name.lower()), args.max_concurrency, args.dem_resolution)
                        
                        add_raster_meta_tags(dem_file, reconstruction, tree, embed_gcp_meta=False)

                        if args.cog:
                            convert_to_cogeo(dem_file, max_workers=args.max_concurrency, deflate_predictor=3)
                    else:
                        log.WARNING("Cannot merge %s, %s was not created" % (human_name, dem_file))
                
                else:
                    log.WARNING("Found merged %s in %s" % (human_name, dem_filename))

            if args.merge in ['all', 'dem'] and args.dsm:
                merge_dems("dsm.tif", "DSM")

            if args.merge in ['all', 'dem'] and args.dtm:
                merge_dems("dtm.tif", "DTM")

            self.update_progress(95)

            # Merge reports
            if not io.dir_exists(tree.odm_report):
                system.mkdir_p(tree.odm_report)

            geojson_shots = tree.path(tree.odm_report, "shots.geojson")
            if not io.file_exists(geojson_shots) or self.rerun():
                geojson_shots_files = get_submodel_paths(tree.submodels_path, "odm_report", "shots.geojson")
                log.INFO("Merging %s shots.geojson files" % len(geojson_shots_files))
                merge_geojson_shots(geojson_shots_files, geojson_shots)
            else:
                log.WARNING("Found merged shots.geojson in %s" % tree.odm_report)

            # Merge cameras
            cameras_json = tree.path("cameras.json")
            if not io.file_exists(cameras_json) or self.rerun():
                cameras_json_files = get_submodel_paths(tree.submodels_path, "cameras.json")
                log.INFO("Merging %s cameras.json files" % len(cameras_json_files))
                merge_cameras(cameras_json_files, cameras_json)
            else:
                log.WARNING("Found merged cameras.json in %s" % tree.root_path)

            # Stop the pipeline short by skipping to the postprocess stage.
            # Afterwards, we're done.
            self.next_stage = self.last_stage()
        else:
            log.INFO("Normal dataset, nothing to merge.")
            self.progress = 0.0

        
