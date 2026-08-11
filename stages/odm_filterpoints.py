import os
import numpy as np

from opendm import log
from opendm import io
from opendm import system
from opendm import context
from opendm import point_cloud
from opendm import types
from opendm import gsd
from opendm.boundary import boundary_offset, compute_boundary_from_shots
from opendm.georeferencing import transform_output_boundary
from stages.odm_georeferencing import resolve_project_coordinate_contract


def _finest_boundary_resolution(args):
    resolutions = []
    orthophoto_resolution = getattr(args, "orthophoto_resolution", 0.0)
    if orthophoto_resolution > 0.0:
        resolutions.append(orthophoto_resolution / 100.0)
    if getattr(args, "dsm", False) or getattr(args, "dtm", False):
        dem_resolution = getattr(args, "dem_resolution", 0.0)
        if dem_resolution > 0.0:
            resolutions.append(dem_resolution / 100.0)
    return min(resolutions) if resolutions else 0.01


class ODMFilterPoints(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        if not os.path.exists(tree.odm_filterpoints): system.mkdir_p(tree.odm_filterpoints)

        boundary_contract = None
        if reconstruction.is_georeferenced() and (
            "boundary" in outputs or getattr(args, "auto_boundary", False)
        ):
            boundary_contract = resolve_project_coordinate_contract(
                tree, reconstruction, args
            )
            outputs["coordinate_contract"] = boundary_contract

        if "boundary" in outputs and boundary_contract is not None:
            outputs["boundary_finest_resolution"] = _finest_boundary_resolution(args)
            outputs["topocentric_boundary"] = transform_output_boundary(
                outputs["boundary"],
                boundary_contract,
                finest_resolution=outputs["boundary_finest_resolution"],
                serialization_bound=1e-9,
            )[:, :2].tolist()

        inputPointCloud = ""
        
        # check if reconstruction was done before
        if not io.file_exists(tree.filtered_point_cloud) or self.rerun():
            if args.fast_orthophoto:
                inputPointCloud = os.path.join(tree.opensfm, 'reconstruction.ply')
            else:
                inputPointCloud = tree.openmvs_model

            # Check if we need to compute boundary
            if args.auto_boundary:
                if reconstruction.is_georeferenced():
                    if not 'boundary' in outputs:
                        boundary_distance = None

                        if args.auto_boundary_distance > 0:
                            boundary_distance = args.auto_boundary_distance
                        else:
                            avg_gsd = gsd.opensfm_reconstruction_average_gsd(
                                tree.opensfm_topocentric_reconstruction
                            )
                            if avg_gsd is not None:
                                boundary_distance = avg_gsd * 100 # 100 is arbitrary
                            
                        if boundary_distance is not None:
                            topocentric_reconstruction = tree.opensfm_topocentric_reconstruction
                            outputs['topocentric_boundary'] = compute_boundary_from_shots(
                                topocentric_reconstruction, boundary_distance, (0, 0)
                            )
                            if outputs['topocentric_boundary'] is None:
                                log.WARNING("Cannot compute boundary from camera shots")
                            else:
                                topocentric = np.asarray(
                                    outputs['topocentric_boundary'], dtype=np.float64
                                )
                                topocentric_3d = np.column_stack((
                                    topocentric[:, :2], np.zeros(len(topocentric))
                                ))
                                outputs['boundary'] = boundary_contract.transform_points(
                                    topocentric_3d, apply_storage_offset=False
                                )[:, :2].tolist()
                        else:
                            log.WARNING("Cannot compute boundary (GSD cannot be estimated)")
                    else:
                        log.WARNING("--auto-boundary set but so is --boundary, will use --boundary")
                else:
                    log.WARNING("Not a georeferenced reconstruction, will ignore --auto-boundary")
                    
            point_cloud.filter(inputPointCloud, tree.filtered_point_cloud, tree.filtered_point_cloud_stats,
                                standard_deviation=args.pc_filter, 
                                sample_radius=args.pc_sample,
                                boundary=outputs.get(
                                    'topocentric_boundary',
                                    boundary_offset(
                                        outputs.get('boundary'), reconstruction.get_proj_offset()
                                    ),
                                ),
                                max_concurrency=args.max_concurrency)
            
            # Quick check
            info = point_cloud.ply_info(tree.filtered_point_cloud)
            if info["vertex_count"] == 0:
                extra_msg = ''
                if 'boundary' in outputs:
                    extra_msg = '. Also, since you used a boundary setting, make sure that the boundary polygon you specified covers the reconstruction area correctly.'
                raise system.ExitException("Uh oh! We ended up with an empty point cloud. This means that the reconstruction did not succeed. Have you followed best practices for data acquisition? See https://docs.webodm.org/flying-tips/%s" % extra_msg)
        else:
            log.WARNING('Found a valid point cloud file in: %s' %
                            tree.filtered_point_cloud)
        
        if args.optimize_disk_space and inputPointCloud:
            if os.path.isfile(inputPointCloud):
                os.remove(inputPointCloud)
