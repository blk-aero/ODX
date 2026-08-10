import os
import json

from opendm import io
from opendm import log
from opendm import types
from opendm.utils import copy_paths, get_processing_results_paths
from opendm.ogctiles import build_3dtiles
from opendm.gltf import obj2glb
from opendm.georeferencing import (
    SharedControlObservation,
    SharedOutputObservation,
    persist_split_merge_derivatives,
)
from opendm.osfm import is_submodel
from opendm.shots import get_origin

class ODMPostProcess(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        log.INFO("Post Processing")

        if args.gltf:
            textured_model = os.path.join(tree.odm_texturing, tree.odm_textured_model_obj)
            textured_model_25d = os.path.join(tree.odm_25dtexturing, tree.odm_textured_model_obj)
            
            if os.path.isfile(textured_model) and not args.skip_3dmodel:
                input_obj = textured_model
            elif os.path.isfile(textured_model_25d):
                input_obj = textured_model_25d
            else:
                input_obj = textured_model
            
            odm_textured_model_glb = os.path.join(os.path.dirname(input_obj), tree.odm_textured_model_glb)

            if not os.path.exists(odm_textured_model_glb) or self.rerun():
                log.INFO("Generating glTF Binary")
                try:
                    coordinate_contract = outputs.get("coordinate_contract")
                    rtc = (
                        coordinate_contract.storage_offset
                        if coordinate_contract is not None
                        else reconstruction.get_proj_offset()
                    )
                    obj2glb(input_obj, odm_textured_model_glb, rtc=rtc, _info=log.INFO)
                except Exception as e:
                    log.WARNING(str(e))

        if getattr(args, '3d_tiles'):
            build_3dtiles(args, tree, reconstruction, self.rerun())

        if reconstruction.is_georeferenced() and is_submodel(tree.opensfm):
            shared_control_observations = [
                SharedControlObservation(
                    photo.filename, photo.longitude, photo.latitude, photo.altitude
                )
                for photo in reconstruction.photos
                if None not in (
                    getattr(photo, "longitude", None),
                    getattr(photo, "latitude", None),
                    getattr(photo, "altitude", None),
                )
            ]
            gcp_geojson = tree.path(
                "odm_georeferencing", "ground_control_points.geojson"
            )
            if os.path.isfile(gcp_geojson):
                with open(gcp_geojson) as gcp_file:
                    for feature in json.load(gcp_file).get("features", []):
                        coordinates = feature.get("geometry", {}).get("coordinates", ())
                        identifier = feature.get("properties", {}).get("id")
                        if identifier is not None and len(coordinates) >= 3:
                            shared_control_observations.append(
                                SharedControlObservation(
                                    "gcp:" + str(identifier), *coordinates[:3]
                                )
                            )
            shared_control_observations.sort(key=lambda item: item.identifier)
            topocentric_positions = {}
            with open(tree.opensfm_topocentric_reconstruction) as reconstruction_file:
                for partial in json.load(reconstruction_file):
                    for filename, shot in partial.get("shots", {}).items():
                        topocentric_positions[filename] = tuple(get_origin(shot))
            shared_output_observations = []
            shots_geojson = tree.path("odm_report", "shots.geojson")
            if os.path.isfile(shots_geojson):
                with open(shots_geojson) as shots_file:
                    for feature in json.load(shots_file).get("features", []):
                        properties = feature.get("properties", {})
                        identifier = properties.get("filename")
                        output_position = properties.get("translation")
                        if (
                            identifier in topocentric_positions
                            and output_position is not None
                            and len(output_position) == 3
                        ):
                            shared_output_observations.append(
                                SharedOutputObservation(
                                    identifier,
                                    topocentric_positions[identifier],
                                    tuple(output_position),
                                )
                            )
            shared_output_observations.sort(key=lambda item: item.identifier)
            persist_split_merge_derivatives(
                tree.root_path, outputs["coordinate_contract"],
                shared_control_observations=shared_control_observations,
                shared_output_observations=shared_output_observations,
            )

        if args.copy_to:
            try:
                copy_paths([os.path.join(args.project_path, p) for p in get_processing_results_paths()], args.copy_to, self.rerun())
            except Exception as e:
                log.WARNING("Cannot copy to %s: %s" % (args.copy_to, str(e)))
