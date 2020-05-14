import cv2
import argparse
import numpy as np

from pipeline import utils
from pipeline.base_pipeline import BasePipeline


class VideoPipelineMK1(BasePipeline):
    TYPE_CALIBRATION_MATRIX = 0
    TYPE_CAMERA = 1
    TYPE_POINT = 2

    def __init__(self, dir, save_debug_visualization=False):
        self.dir = dir
        self.save_debug_visualization = save_debug_visualization

        # Number of frames to skip between used frames (eg. 2 means using frames 1, 4, 7 and dropping 2, 3, 5, 6)
        self.frames_to_skip = 1

        # Number of frames for initial estimation
        self.num_frames_initial_estimation = 10

        # Number of frames between baseline reset. Hack to avoid further frames having no or few features
        self.feature_reset_rate = 10

        # params for ShiTomasi corner detection
        self.feature_params = {
            "maxCorners": 200,
            "qualityLevel": 0.5,
            "minDistance": 10,
            "blockSize": 10,
        }

        # Parameters for lucas kanade optical flow
        self.lk_params = {
            "winSize": (15, 15),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        }

        self.image_size = None

        self.camera_matrix = np.array(
            [
                [765.16859169, 0.0, 379.11876567],
                [0.0, 762.38664643, 497.22086655],
                [0.0, 0.0, 1.0],
            ]
        )

        self.k_ = None

        self.recover_pose_reconstruction_distance_threshold = 50
        self.find_essential_mat_threshold = 3
        self.find_essential_mat_prob = 0.98

        self.debug_colors = np.random.randint(
            0, 255, (self.feature_params["maxCorners"], 3)
        )

    def run(self):
        # Start by finding the images
        file, filename = self._get_video(self.dir)

        prev_frame_id = None
        prev_feature_pack_id = None
        prev_track_slice = None

        Rs = [np.eye(3)]
        Ts = [np.zeros((3, 1))]
        stored_points = {}

        comp_R = Rs[0]
        comp_T = Ts[0]

        feature_set_counter = 0

        # Loop through frames (using generators)
        for (
            next_track_slice,
            next_track_slice_index_mask,
            is_new_feature_set,
        ) in self._process_next_frame(file):
            if is_new_feature_set:
                feature_set_counter += 1
                prev_track_slice = next_track_slice
                continue

            tracks = np.array([prev_track_slice, next_track_slice])

            # calculate camera 2 position relative to camera 1
            (rel_R, rel_T, rel_points_3d, point_mask,) = self._get_pose_from_two_tracks(
                tracks
            )
            if rel_R is None:
                continue

            # translate 3D point position back from camera 1 frame to reference frame
            abs_points_3d = utils.translate_points_to_base_frame(
                comp_R, comp_T, rel_points_3d
            )
            # abs_points_3d = (comp_T + np.matmul(comp_R, rel_points_3d.transpose())).transpose()

            # translate camera position back to reference frame
            comp_R, comp_T = utils.compose_RTs(rel_R, rel_T, comp_R, comp_T)

            # store everything for later use
            stored_points = self._store_new_points(
                stored_points, abs_points_3d, point_mask, feature_set_counter
            )
            Rs += [comp_R]
            Ts += [comp_T]

            # prepare everything for next round
            prev_track_slice = next_track_slice

        points = np.array(
            [point_data["avg_point"] for _, point_data in stored_points.items()]
        )

        utils.write_to_viz_file(self.camera_matrix, Rs, Ts, points)
        utils.call_viz()

    # =================== INTERNAL FUNCTIONS ===========================================================================

    def _store_new_points(
        self, stored_points, points_3d, point_mask, feature_set_counter
    ):
        point_indexes = np.arange(len(points_3d))[point_mask]
        for point_index, point in zip(point_indexes, points_3d):
            point_index += feature_set_counter * self.feature_params["maxCorners"]

            if point_index not in stored_points:
                stored_points[point_index] = {"accum": point, "count": 1}
            else:
                stored_points[point_index]["count"] += 1
                stored_points[point_index]["accum"] += point

            stored_points[point_index]["avg_point"] = (
                stored_points[point_index]["accum"]
                / stored_points[point_index]["count"]
            )

        return stored_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with image files for reconstructions")
    parser.add_argument(
        "-sd",
        "--save_debug_visualization",
        help="Save debug visualizations to files?",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    sfm = VideoPipelineMK1(**vars(args))
    sfm.run()
