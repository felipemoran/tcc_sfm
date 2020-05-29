import cv2
import itertools
import numpy as np

from math import pi
from pipeline import utils
from pipeline.video_pipeline import VideoPipeline
from pipeline.config import VideoPipelineConfig
from ruamel.yaml import YAML
import dacite

np.set_printoptions(3, suppress=True)

TYPE_CALIBRATION_MATRIX = 0
TYPE_CAMERA = 1
TYPE_POINT = 2


class SyntheticPipeline(VideoPipeline):
    def __init__(self, config):
        super().__init__(config)

        self.synthetic_case = 2

    def _get_video(self, file_path):
        return None, None

    def _file_to_tracks(self, file):
        """Replaces the original function to generate synthetic data while maintaining everything else"""

        num_cameras = 2

        points_3d = self._get_synthetic_points()

        Rs = self._get_synthetic_camera_rotations()[:num_cameras]
        Ts = self._get_synthetic_camera_translations()[:num_cameras]

        for index, (R, T) in enumerate(zip(Rs, Ts)):
            # convert to the camera base, important!
            R_cam, T_cam = utils.invert_reference_frame(R, T)
            R_cam_vec = cv2.Rodrigues(R_cam)[0]

            track_slice = cv2.projectPoints(
                points_3d, R_cam_vec, T_cam, self.config.camera_matrix, None
            )[0].squeeze()

            slice_mask = (track_slice > 0).all(axis=1)

            drop = np.arange(1, 4) * 4 + index
            drop_bool = np.full(slice_mask.shape, True)
            drop_bool[drop] = False
            slice_mask = slice_mask & drop_bool

            index_mask = np.arange(len(points_3d))[slice_mask]

            track_slice = track_slice[slice_mask]

            yield track_slice, index_mask

    def _get_synthetic_points(self):
        # points_3d = np.array(
        #     list(itertools.product([9, 10, 11], [4, 5, 6], [-1, 0, 1])), dtype=np.float_
        # )
        # points_3d = np.array(
        #     list(itertools.product([9, 11], [4, 6], [-1, 1])), dtype=np.float_
        # )
        points_3d = np.array(
            list(itertools.product([8, 9, 10, 11, 12], [4, 5, 6], [0]))
            + list(itertools.product([8, 9, 10], [4, 5], [1]))
            + list(itertools.product([8], [4], [2])),
            dtype=np.float_,
        )
        return points_3d

    def _get_synthetic_camera_rotations(self):
        # matrizes de rotação para o posicionamento das cameras
        r1 = cv2.Rodrigues(np.array([-pi / 2, 0.0, 0.0]))[0]
        r2 = cv2.Rodrigues(np.array([0, 0, -pi / 4]))[0]
        r3 = cv2.Rodrigues(np.array([0, -pi / 2, 0]))[0]

        # vetores de rotação das cameras na base global
        Rs = np.array(
            [
                r1,
                np.matmul(r1, r3),
                np.matmul(r1, np.matmul(r3, np.matmul(r3, r2))),
                np.matmul(r1, np.matmul(r3, np.matmul(r3, r3))),
                np.matmul(r1, r1),
            ]
        )
        return Rs

    def _get_synthetic_camera_translations(self):
        # vetores de translação das câmeras na base global
        if self.synthetic_case == 1:
            Ts = np.array(
                [[10, 0, 0], [15, 5, 0], [10, 10, 0], [5, 5, 0], [10, 5, 5],],
                dtype=np.float_,
            )
        elif self.synthetic_case == 2:
            Ts = np.array(
                [[10, 0, 0], [20, 5, 0], [10, 12.5, 0], [5, 5, 0], [10, 5, 5],],
                dtype=np.float_,
            )
        else:
            raise ValueError
        return Ts


if __name__ == "__main__":
    yaml = YAML()
    with open("config.yaml", "r") as f:
        config_raw = yaml.load(f)
    config = dacite.from_dict(data=config_raw, data_class=VideoPipelineConfig)

    sp = SyntheticPipeline(config)
    sp.run(None)
