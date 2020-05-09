
import numpy as np
from pipeline.synthetic_pipeline_mk2 import SyntheticPipelineMK2
from pipeline.video_pipeline_mk3 import VideoPipelineMK3

np.set_printoptions(3, suppress=True)

TYPE_CALIBRATION_MATRIX = 0
TYPE_CAMERA = 1
TYPE_POINT = 2


class SyntheticPipelineMK3(SyntheticPipelineMK2, VideoPipelineMK3):
    def __init__(self):
        VideoPipelineMK3.__init__(self, None)
        SyntheticPipelineMK2.__init__(self)
        # camera arbitraria
        self.camera_matrix = np.array([
            [500, 0.0, 500],
            [0.0, 500, 500],
            [0.0, 0.0, 1.0],
        ], dtype=np.float_)

    def run(self):
        return VideoPipelineMK3.run(self)


if __name__ == '__main__':
    sp = SyntheticPipelineMK3()
    sp.run()
