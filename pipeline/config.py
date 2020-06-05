from dataclasses import dataclass
from typing import Tuple, Any, List, Union
import ruamel.yaml
from ruamel.yaml.constructor import SafeConstructor
import dacite
import numpy as np


@dataclass
class CornerSelectionConfig:
    quality_level: float
    min_distance: int
    block_size: int


@dataclass
class OpticalFlowWindow:
    width: int
    height: int


@dataclass
class OpticalFlowStopCriteria:
    criteria_sum: int
    max_iter: int
    eps: float


@dataclass
class OpticalFlowConfig:
    window_size: OpticalFlowWindow
    max_level: int
    criteria: OpticalFlowStopCriteria


@dataclass
class KLTConfig:
    calculate_every_frame: bool
    display_klt_debug_frames: bool
    klt_debug_frames_delay: int
    frames_to_skip: int
    reset_period: int
    closeness_threshold: int
    max_features: int

    corner_selection: CornerSelectionConfig
    optical_flow: OpticalFlowConfig


@dataclass
class RollingWindowConfig:
    method: str
    period: int
    length: int
    step: int


@dataclass
class BundleAdjustmentConfig:
    tol: float
    method: str
    verbose: int
    camera_matrix: Any

    use_with_rolling_window: bool
    use_at_end: bool
    rolling_window: RollingWindowConfig

    def __post_init__(self):
        self.camera_matrix = np.array(self.camera_matrix)


@dataclass
class FivePointAlgorithmConfig:
    min_number_of_points: int
    essential_mat_threshold: float
    ransac_probability: float
    refine_matches_repetitions: int
    save_optimized_projections: bool
    camera_matrix: Any
    distance_threshold: float

    def __post_init__(self):
        self.camera_matrix = np.array(self.camera_matrix)


@dataclass
class SolvePnPConfig:
    min_number_of_points: int
    camera_matrix: Any
    use_epnp: bool
    use_iterative_pnp: bool

    def __post_init__(self):
        self.camera_matrix = np.array(self.camera_matrix)


# @dataclass
# class FivePointInitConfig:
#     five_pt_algorithm: FivePointAlgorithmConfig
#     use_bundle_adjustment: bool
#     bundle_adjustment: Union[BundleAdjustmentConfig, None]
#
#     def __post_init__(self):
#         assert not (
#             self.use_bundle_adjustment
#             and not (self.bundle_adjustment is not None)
#         )
#
#
# @dataclass
# class ThreeFrameInitConfig:
#     five_pt_algorithm: FivePointAlgorithmConfig
#     solve_pnp: SolvePnPConfig
#     use_bundle_adjustment: bool
#     bundle_adjustment: Union[BundleAdjustmentConfig, None]
#
#     def __post_init__(self):
#         assert not (
#             self.use_bundle_adjustment
#             and not (self.bundle_adjustment is not None)
#         )


@dataclass
class InitConfig:
    error_threshold: float
    num_reconstruction_frames: int
    num_error_calculation_frames: int


@dataclass
class VideoPipelineConfig:
    camera_matrix: Any

    use_five_pt_algorithm: bool
    use_solve_pnp: bool
    use_reconstruct_tracks: bool

    klt: KLTConfig

    init: InitConfig

    five_pt_algorithm: FivePointAlgorithmConfig
    solve_pnp: SolvePnPConfig

    bundle_adjustment: BundleAdjustmentConfig

    def __post_init__(self):
        assert (
            self.use_five_pt_algorithm or self.use_solve_pnp
        ), "At least one algorithm between fiv-pt and solvepnp must be used"

        self.camera_matrix = np.array(self.camera_matrix)


@dataclass
class SyntheticPipelineConfig(VideoPipelineConfig):
    synthetic_case: int


def construct_tuple(self, node):
    seq = self.construct_sequence(node)
    if seq:
        return tuple(seq)


SafeConstructor.add_constructor(u"!tuple", construct_tuple)


def load(file):

    config_raw = ruamel.yaml.safe_load(file)

    config = dacite.from_dict(data=config_raw, data_class=VideoPipelineConfig)
    return config
