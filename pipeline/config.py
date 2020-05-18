from dataclasses import dataclass
from typing import Tuple, Any, List
import ruamel.yaml
from ruamel.yaml.constructor import SafeConstructor
import dacite


@dataclass
class CornerSelectionConfig:
    max_corners: int
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
    frames_to_skip: int
    reset_period: int

    corner_selection: CornerSelectionConfig
    optical_flow: OpticalFlowConfig


@dataclass
class FivePointAlgorithm:
    min_number_of_points: int
    threshold: float
    probability: float


@dataclass
class RecoverPoseConfig:
    distance_threshold: float


@dataclass
class SolvePnPConfig:
    min_number_of_points: int


@dataclass
class RollingWindowConfig:
    period: int
    length: int


@dataclass
class BundleAdjustmentConfig:
    use_with_rolling_window: bool
    use_at_end: bool
    rolling_window: RollingWindowConfig
    verbosity: int


@dataclass
class VideoPipelineConfig:
    camera_matrix: Any

    use_five_pt_algorithm: bool
    use_solve_pnp: bool
    use_reconstruct_tracks: bool

    min_number_of_points_in_cloud: int

    klt: KLTConfig

    five_point_algorithm: FivePointAlgorithm
    recover_pose_algorithm: RecoverPoseConfig
    solvepnp: SolvePnPConfig

    bundle_adjustment: BundleAdjustmentConfig

    def __post_init__(self):
        assert (
            self.use_five_pt_algorithm or self.use_solve_pnp
        ), "At least one algorithm between fiv-pt and solvepnp must be used"


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
