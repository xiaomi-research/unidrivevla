from lead.common.constants import TargetDataset
from lead.training.config_training import TrainingConfig


class PlantConfig(TrainingConfig):
    """Training config for PlanT. Sets PlanT defaults and overrides."""

    model_type = "plant"

    # --- Backbone ---
    # HuggingFace pretrained BERT model ID. Determines hidden_size (384 for bert-medium).
    # The BERT weights are used as the architecture template only -- word_embeddings are
    # discarded, replaced by object token embeddings.
    plant_hf_checkpoint = "prajjwal1/bert-medium"
    # Whether to apply dropout after embedding concatenation, before feeding into BERT.
    plant_use_dropout = True
    # Dropout probability for the embedding dropout layer.
    plant_embd_pdrop = 0.1

    # --- Input tokenization ---
    # Number of route waypoints fed to the backbone as a single token.
    # The dataset pads / truncates ``route_original`` to this length.
    plant_num_route_points = 20
    # Detection range in meters. Objects beyond this radius are discarded.
    plant_range = 50
    # Multiplier for the front detection range. With range=50 and factor=2,
    # front range is 100m while side/rear range is 50m.
    plant_range_factor_front = 2
    # Whether to encode the HD map as an additional input token via ResNet18.
    # This is the raw road-layout map from the simulator, NOT the BEV semantic
    # segmentation used by TFv6 (which overlays occupancy on top of the HD map).
    plant_input_hdmap = False
    # Whether to include parked/static cars as object tokens (class "static_car").
    # When False, static cars are filtered out during tokenization.
    plant_input_static_cars = True
    # Whether to encode ego vehicle speed as an additional input token.
    # Paper found this can cause causal confusion -- disabled by default.
    plant_input_ego_speed = False

    # PlanT always needs the planning decoder (waypoints, route, target speed).
    use_planning_decoder = True

    # PlanT does not use any perception auxiliary tasks.
    use_semantic = False
    use_bev_semantic = False
    detect_boxes = False
    use_depth = False
    radar_detection = False
    use_radar_detection = False

    # PlanT consumes only object tokens and meta data -- skip loading all heavy sensors.
    use_rgb = False
    use_lidar = False
    use_radars = False

    # No sensor data to cache -- avoid the lzma + BeeGFS round-trip on every sample.
    use_persistent_cache = False

    # Datset
    carla_root = "data/carla_leaderboard2_360"

    @property
    def target_dataset(self):
        return TargetDataset.CARLA_LEADERBOARD2_3CAMERAS
