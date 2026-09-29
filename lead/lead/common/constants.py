from enum import IntEnum, auto


class CameraPointCloudIndex(IntEnum):
    """Index to access point cloud array of camera."""

    X = 0
    Y = 1
    Z = 2
    UNREAL_SEMANTICS_ID = 3
    UNREAL_INSTANCE_ID = 4


class TargetDataset(IntEnum):
    """Dataset we target to collect data/train/evaluate on."""

    UNKNOWN = 0
    CARLA_LEADERBOARD2_3CAMERAS = auto()
    CARLA_LEADERBOARD2_6CAMERAS = auto()
    CARLA_LEADERBOARD2_1CAMERA = auto()
    NAVSIM_4CAMERAS = auto()
    WAYMO_E2E_2025_3CAMERAS = auto()
    CARLA_PY123D_6CAMERAS = auto()


class SourceDataset(IntEnum):
    """Dataset intenum a sample originates from."""

    UNKNOWN = 0
    CARLA = auto()
    NAVSIM = auto()
    WAYMO_E2E_2025 = auto()


SOURCE_DATASET_NAME_MAP = {
    SourceDataset.CARLA: "carla",
    SourceDataset.NAVSIM: "navsim",
    SourceDataset.WAYMO_E2E_2025: "waymo_e2e_2025",
}


class RadarLabels(IntEnum):
    """Index to access radar label array."""

    X = 0
    Y = 1
    V = 2
    VALID = 3


class RadarDataIndex(IntEnum):
    """Index to access radar data array."""

    X = 0
    Y = 1
    Z = 2
    V = 3
    SENSOR_ID = 4


class TransfuserBoundingBoxIndex(IntEnum):
    """Index to access bounding array of TransFuser."""

    X = 0
    Y = 1
    W = 2
    H = 3
    YAW = 4
    VELOCITY = 5
    BRAKE = 6
    CLASS = 7
    SCORE = 8  # Only available for prediction
    NUM_RADAR_POINTS = 8  # Only available for ground truth


class TransfuserBoundingBoxClass(IntEnum):
    """Bounding box classes used in TransFuser."""

    VEHICLE = 0
    WALKER = 1
    TRAFFIC_LIGHT = 2
    STOP_SIGN = 3
    SPECIAL = 4
    OBSTACLE = 5
    PARKING = 6
    BIKER = 7


class CarlaNavigationCommand(IntEnum):
    """Source: https://carla.org/Doxygen/html/d0/db7/namespacecarla_1_1traffic__manager.html#a5734807dba08623eeca046a963ade360"""

    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


# Mapping from CARLA navigation command to natural language string
CARLA_NAVIGATION_COMMAND_STR_MAP = {
    CarlaNavigationCommand.UNKNOWN: "Unknown",
    CarlaNavigationCommand.LEFT: "Turn Left",
    CarlaNavigationCommand.RIGHT: "Turn Right",
    CarlaNavigationCommand.STRAIGHT: "Straight",
    CarlaNavigationCommand.LANEFOLLOW: "Follow Lane",
    CarlaNavigationCommand.CHANGELANELEFT: "Change Lane Left",
    CarlaNavigationCommand.CHANGELANERIGHT: "Change Lane Right",
}


class ChaffeurNetBEVSemanticClass(IntEnum):
    """Indices to access BEV semantic map produced by ChaffeurNet."""

    UNLABELED = 0
    ROAD = 1
    SIDEWALK = 2
    LANE_MARKERS = 3
    LANE_MARKERS_BROKEN = 4
    STOP_SIGNS = 5
    TRAFFIC_GREEN = 6
    TRAFFIC_YELLOW = 7
    TRAFFIC_RED = 8


class TransfuserBEVSemanticClass(IntEnum):
    """Indices to access BEV semantic map produced by TransFuser."""

    UNLABELED = 0
    ROAD = 1
    LANE_MARKERS = 2
    STOP_SIGNS = 3
    VEHICLE = 4
    WALKER = 5
    OBSTACLE = 6
    PARKING_VEHICLE = 7
    SPECIAL_VEHICLE = 8
    BIKER = 9
    TRAFFIC_GREEN = 10
    TRAFFIC_RED_NORMAL = 11
    TRAFFIC_RED_NOT_NORMAL = 12


class TransfuserBEVOccupancyClass(IntEnum):
    """Indices to access BEV occupancy map produced by TransFuser."""

    UNLABELED = 0
    VEHICLE = 1
    WALKER = 2
    OBSTACLE = 3
    PARKING_VEHICLE = 4
    SPECIAL_VEHICLE = 5
    BIKER = 6
    TRAFFIC_GREEN = 7
    TRAFFIC_RED_NORMAL = 8
    TRAFFIC_RED_NOT_NORMAL = 9


class WeatherVisibility(IntEnum):
    """Visibility conditions in CARLA simulator, classified by TransFuser."""

    CLEAR = 0
    OK = 1
    LIMITED = 2
    VERY_LIMITED = 3


class CarlaSemanticSegmentationClass(IntEnum):
    """https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera"""

    Unlabeled = 0
    Roads = 1
    SideWalks = 2
    Building = 3
    Wall = 4
    Fence = 5
    Pole = 6
    TrafficLight = 7
    TrafficSign = 8
    Vegetation = 9
    Terrain = 10
    Sky = 11
    Pedestrian = 12
    Rider = 13
    Car = 14
    Truck = 15
    Bus = 16
    Train = 17
    Motorcycle = 18
    Bicycle = 19
    Static = 20
    Dynamic = 21
    Other = 22
    Water = 23
    RoadLine = 24
    Ground = 25
    Bridge = 26
    RailTrack = 27
    GuardRail = 28
    # --- Own classes ---
    ConeAndTrafficWarning = 29
    SpecialVehicles = 30
    StopSign = 31


class TransfuserSemanticSegmentationClass(IntEnum):
    """Semantic segmentation classes used in TransFuser."""

    UNLABELED = 0
    VEHICLE = 1
    ROAD = 2
    TRAFFIC_LIGHT = 3
    PEDESTRIAN = 4
    ROAD_LINE = 5
    OBSTACLE = 6
    SPECIAL_VEHICLE = 7
    STOP_SIGN = 8
    BIKER = 9


def rgb(r, g, b):
    """Dummy function to help with visualizing RGB colors by using a VSCode extension."""
    return (r, g, b)


# Other visualization
LIDAR_COLOR = rgb(0, 0, 0)
EGO_BB_COLOR = rgb(151, 15, 48)
TP_DEFAULT_COLOR = rgb(255, 10, 10)
RADAR_COLOR = rgb(24, 237, 3)
RADAR_DETECTION_COLOR = rgb(255, 24, 0)

# Planning visualization
GROUNDTRUTH_BB_WP_COLOR = rgb(15, 60, 255)
GROUNDTRUTH_FUTURE_WAYPOINT_COLOR = rgb(0, 0, 0)
GROUND_TRUTH_PAST_WAYPOINT_COLOR = rgb(0, 0, 0)
PREDICTION_WAYPOINT_COLOR = rgb(255, 0, 0)
PREDICTION_ROUTE_COLOR = rgb(0, 0, 255)
PREDICTION_WAYPOINT_RADIUS = 10
PREDICTION_ROUTE_RADIUS = 6

# TransFuser BEV Semantic class to color
CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER = {
    TransfuserBEVSemanticClass.UNLABELED: rgb(0, 0, 0),
    TransfuserBEVSemanticClass.ROAD: rgb(250, 250, 250),
    TransfuserBEVSemanticClass.LANE_MARKERS: rgb(255, 255, 0),
    TransfuserBEVSemanticClass.STOP_SIGNS: rgb(160, 160, 0),
    TransfuserBEVSemanticClass.VEHICLE: rgb(15, 60, 255),
    TransfuserBEVSemanticClass.WALKER: rgb(0, 255, 0),
    TransfuserBEVSemanticClass.OBSTACLE: rgb(255, 0, 0),
    TransfuserBEVSemanticClass.PARKING_VEHICLE: rgb(116, 150, 65),
    TransfuserBEVSemanticClass.SPECIAL_VEHICLE: rgb(255, 0, 255),
    TransfuserBEVSemanticClass.BIKER: rgb(255, 0, 0),
    TransfuserBEVSemanticClass.TRAFFIC_GREEN: rgb(0, 255, 0),
    TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL: rgb(255, 0, 0),
    TransfuserBEVSemanticClass.TRAFFIC_RED_NOT_NORMAL: rgb(0, 0, 255),
}

# TransFuser++ semantic segmentation colors for CARLA data
TRANSFUSER_SEMANTIC_COLORS = {
    TransfuserSemanticSegmentationClass.UNLABELED: rgb(255, 255, 255),
    TransfuserSemanticSegmentationClass.VEHICLE: rgb(31, 119, 180),
    TransfuserSemanticSegmentationClass.ROAD: rgb(128, 64, 128),
    TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT: rgb(250, 170, 30),
    TransfuserSemanticSegmentationClass.PEDESTRIAN: rgb(0, 255, 60),
    TransfuserSemanticSegmentationClass.ROAD_LINE: rgb(157, 234, 50),
    TransfuserSemanticSegmentationClass.OBSTACLE: rgb(255, 0, 0),
    TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE: rgb(255, 255, 0),
    TransfuserSemanticSegmentationClass.STOP_SIGN: rgb(125, 0, 0),
    TransfuserSemanticSegmentationClass.BIKER: rgb(220, 20, 60),
}

# TransFuser++ bounding box colors
TRANSFUSER_BOUNDING_BOX_COLORS = {
    TransfuserBoundingBoxClass.VEHICLE: rgb(0, 0, 255),
    TransfuserBoundingBoxClass.WALKER: rgb(0, 255, 0),
    TransfuserBoundingBoxClass.TRAFFIC_LIGHT: rgb(255, 0, 0),
    TransfuserBoundingBoxClass.STOP_SIGN: rgb(250, 160, 160),
    TransfuserBoundingBoxClass.SPECIAL: rgb(0, 0, 255),
    TransfuserBoundingBoxClass.OBSTACLE: rgb(0, 255, 13),
    TransfuserBoundingBoxClass.PARKING: rgb(116, 150, 65),
    TransfuserBoundingBoxClass.BIKER: rgb(255, 0, 0),
}

# Mapping from CARLA semantic segmentation classes to TransFuser semantic segmentation classes
SEMANTIC_SEGMENTATION_CONVERTER = {
    CarlaSemanticSegmentationClass.Unlabeled: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Roads: TransfuserSemanticSegmentationClass.ROAD,
    CarlaSemanticSegmentationClass.SideWalks: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Building: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Wall: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Fence: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pole: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.TrafficLight: TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT,
    CarlaSemanticSegmentationClass.TrafficSign: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Vegetation: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Terrain: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Sky: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pedestrian: TransfuserSemanticSegmentationClass.PEDESTRIAN,
    CarlaSemanticSegmentationClass.Rider: TransfuserSemanticSegmentationClass.BIKER,
    CarlaSemanticSegmentationClass.Car: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Truck: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bus: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Train: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Motorcycle: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bicycle: TransfuserSemanticSegmentationClass.BIKER,
    CarlaSemanticSegmentationClass.Static: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Dynamic: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Other: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Water: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.RoadLine: TransfuserSemanticSegmentationClass.ROAD_LINE,
    CarlaSemanticSegmentationClass.Ground: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Bridge: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.RailTrack: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.GuardRail: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.ConeAndTrafficWarning: TransfuserSemanticSegmentationClass.OBSTACLE,
    CarlaSemanticSegmentationClass.SpecialVehicles: TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE,
    CarlaSemanticSegmentationClass.StopSign: TransfuserSemanticSegmentationClass.STOP_SIGN,
}

# Mapping from ChaffeurNet BEV semantic classes to TransFuser BEV semantic classes
CHAFFEURNET_TO_TRANSFUSER_BEV_SEMANTIC_CONVERTER = {
    ChaffeurNetBEVSemanticClass.UNLABELED: TransfuserBEVSemanticClass.UNLABELED,  # unlabeled
    ChaffeurNetBEVSemanticClass.ROAD: TransfuserBEVSemanticClass.ROAD,  # road
    ChaffeurNetBEVSemanticClass.SIDEWALK: TransfuserBEVSemanticClass.UNLABELED,  # sidewalk
    ChaffeurNetBEVSemanticClass.LANE_MARKERS: TransfuserBEVSemanticClass.LANE_MARKERS,  # lane_markers
    ChaffeurNetBEVSemanticClass.LANE_MARKERS_BROKEN: TransfuserBEVSemanticClass.LANE_MARKERS,  # lane_markers broken
    ChaffeurNetBEVSemanticClass.STOP_SIGNS: TransfuserBEVSemanticClass.STOP_SIGNS,  # stop_signs
    ChaffeurNetBEVSemanticClass.TRAFFIC_GREEN: TransfuserBEVSemanticClass.UNLABELED,  # traffic light green
    ChaffeurNetBEVSemanticClass.TRAFFIC_YELLOW: TransfuserBEVSemanticClass.UNLABELED,  # traffic light yellow
    ChaffeurNetBEVSemanticClass.TRAFFIC_RED: TransfuserBEVSemanticClass.UNLABELED,  # traffic light red
}

SCENARIO_TYPES = [
    "Accident",
    "AccidentTwoWays",
    "BlockedIntersection",
    "ConstructionObstacle",
    "ConstructionObstacleTwoWays",
    "ControlLoss",
    "CrossJunctionDefectTrafficLight",
    "CrossingBicycleFlow",
    "DynamicObjectCrossing",
    "EnterActorFlow",
    "EnterActorFlowV2",
    "HardBreakRoute",
    "HazardAtSideLane",
    "HazardAtSideLaneTwoWays",
    "HighwayCutIn",
    "HighwayExit",
    "InterurbanActorFlow",
    "InterurbanAdvancedActorFlow",
    "InvadingTurn",
    "MergerIntoSlowTraffic",
    "MergerIntoSlowTrafficV2",
    "NonSignalizedJunctionLeftTurn",
    "NonSignalizedJunctionLeftTurnEnterFlow",
    "NonSignalizedJunctionRightTurn",
    "noScenarios",
    "OppositeVehicleRunningRedLight",
    "OppositeVehicleTakingPriority",
    "ParkedObstacle",
    "ParkedObstacleTwoWays",
    "ParkingCrossingPedestrian",
    "ParkingCutIn",
    "ParkingExit",
    "PedestrianCrossing",
    "PriorityAtJunction",
    "RedLightWithoutLeadVehicle",
    "SequentialLaneChange",
    "SignalizedJunctionLeftTurn",
    "SignalizedJunctionLeftTurnEnterFlow",
    "SignalizedJunctionRightTurn",
    "StaticCutIn",
    "T_Junction",
    "VanillaNonSignalizedTurn",
    "VanillaNonSignalizedTurnEncounterStopsign",
    "VanillaSignalizedTurnEncounterGreenLight",
    "VanillaSignalizedTurnEncounterRedLight",
    "VehicleOpensDoorTwoWays",
    "VehicleTurningRoute",
    "VehicleTurningRoutePedestrian",
    "YieldToEmergencyVehicle",
    "NA",
]

EMERGENCY_MESHES = {
    "vehicle.dodge.charger_police_2020",
    "vehicle.dodge.charger_police",
    "vehicle.ford.ambulance",
    "vehicle.carlamotors.firetruck",
}

CONSTRUCTION_MESHES = {"static.prop.constructioncone", "static.prop.trafficwarning"}

BIKER_MESHES = {
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
    "vehicle.bh.crossbike",
}

URBAN_MAX_SPEED_LIMIT = 15
SUBURBAN_MAX_SPEED_LIMIT = 25
HIGHWAY_MAX_SPEED_LIMIT = 35

LOOKUP_TABLE = {
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": [
        2.44619083404541,
        1.115301489830017,
        0.7606233954429626,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": [
        2.5039126873016357,
        1.0485419034957886,
        0.7673624753952026,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": [
        2.2210919857025146,
        0.9388753771781921,
        0.9936029314994812,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": [
        2.6828393936157227,
        0.9732309579849243,
        0.7874829173088074,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": [
        2.782914400100708,
        1.217571496963501,
        1.022573471069336,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": [
        2.3368194103240967,
        1.0011461973190308,
        0.7259762287139893,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": [
        2.3958897590637207,
        1.081725001335144,
        0.7438300848007202,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": [
        2.2763495445251465,
        1.0926425457000732,
        0.8835831880569458,
    ],
    "/Game/Carla/Static/Dynamic/Garden/SM_PlasticTable.SM_PlasticTable": [
        1.241101622581482,
        1.241101622581482,
        1.239898920059204,
    ],
    "/Game/Carla/Static/Dynamic/Garden/SM_PlasticChair.SM_PlasticChair": [
        0.36523768305778503,
        0.37522444128990173,
        0.6356779336929321,
    ],
    "/Game/Carla/Static/Dynamic/Construction/SM_ConstructionCone.SM_ConstructionCone": [
        0.1720348298549652,
        0.1720348298549652,
        0.2928849756717682,
    ],
}

CARLA_MAP_PATHS = {
    **{
        location.lower(): f"CarlaUE4/Content/Carla/Maps/OpenDrive/{location}.xodr"
        for location in [
            "Town01",
            "Town02",
            "Town03",
            "Town04",
            "Town05",
            "Town06",
            "Town07",
            "Town10HD",
        ]
    },
    **{
        location.lower(): f"CarlaUE4/Content/Carla/Maps/{location}/OpenDrive/{location}.xodr"
        for location in ["Town11", "Town12", "Town13", "Town15"]
    },
}

CONSTRUCTION_CONE_BB_SIZE = [0.1720348298549652, 0.1720348298549652]

TRAFFIC_WARNING_BB_SIZE = [1.186714768409729, 1.4352929592132568]

OLD_TOWNS = {
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
}

# List of all CARLA towns for logging purposes
ALL_TOWNS = [
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town11",
    "Town12",
    "Town13",
    "Town15",
]

# Mapping from town name to a zero-padded index for logging (e.g., Town01 -> "01")
TOWN_NAME_TO_INDEX = {
    "Town01": "01",
    "Town02": "02",
    "Town03": "03",
    "Town04": "04",
    "Town05": "05",
    "Town06": "06",
    "Town07": "07",
    "Town10HD": "10HD",
    "Town11": "11",
    "Town12": "12",
    "Town13": "13",
    "Town15": "15",
}

# NavSim/NuPlan camera calibration parameters
NUPLAN_CAMERA_CALIBRATION = {
    "CAM_L0": {
        "pos": [0.2567803445203347, -0.14912709068475835, 1.9611907818710856],
        "rot": [-1.7908743283844297, -0.18030832979657968, -56.0098600797478],
        "fov": 63.71,
        "width": 1920 // 4,
        "height": 1120 // 4,
        "cropped_height": 1080 // 4,
    },
    "CAM_F0": {
        "pos": [0.3435966588946209, 0.00981503465349912, 1.9520959734648988],
        "rot": [0.004047525205852703, -2.344563417746492, -1.0253844128360994],
        "fov": 63.71,
        "width": 1920 // 4,
        "height": 1120 // 4,
        "cropped_height": 1080 // 4,
    },
    "CAM_R0": {
        "pos": [0.362775037177945, 0.16380984892156114, 1.9540305698009064],
        "rot": [0.8908895493227222, -0.5177262293066659, 54.26239302855094],
        "fov": 63.71,
        "width": 1920 // 4,
        "height": 1120 // 4,
        "cropped_height": 1080 // 4,
    },
    "CAM_B0": {
        "pos": [-1.8371898892894447, 0.023124645489646514, 1.9105230244574516],
        "rot": [0.004047525205852703, 1.9819092884563787, 180],
        "fov": 63.71,
        "width": 1920 // 4,
        "height": 1120 // 4,
        "cropped_height": 1080 // 4,
    },
}


class NavSimBEVSemanticClass(IntEnum):  # dead: disable
    """Indices to access BEV semantic map produced by NavSim.

    See: https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_config.py#L83
    """

    UNLABELED = 0
    ROAD = auto()
    WALKWAYS = auto()
    CENTERLINE = auto()
    STATIC_OBJECTS = auto()
    VEHICLES = auto()
    PEDESTRIANS = auto()


class NavSimBoundingBoxIndex(IntEnum):
    """Index to access NavSim bounding box attribute array.

    See: https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_features.py#L174
    """

    X = 0
    Y = 1
    HEADING = 2
    LENGTH = 3
    WIDTH = 4


class NavSimBBClass(IntEnum):
    """Bounding box classes used in NavSim."""

    GENERIC_CLASS = 0


class NavSimStatusFeature(IntEnum):  # dead: disable
    """Status feature indices used in NavSim.

    See: https://github.com/OpenDriveLab/OpenScene/blob/main/DriveEngine/process_data/helpers/driving_command.py#L40"""

    DRIVING_COMMAND_LEFT = 0
    DRIVING_COMMAND_STRAIGHT = 1
    DRIVING_COMMAND_RIGHT = 2
    DRIVING_COMMAND_UNKNOWN = 3
    EGO_VELOCITY_X = 4
    EGO_VELOCITY_Y = 5
    ACCELERATION_X = 6
    ACCELERATION_Y = 7


SIM2REAL_SEMANTIC_SEGMENTATION_CONVERTER = {
    TransfuserSemanticSegmentationClass.UNLABELED: TransfuserSemanticSegmentationClass.UNLABELED,
    TransfuserSemanticSegmentationClass.VEHICLE: TransfuserSemanticSegmentationClass.VEHICLE,
    TransfuserSemanticSegmentationClass.ROAD: TransfuserSemanticSegmentationClass.ROAD,
    TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT: TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT,
    TransfuserSemanticSegmentationClass.PEDESTRIAN: TransfuserSemanticSegmentationClass.PEDESTRIAN,
    TransfuserSemanticSegmentationClass.ROAD_LINE: TransfuserSemanticSegmentationClass.ROAD_LINE,
    TransfuserSemanticSegmentationClass.OBSTACLE: TransfuserSemanticSegmentationClass.OBSTACLE,
    TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE: TransfuserSemanticSegmentationClass.VEHICLE,
    TransfuserSemanticSegmentationClass.STOP_SIGN: TransfuserSemanticSegmentationClass.STOP_SIGN,
    TransfuserSemanticSegmentationClass.BIKER: TransfuserSemanticSegmentationClass.BIKER,
}


SIM2REAL_BEV_SEMANTIC_SEGMENTATION_CONVERTER = {
    TransfuserBEVSemanticClass.UNLABELED: TransfuserBEVSemanticClass.UNLABELED,
    TransfuserBEVSemanticClass.ROAD: TransfuserBEVSemanticClass.ROAD,
    TransfuserBEVSemanticClass.LANE_MARKERS: TransfuserBEVSemanticClass.LANE_MARKERS,
    TransfuserBEVSemanticClass.STOP_SIGNS: TransfuserBEVSemanticClass.UNLABELED,
    TransfuserBEVSemanticClass.VEHICLE: TransfuserBEVSemanticClass.VEHICLE,
    TransfuserBEVSemanticClass.WALKER: TransfuserBEVSemanticClass.WALKER,
    TransfuserBEVSemanticClass.OBSTACLE: TransfuserBEVSemanticClass.OBSTACLE,
    TransfuserBEVSemanticClass.PARKING_VEHICLE: TransfuserBEVSemanticClass.VEHICLE,
    TransfuserBEVSemanticClass.SPECIAL_VEHICLE: TransfuserBEVSemanticClass.SPECIAL_VEHICLE,
    TransfuserBEVSemanticClass.BIKER: TransfuserBEVSemanticClass.BIKER,
    TransfuserBEVSemanticClass.TRAFFIC_GREEN: TransfuserBEVSemanticClass.TRAFFIC_GREEN,
    TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL: TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL,
    TransfuserBEVSemanticClass.TRAFFIC_RED_NOT_NORMAL: TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL,
}

SIM2REAL_BOUNDING_BOX_CLASS_CONVERTER = {
    TransfuserBoundingBoxClass.VEHICLE: TransfuserBoundingBoxClass.VEHICLE,
    TransfuserBoundingBoxClass.WALKER: TransfuserBoundingBoxClass.WALKER,
    TransfuserBoundingBoxClass.TRAFFIC_LIGHT: TransfuserBoundingBoxClass.TRAFFIC_LIGHT,
    TransfuserBoundingBoxClass.STOP_SIGN: TransfuserBoundingBoxClass.STOP_SIGN,
    TransfuserBoundingBoxClass.SPECIAL: TransfuserBoundingBoxClass.VEHICLE,
    TransfuserBoundingBoxClass.OBSTACLE: TransfuserBoundingBoxClass.OBSTACLE,
    TransfuserBoundingBoxClass.PARKING: TransfuserBoundingBoxClass.VEHICLE,
    TransfuserBoundingBoxClass.BIKER: TransfuserBoundingBoxClass.BIKER,
}

SIM2REAL_BEV_OCCUPANCY_CLASS_CONVERTER = {
    TransfuserBEVOccupancyClass.UNLABELED: TransfuserBEVOccupancyClass.UNLABELED,
    TransfuserBEVOccupancyClass.VEHICLE: TransfuserBEVOccupancyClass.VEHICLE,
    TransfuserBEVOccupancyClass.WALKER: TransfuserBEVOccupancyClass.WALKER,
    TransfuserBEVOccupancyClass.OBSTACLE: TransfuserBEVOccupancyClass.UNLABELED,
    TransfuserBEVOccupancyClass.PARKING_VEHICLE: TransfuserBEVOccupancyClass.VEHICLE,
    TransfuserBEVOccupancyClass.SPECIAL_VEHICLE: TransfuserBEVOccupancyClass.VEHICLE,
    TransfuserBEVOccupancyClass.BIKER: TransfuserBEVOccupancyClass.BIKER,
    TransfuserBEVOccupancyClass.TRAFFIC_GREEN: TransfuserBEVOccupancyClass.TRAFFIC_GREEN,
    TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL: TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL,
    TransfuserBEVOccupancyClass.TRAFFIC_RED_NOT_NORMAL: TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL,
}

CARLA_REAR_AXLE = [
    -1.389,
    0.0,
    0.360,
]  # Rear axle position relative to the vehicle center

# Waymo E2E 2025 camera intrinsics
WAYMO_E2E_INTRINSIC = [
    1112.17806333,
    1112.04114501,
    488.128479,
    719.15600586,
    -0.073584,
    -0.036582,
    0.0,
    0.0,
    0.0,
]

# Waymo E2E 2025 camera extrinsics
WAYMO_E2E_2025_CAMERA_SETTING = {
    "FRONT_LEFT": {
        "extrinsic": [
            [0.70549636, -0.70869903, -0.00026687, 1.44509995],
            [0.70868651, 0.70547806, 0.00679447, 0.15270001],
            [-0.00462506, -0.00498361, 0.99996268, 1.80649996],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 972,
        "height": 1440,
        "cropped_height": 1080,
    },
    "FRONT": {
        "extrinsic": [
            [0.99998026, -0.00215741, 0.00378466, 1.51909995],
            [0.00214115, 0.99997577, 0.0045841, 0.0258],
            [-0.00379113, -0.00457491, 0.9999674, 1.80649996],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 972,
        "height": 1440,
        "cropped_height": 1080,
    },
    "FRONT_RIGHT": {
        "extrinsic": [
            [0.70908528, 0.70508644, 0.00506919, 1.48150003],
            [-0.7050955, 0.70909768, -0.00009787, -0.1163],
            [-0.00366141, -0.00350169, 0.99996981, 1.80649996],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 972,
        "height": 1440,
        "cropped_height": 1080,
    },
}


class CarlaImageCroppingType(IntEnum):
    TOP = 0
    BOTTOM = auto()
    BOTH = auto()
    NONE = auto()


WAYMO_DOWN_SAMPLE_FACTOR = 3  # Down-sample factor for Waymo E2E 2025 images
WAYMO_E2E_REAL_DATA_JPEG_LEVEL = (
    50  # JPEG compression level for Waymo E2E 2025 real data
)
