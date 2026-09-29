"""
Functionality used to store CARLA maps as h5 files.
Code adapted from https://github.com/zhejz/carla-roach
"""

import argparse
import socket
import subprocess
import time
from pathlib import Path

import carla
import cv2 as cv
import h5py
import numpy as np
import psutil
from tqdm import tqdm
from traffic_light import TrafficLightHandler

COLOR_WHITE = (255, 255, 255)


def kill(proc_pid):
    if psutil.pid_exists(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()


def next_free_port(port=1024, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise OSError("no free ports")


def kill_all_carla_servers(ports):
    # Need a failsafe way to find and kill all carla servers. We do so by port.
    for proc in psutil.process_iter():
        # check whether the process name matches
        try:
            proc_connections = proc.connections(kind="all")
        except (PermissionError, psutil.AccessDenied):  # Avoid sudo processes
            proc_connections = None

        if proc_connections is not None:
            for conns in proc_connections:
                if not isinstance(conns.laddr, str):  # Avoid unix paths
                    if conns.laddr.port in ports:
                        try:
                            proc.kill()
                        except (
                            psutil.NoSuchProcess
                        ):  # Catch the error caused by the process no longer existing
                            pass  # Ignore it


class MapImage:
    """
    Functionality used to store CARLA maps as h5 files.
    """

    @staticmethod
    def draw_map_image(carla_map_local, pixels_per_meter_local, precision=0.05):
        waypoints = carla_map_local.generate_waypoints(2)
        margin = 100
        max_x = (
            max(waypoints, key=lambda x: x.transform.location.x).transform.location.x
            + margin
        )
        max_y = (
            max(waypoints, key=lambda x: x.transform.location.y).transform.location.y
            + margin
        )
        min_x = (
            min(waypoints, key=lambda x: x.transform.location.x).transform.location.x
            - margin
        )
        min_y = (
            min(waypoints, key=lambda x: x.transform.location.y).transform.location.y
            - margin
        )

        world_offset = np.array([min_x, min_y], dtype=np.float32)
        width_in_meters = max(max_x - min_x, max_y - min_y)
        width_in_pixels = round(pixels_per_meter_local * width_in_meters)

        road_surface = np.zeros((width_in_pixels, width_in_pixels))
        shoulder_surface = np.zeros((width_in_pixels, width_in_pixels))
        parking_surface = np.zeros((width_in_pixels, width_in_pixels))
        sidewalk_surface = np.zeros((width_in_pixels, width_in_pixels))
        lane_marking_yellow_broken_surface = np.zeros(
            (width_in_pixels, width_in_pixels),
        )
        lane_marking_yellow_solid_surface = np.zeros((width_in_pixels, width_in_pixels))
        lane_marking_white_broken_surface = np.zeros((width_in_pixels, width_in_pixels))
        lane_marking_white_solid_surface = np.zeros((width_in_pixels, width_in_pixels))
        lane_marking_all_surface = np.zeros((width_in_pixels, width_in_pixels))

        topology = [x[0] for x in carla_map_local.get_topology()]
        topology = sorted(topology, key=lambda w: w.transform.location.z)

        for waypoint in tqdm(topology):
            waypoints = [waypoint]
            # Generate waypoints of a road id. Stop when road id differs
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break

            # Draw Shoulders, Parkings and Sidewalks
            shoulder = [[], []]
            parking = [[], []]
            sidewalk = [[], []]

            for w in waypoints:
                # Classify lane types until there are no waypoints by going left
                left = w.get_left_lane()
                while left and left.lane_type != carla.LaneType.Driving:
                    if left.lane_type == carla.LaneType.Shoulder:
                        shoulder[0].append(left)
                    if left.lane_type == carla.LaneType.Parking:
                        parking[0].append(left)
                    if left.lane_type == carla.LaneType.Sidewalk:
                        sidewalk[0].append(left)
                    left = left.get_left_lane()
                # Classify lane types until there are no waypoints by going right
                right = w.get_right_lane()
                while right and right.lane_type != carla.LaneType.Driving:
                    if right.lane_type == carla.LaneType.Shoulder:
                        shoulder[1].append(right)
                    if right.lane_type == carla.LaneType.Parking:
                        parking[1].append(right)
                    if right.lane_type == carla.LaneType.Sidewalk:
                        sidewalk[1].append(right)
                    right = right.get_right_lane()

            MapImage.draw_lane(
                road_surface,
                waypoints,
                COLOR_WHITE,
                pixels_per_meter_local,
                world_offset,
            )
            MapImage.draw_lane(
                sidewalk_surface,
                sidewalk[0],
                COLOR_WHITE,
                pixels_per_meter_local,
                world_offset,
            )
            MapImage.draw_lane(
                sidewalk_surface,
                sidewalk[1],
                COLOR_WHITE,
                pixels_per_meter_local,
                world_offset,
            )
            MapImage.draw_lane(
                shoulder_surface,
                shoulder[0],
                COLOR_WHITE,
                pixels_per_meter_local,
                world_offset,
            )
            MapImage.draw_lane(
                shoulder_surface,
                shoulder[1],
                COLOR_WHITE,
                pixels_per_meter_local,
                world_offset,
            )
            MapImage.draw_lane(
                parking_surface,
                parking[0],
                COLOR_WHITE,
                pixels_per_meter_local,
                world_offset,
            )
            MapImage.draw_lane(
                parking_surface,
                parking[1],
                COLOR_WHITE,
                pixels_per_meter_local,
                world_offset,
            )

            if not waypoint.is_junction:
                MapImage.draw_lane_marking_single_side(
                    lane_marking_yellow_broken_surface,
                    lane_marking_yellow_solid_surface,
                    lane_marking_white_broken_surface,
                    lane_marking_white_solid_surface,
                    lane_marking_all_surface,
                    waypoints,
                    -1,
                    pixels_per_meter_local,
                    world_offset,
                )
                MapImage.draw_lane_marking_single_side(
                    lane_marking_yellow_broken_surface,
                    lane_marking_yellow_solid_surface,
                    lane_marking_white_broken_surface,
                    lane_marking_white_solid_surface,
                    lane_marking_all_surface,
                    waypoints,
                    1,
                    pixels_per_meter_local,
                    world_offset,
                )

        # stoplines
        stopline_surface = np.zeros((width_in_pixels, width_in_pixels))

        for stopline_vertices in TrafficLightHandler.list_stopline_vtx:
            for loc_left, loc_right in stopline_vertices:
                stopline_points = [
                    MapImage.world_to_pixel(
                        loc_left,
                        pixels_per_meter_local,
                        world_offset,
                    ),
                    MapImage.world_to_pixel(
                        loc_right,
                        pixels_per_meter_local,
                        world_offset,
                    ),
                ]
                MapImage.draw_line(stopline_surface, stopline_points, 2)

        # np.uint8 mask
        def _make_mask(x):
            return x.astype(np.uint8).T  # Change coordinate system to match pygame

        # make a dict
        dict_masks_local = {
            "road": _make_mask(road_surface),
            "shoulder": _make_mask(shoulder_surface),
            "parking": _make_mask(parking_surface),
            "sidewalk": _make_mask(sidewalk_surface),
            "lane_marking_yellow_broken": _make_mask(
                lane_marking_yellow_broken_surface,
            ),
            "lane_marking_yellow_solid": _make_mask(lane_marking_yellow_solid_surface),
            "lane_marking_white_broken": _make_mask(lane_marking_white_broken_surface),
            "lane_marking_white_solid": _make_mask(lane_marking_white_solid_surface),
            "lane_marking_all": _make_mask(lane_marking_all_surface),
            "stopline": _make_mask(stopline_surface),
            "world_offset": world_offset,
            "pixels_per_meter": pixels_per_meter_local,
            "width_in_meters": width_in_meters,
            "width_in_pixels": width_in_pixels,
        }
        return dict_masks_local

    @staticmethod
    def draw_lane_marking_single_side(
        lane_marking_yellow_broken_surface,
        lane_marking_yellow_solid_surface,
        lane_marking_white_broken_surface,
        lane_marking_white_solid_surface,
        lane_marking_all_surface,
        waypoints,
        sign,
        pixels_per_meter_local,
        world_offset,
    ):
        """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
        the waypoint based on the sign parameter"""
        lane_marking = None
        previous_marking_type = carla.LaneMarkingType.NONE
        previous_marking_color = carla.LaneMarkingColor.Other
        current_lane_marking = carla.LaneMarkingType.NONE

        markings_list = []
        temp_waypoints = []
        for sample in waypoints:
            lane_marking = (
                sample.left_lane_marking if sign < 0 else sample.right_lane_marking
            )

            if lane_marking is None:
                continue

            if current_lane_marking != lane_marking.type:
                # Get the list of lane markings to draw
                markings = MapImage.get_lane_markings(
                    previous_marking_type,
                    previous_marking_color,
                    temp_waypoints,
                    sign,
                    pixels_per_meter_local,
                    world_offset,
                )
                current_lane_marking = lane_marking.type

                # Append each lane marking in the list
                for marking in markings:
                    markings_list.append(marking)

                temp_waypoints = temp_waypoints[-1:]

            else:
                temp_waypoints.append(sample)
                previous_marking_type = lane_marking.type
                previous_marking_color = lane_marking.color

        # Add last marking
        last_markings = MapImage.get_lane_markings(
            previous_marking_type,
            previous_marking_color,
            temp_waypoints,
            sign,
            pixels_per_meter_local,
            world_offset,
        )

        for marking in last_markings:
            markings_list.append(marking)
        # Once the lane markings have been simplified to Solid or Broken lines, we draw them
        for markings in markings_list:
            if (
                markings[1] == carla.LaneMarkingColor.White
                and markings[0] == carla.LaneMarkingType.Solid
            ):
                MapImage.draw_line(lane_marking_white_solid_surface, markings[2], 1)
            elif (
                markings[1] == carla.LaneMarkingColor.Yellow
                and markings[0] == carla.LaneMarkingType.Solid
            ):
                MapImage.draw_line(lane_marking_yellow_solid_surface, markings[2], 1)
            elif (
                markings[1] == carla.LaneMarkingColor.White
                and markings[0] == carla.LaneMarkingType.Broken
            ):
                MapImage.draw_line(lane_marking_white_broken_surface, markings[2], 1)
            elif (
                markings[1] == carla.LaneMarkingColor.Yellow
                and markings[0] == carla.LaneMarkingType.Broken
            ):
                MapImage.draw_line(lane_marking_yellow_broken_surface, markings[2], 1)

            MapImage.draw_line(lane_marking_all_surface, markings[2], 1)

    @staticmethod
    def get_lane_markings(
        lane_marking_type,
        lane_marking_color,
        waypoints,
        sign,
        pixels_per_meter_local,
        world_offset,
    ):
        """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
        as a combination of Broken and Solid lines"""
        margin = 0.25
        marking_1 = [
            MapImage.world_to_pixel(
                MapImage.lateral_shift(w.transform, sign * w.lane_width * 0.5),
                pixels_per_meter_local,
                world_offset,
            )
            for w in waypoints
        ]

        if lane_marking_type in (
            carla.LaneMarkingType.Broken,
            carla.LaneMarkingType.Solid,
        ):
            return [(lane_marking_type, lane_marking_color, marking_1)]
        else:
            marking_2 = [
                MapImage.world_to_pixel(
                    MapImage.lateral_shift(
                        w.transform,
                        sign * (w.lane_width * 0.5 + margin * 2),
                    ),
                    pixels_per_meter_local,
                    world_offset,
                )
                for w in waypoints
            ]
            if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                return [
                    (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                    (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
                ]
            elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                return [
                    (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                    (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
                ]
            elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                return [
                    (carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                    (carla.LaneMarkingType.Broken, lane_marking_color, marking_2),
                ]
            elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                return [
                    (carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                    (carla.LaneMarkingType.Solid, lane_marking_color, marking_2),
                ]
        return [(carla.LaneMarkingType.NONE, lane_marking_color, marking_1)]

    @staticmethod
    def draw_line(surface, points, width):
        """Draws solid lines in a surface given a set of points, width and color"""
        if len(points) >= 2:
            cv.polylines(
                surface,
                np.array([points], dtype=np.int32),
                False,
                255,
                thickness=1,
            )

    @staticmethod
    def draw_lane(surface, wp_list, color, pixels_per_meter_local, world_offset):
        """Renders a single lane in a surface and with a specified color"""
        lane_left_side = [
            MapImage.lateral_shift(w.transform, -w.lane_width * 0.5) for w in wp_list
        ]
        lane_right_side = [
            MapImage.lateral_shift(w.transform, w.lane_width * 0.5) for w in wp_list
        ]

        polygon = lane_left_side + list(reversed(lane_right_side))
        polygon = [
            MapImage.world_to_pixel(x, pixels_per_meter_local, world_offset)
            for x in polygon
        ]

        if len(polygon) > 2:
            cv.fillPoly(surface, np.array([polygon], dtype=np.int32), 255)

    @staticmethod
    def lateral_shift(transform, shift):
        """Makes a lateral shift of the forward vector of a transform"""
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    @staticmethod
    def world_to_pixel(location, pixels_per_meter_local, world_offset):
        """Converts the world coordinates to pixel coordinates"""
        x = pixels_per_meter_local * (location.x - world_offset[0])
        y = pixels_per_meter_local * (location.y - world_offset[1])
        return [int(round(y)), int(round(x))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="carla_gym/core/obs_manager/birdview/maps",
    )
    parser.add_argument("--pixels_per_meter", type=float, default=2.0)
    parser.add_argument(
        "--carla_sh_path",
        default="/home/ubuntu/apps/carla/carla994/CarlaUE4.sh",
    )
    parser.add_argument(
        "--carla_root",
        type=str,
        default=r"/home/jaeger/ordnung/internal/carla_9_15",
        help="folder containing carla",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="id to run the carla server on",
    )

    args = parser.parse_args()
    client_ports = []

    current_port_0 = next_free_port(1024)
    current_port_1 = current_port_0 + 3
    current_port_1 = next_free_port(current_port_1)
    carla_servers = []
    client_ports.append(current_port_0)
    carla_servers.append(
        subprocess.Popen(  # pylint: disable=locally-disabled, consider-using-with
            f"bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={current_port_0} -nosound -nullrhi -RenderOffScreen -carla-streaming-port={current_port_1} -graphicsadapter={args.gpu_id}",  # noqa: E501
            shell=True,
        ),
    )
    time.sleep(60)
    if carla_servers[0].poll() is not None:
        print("Carla server crashed")

    # kill running carla server
    # with subprocess.Popen('killall -9 -r CarlaUE4-Linux', shell=True) as kill_process:
    #  kill_process.wait()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pixels_per_meter = float(args.pixels_per_meter)
    client = carla.Client("localhost", current_port_0)
    client.set_timeout(1000)
    settings = carla.WorldSettings(
        synchronous_mode=True,
        fixed_delta_seconds=0.1,
        deterministic_ragdolls=True,
        no_rendering_mode=False,
        spectator_as_ego=False,
    )
    client.get_world().apply_settings(settings)
    map_names = [
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
    for carla_map in map_names:
        hf_file_path = save_dir / (carla_map + ".h5")

        # pass if map h5 already exists
        if hf_file_path.exists():
            map_hf = h5py.File(hf_file_path, "r")
            hf_pixels_per_meter = float(map_hf.attrs["pixels_per_meter"])
            map_hf.close()
            if np.isclose(hf_pixels_per_meter, pixels_per_meter):
                print(
                    f"{carla_map}.h5 with pixels_per_meter={pixels_per_meter:.2f} already exists.",
                )
                continue

        print(
            f"Generating {carla_map}.h5 with pixels_per_meter={pixels_per_meter:.2f}.",
        )
        world = client.load_world(carla_map, reset_settings=False)

        dict_masks = MapImage.draw_map_image(world.get_map(), pixels_per_meter)

        with h5py.File(hf_file_path, "w") as hf:
            hf.attrs["pixels_per_meter"] = pixels_per_meter
            hf.attrs["world_offset_in_meters"] = dict_masks["world_offset"]
            hf.attrs["width_in_meters"] = dict_masks["width_in_meters"]
            hf.attrs["width_in_pixels"] = dict_masks["width_in_pixels"]
            hf.create_dataset(
                "road",
                data=dict_masks["road"],
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "shoulder",
                data=dict_masks["shoulder"],
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "parking",
                data=dict_masks["parking"],
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "sidewalk",
                data=dict_masks["sidewalk"],
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "stopline",
                data=dict_masks["stopline"],
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "lane_marking_all",
                data=dict_masks["lane_marking_all"],
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "lane_marking_yellow_broken",
                data=dict_masks["lane_marking_yellow_broken"],
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "lane_marking_yellow_solid",
                data=dict_masks["lane_marking_yellow_solid"],
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "lane_marking_white_broken",
                data=dict_masks["lane_marking_white_broken"],
                compression="gzip",
                compression_opts=9,
            )
            hf.create_dataset(
                "lane_marking_white_solid",
                data=dict_masks["lane_marking_white_solid"],
                compression="gzip",
                compression_opts=9,
            )

    kill(carla_servers[0].pid)
    print("Done generating routes.")
    kill_all_carla_servers(client_ports)
    del carla_servers
