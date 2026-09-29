#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import numpy as np
import json

try:
	import pygame
	from pygame.locals import K_DOWN
	from pygame.locals import K_LEFT
	from pygame.locals import K_RIGHT
	from pygame.locals import K_SPACE
	from pygame.locals import K_UP
	from pygame.locals import K_a
	from pygame.locals import K_d
	from pygame.locals import K_s
	from pygame.locals import K_w
	from pygame.locals import K_q
	from pygame.locals import K_r
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla
import math
import os
import time
from datetime import datetime, timedelta

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
	def __init__(self, font, dim, pos):
		self.font = font
		self.dim = dim
		self.pos = pos
		self.seconds_left = 0
		self.surface = pygame.Surface(self.dim)

	def set_text(self, text, color=(255, 255, 255), seconds=2.0):
		text_texture = self.font.render(text, True, color)
		self.surface = pygame.Surface(self.dim)
		self.seconds_left = seconds
		self.surface.fill((0, 0, 0, 0))
		self.surface.blit(text_texture, (10, 11))

	def tick(self, delta_seconds):
		self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
		self.surface.set_alpha(500.0 * self.seconds_left)

	def render(self, display):
		display.blit(self.surface, self.pos)


def get_entry_point():
	return 'HumanAgentSteeringWheel'


class HumanInterface(object):
	"""
	Class to control a vehicle manually for debugging purposes
	"""

	def __init__(self, width, height, side_scale, vehicle, map, left_mirror=False, right_mirror=False):
		self._width = width
		self._height = height
		self.dim = (width, height)
		self._scale = side_scale
		self._surface = None
		self.camera_idx = 0

		self._left_mirror = left_mirror
		self._right_mirror = right_mirror

		pygame.init()
		pygame.font.init()
		self._clock = pygame.time.Clock()
		self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
		pygame.display.set_caption("Human Agent")

		font = pygame.font.Font(pygame.font.get_default_font(), 20)
		self._notifications = FadingText(font, (width, 40), (0, height - 40))

		font_name = 'courier' if os.name == 'nt' else 'mono'
		fonts = [x for x in pygame.font.get_fonts() if font_name in x]
		default_font = 'ubuntumono'
		mono = default_font if default_font in fonts else fonts[0]
		mono = pygame.font.match_font(mono)
		self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
		self._font_speed = pygame.font.Font(mono, 50)

		self._info_text = []

		self._show_info = True

	def tick(self, control, fps, timestamp):
		self._info_text = [
			'Framerate:  % 14.0f FPS' % fps,
			'',
			'Vehicle: % 20s' % self.vehicle,
			'Map:     % 20s' % self.map,
			'Time:            % 12s' % timedelta(seconds=int(timestamp)),
			'',
			'Speed:%2.0f km/h' % 1,
			'',
			'']

		self._info_text += [
			('Throttle:', control.throttle, 0.0, 1.0),
			('Steer:', control.steer, -1.0, 1.0),
			('Reverse:', control.reverse),
		]

	def run_interface(self, input_data):
		"""
		Run the GUI
		"""

		# Process sensor data
		if self.camera_idx == 0:
			image_center = input_data['Center_0'][1][:, :, -2::-1]
		elif self.camera_idx == 1:
			image_center = input_data['Center_1'][1][:, :, -2::-1]
		# elif self.camera_idx == 2:
		#     image_center = input_data['Center_2'][1][:, :, -2::-1]
		else:
			raise NotImplementedError('This camera index is not implemented yet.')

		self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))

		# Add the left mirror
		if self._left_mirror:
			image_left = input_data['Left'][1][:, :, -2::-1]
			left_surface = pygame.surfarray.make_surface(image_left.swapaxes(0, 1))
			self._surface.blit(left_surface, (0, (1 - self._scale) * self._height))

		# Add the right mirror
		if self._right_mirror:
			image_right = input_data['Right'][1][:, :, -2::-1]
			right_surface = pygame.surfarray.make_surface(image_right.swapaxes(0, 1))
			self._surface.blit(right_surface, ((1 - self._scale) * self._width, (1 - self._scale) * self._height))

		# Display image
		if self._surface is not None:
			self._display.blit(self._surface, (0, 0))

		if self._show_info:
			surface_height = (1 - self._scale) * self._height if self._left_mirror else self._height
			info_surface = pygame.Surface((220, surface_height))
			info_surface.set_alpha(100)
			self._display.blit(info_surface, (0, 0))
			v_offset = 4
			bar_h_offset = 100
			bar_width = 106
			for item in self._info_text:
				if v_offset + 18 > self.dim[1]:
					break
				if isinstance(item, list):
					if len(item) > 1:
						points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
						pygame.draw.lines(self._display, (255, 136, 0), False, points, 2)
					item = None
					v_offset += 18
				elif isinstance(item, tuple):
					if isinstance(item[1], bool):
						rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
						pygame.draw.rect(self._display, (255, 255, 255), rect, 0 if item[1] else 1)
					else:
						rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
						pygame.draw.rect(self._display, (255, 255, 255), rect_border, 1)
						f = (item[1] - item[2]) / (item[3] - item[2])
						if item[2] < 0.0:
							rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
						else:
							rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
						pygame.draw.rect(self._display, (255, 255, 255), rect)
					item = item[0]
				if item:  # At this point has to be a str.
					if item.startswith('Speed'):
						surface = self._font_speed.render(' Speed', True, (255, 255, 255))
						self._display.blit(surface, (8, v_offset))
						v_offset += 50
						surface = self._font_speed.render("{:5.1f}".format(3.6 * input_data['speedometer'][1]['speed']),
						                                  True, (255, 255, 255))
					else:
						surface = self._font_mono.render(item, True, (255, 255, 255))
					self._display.blit(surface, (8, v_offset))
				v_offset += 18

		self._notifications.render(self._display)

		pygame.display.flip()

	def set_black_screen(self):
		"""Set the surface to black"""
		black_array = np.zeros([self._width, self._height])
		self._surface = pygame.surfarray.make_surface(black_array)
		if self._surface is not None:
			self._display.blit(self._surface, (0, 0))
		pygame.display.flip()

	def _quit(self):
		pygame.quit()


class HumanAgentSteeringWheel(AutonomousAgent):
	"""
	Human agent to control the ego vehicle via keyboard
	"""

	current_control = None
	agent_engaged = False

	def setup(self, path_to_conf_file):
		"""
		Setup the agent parameters
		"""
		self.track = Track.SENSORS

		self.agent_engaged = False
		self.camera_width = 1280
		self.camera_height = 720
		self._side_scale = 0.3
		self._left_mirror = False
		self._right_mirror = False

		self._hic = HumanInterface(
			self.camera_width,
			self.camera_height,
			self._side_scale,
			self._left_mirror,
			self._right_mirror
		)

		self._controller = SteeringWheelControl(path_to_conf_file, self._hic)
		self._prev_timestamp = 0

		self._clock = pygame.time.Clock()

	def sensors(self):
		"""
		Define the sensor suite required by the agent

		:return: a list containing the required sensors in the following format:

		[
			{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					  'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

			{'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					  'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

			{'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
			 'id': 'LIDAR'}
		]
		"""

		sensors = [
			# {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
			#  'width': self.camera_width, 'height': self.camera_height, 'fov': 100, 'id': 'Center_0'},

			{'type': 'sensor.camera.rgb', 'x': -1.5, 'y': 0.0, 'z': 2.6, 'roll': 0.0, 'pitch': -15.0, 'yaw': 0.0,
			 'width': self.camera_width, 'height': self.camera_height, 'fov': 100, 'id': 'Center_0'},

			{'type': 'sensor.speedometer', 'id': 'speedometer'}
		]

		# if self._left_mirror:
		#     sensors.append(
		#         {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -1.0, 'z': 1, 'roll': 0.0, 'pitch': 0.0, 'yaw': 210.0,
		#          'width': self.camera_width * self._side_scale, 'height': self.camera_height * self._side_scale,
		#          'fov': 100, 'id': 'Left'})
		#
		# if self._right_mirror:
		#     sensors.append(
		#         {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 1.0, 'z': 1, 'roll': 0.0, 'pitch': 0.0, 'yaw': 150.0,
		#          'width': self.camera_width * self._side_scale, 'height': self.camera_height * self._side_scale,
		#          'fov': 100, 'id': 'Right'})

		return sensors

	def run_step(self, input_data, timestamp):
		"""
		Execute one step of navigation.
		"""

		passed_milliseconds = self._clock.tick_busy_loop(20)
		self.agent_engaged = True
		self._controller.tick(passed_milliseconds, self._clock.get_fps(), timestamp)
		self._hic.run_interface(input_data)

		control = self._controller.parse_events(timestamp - self._prev_timestamp)
		self._prev_timestamp = timestamp

		return control

	def destroy(self):
		"""
		Cleanup
		"""
		self._hic.set_black_screen()
		self._hic._quit = True


class SteeringWheelControl(object):
	"""
	Keyboard control for the human agent
	"""

	def __init__(self, path_to_conf_file, hic):
		"""
		Init
		"""
		self._control = carla.VehicleControl()
		self._steer_cache = 0.0
		self._clock = pygame.time.Clock()
		self._hic = hic
		self.recording = False
		self.client = carla.Client('localhost', 2000)
		self._hic.map = self.client.get_world().get_map().name.split('/')[-1]
		self._hic.vehicle = 'Lincoln - MKZ 2020'

		self._notifications = hic._notifications

		# Get the mode
		if path_to_conf_file:
			with (open(path_to_conf_file, "r")) as f:
				lines = f.read().split("\n")
				self._mode = lines[0].split(" ")[1]
				self._endpoint = lines[1].split(" ")[1]

			# Get the needed vars
			if self._mode == "log":
				self._log_data = {'records': []}

			elif self._mode == "playback":
				self._index = 0
				self._control_list = []

				with open(self._endpoint) as fd:
					try:
						self._records = json.load(fd)
						self._json_to_control()
					except json.JSONDecodeError:
						pass

		else:
			self._mode = "normal"
			self._endpoint = None

			# initialize steering wheel
			pygame.joystick.init()

			joystick_count = pygame.joystick.get_count()
			if joystick_count > 1:
				raise ValueError("Please Connect Just One Joystick")

			self._joystick = pygame.joystick.Joystick(0)
			self._joystick.init()

			self._steer_idx = 0
			self._throttle_idx = 2
			self._brake_idx = 3
			self._reverse_idx = 5
			self._handbrake_idx = 4

		# start recording
		self.toggle_recording()


	def tick(self, passed_millis, fps, timestamp):
		self._notifications.tick(passed_millis)
		self._hic.tick(self._control, fps, timestamp)

	def _json_to_control(self):
		# transform strs into VehicleControl commands
		for entry in self._records['records']:
			control = carla.VehicleControl(throttle=entry['control']['throttle'],
			                               steer=entry['control']['steer'],
			                               brake=entry['control']['brake'],
			                               hand_brake=entry['control']['hand_brake'],
			                               reverse=entry['control']['reverse'],
			                               manual_gear_shift=entry['control']['manual_gear_shift'],
			                               gear=entry['control']['gear'])
			self._control_list.append(control)


	def parse_events(self, timestamp):
		"""
		Parse the keyboard events and set the vehicle controls accordingly
		"""
		# Move the vehicle
		if self._mode == "playback":
			self._parse_json_control()
		else:
			self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp * 1000)

		# Record the control
		if self._mode == "log":
			self._record_control()

		return self._control


	def toggle_recording(self):
		self.recording = not self.recording
		text = "Start recording" if self.recording else "Stop recording"
		self._notifications.set_text(text, color=(255, 255, 255), seconds=2.0)
		if self.recording:
			print('Started recording - {}'.format(self._endpoint))
			if not os.path.exists(os.path.join(os.getcwd(), 'logs')):
				os.mkdir(os.path.join(os.getcwd(), 'logs'))
			self.client.start_recorder(os.path.join(os.getcwd(), 'logs/log_{}.log'.format(datetime.now())))
		else:
			self.client.stop_recorder()

	def _parse_vehicle_keys(self, keys, milliseconds):
		"""
		Calculate new vehicle controls based on input keys
		"""

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return True
			elif event.type == pygame.JOYBUTTONDOWN:
				if event.button == 1:
					self._hic._show_info = not self._hic._show_info
				elif event.button == 4:
					pass
					# self._hic._left_mirror = False if self._hic._left_mirror else True
					# self._hic._right_mirror = False if self._hic._right_mirror else True
				# elif event.button == 3: not necessary, it's always good weather
				# world.next_weather()
				# print('next weather')
				elif event.button == 0:
					self._hic.camera_idx = 0
				elif event.button == 5:
					self._control.gear = 1 if self._control.reverse else -1
					self._control.reverse = self._control.gear < 0
			elif event.type == pygame.KEYUP:
				if event.key == K_r:
					self.toggle_recording()

		numAxes = self._joystick.get_numaxes()
		jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
		# print (jsInputs)
		# jsButtons = [float(self._joystick.get_button(i)) for i in
		#              range(self._joystick.get_numbuttons())]

		# Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
		# For the steering, it seems fine as it is
		K1 = 1.0  # 0.55
		steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

		K2 = 1.6  # 1.6
		throttleCmd = K2 + (2.05 * math.log10(
			-0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
		if throttleCmd <= 0:
			throttleCmd = 0
		elif throttleCmd > 1:
			throttleCmd = 1

		brakeCmd = 1.6 + (2.05 * math.log10(
			-0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
		if brakeCmd <= 0:
			brakeCmd = 0
		elif brakeCmd > 1:
			brakeCmd = 1

		self._control.steer = steerCmd
		self._control.brake = brakeCmd
		self._control.throttle = throttleCmd


	def _parse_json_control(self):
		if self._index < len(self._control_list):
			self._control = self._control_list[self._index]
			self._index += 1
		else:
			print("JSON file has no more entries")


	def _record_control(self):
		new_record = {
			'control': {
				'throttle': self._control.throttle,
				'steer': self._control.steer,
				'brake': self._control.brake,
				'hand_brake': self._control.hand_brake,
				'reverse': self._control.reverse,
				'manual_gear_shift': self._control.manual_gear_shift,
				'gear': self._control.gear
			}
		}

		self._log_data['records'].append(new_record)

	def __del__(self):
		if self.recording:
			self.toggle_recording()

		# Get ready to log user commands

		if self._mode == "log" and self._log_data:
			with open(self._endpoint, 'w') as fd:
				json.dump(self._log_data, fd, indent=4, sort_keys=True)
