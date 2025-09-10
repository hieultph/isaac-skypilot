#!/usr/bin/env python3
"""
Comprehensive AI-Controlled Drone System
Combines takeoff, AI model integration, and fluid action execution
"""

import cv2
import numpy as np
import requests
import json_numpy
import threading
import time
from pymavlink import mavutil
import queue
import sys
from typing import Dict, Any, Optional

# Patch json to handle numpy arrays
json_numpy.patch()

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Configuration
AI_MODEL_API_URL = "http://159.223.171.199:11701/act"
AIRSIM_VIDEO_URL = "http://127.0.0.1:5000/video_feed"
MAVLINK_CONNECTION = 'udpin:0.0.0.0:14555'

# Video Configuration
TARGET_VIDEO_SIZE = (1280, 720)
AI_REQUEST_RATE = 5  # Hz - how often to request AI predictions

# Flight Configuration
TAKEOFF_ALTITUDE = 2.0  # meters
HOVER_TOLERANCE = 0.3   # meters
MAX_TAKEOFF_TIME = 30   # seconds

# Control Configuration - Safety Limits
MAX_LINEAR_VELOCITY = 2.0    # m/s
MAX_ANGULAR_VELOCITY = 1.0   # rad/s
MAX_ALTITUDE_VELOCITY = 1.0  # m/s
CONTROL_TIMEOUT = 10.0       # seconds

# Action Buffer Configuration
ACTION_BUFFER_SIZE = 10      # Number of fluid actions returned by AI
CONTROL_RATE = 10           # Hz - how often to execute buffered actions
ACTION_EXECUTION_RATE = 10   # Hz - execute one action every 0.1 seconds

# Safety Configuration
MIN_ALTITUDE = 1.0      # Minimum safe altitude (m)
MAX_ALTITUDE = 10.0     # Maximum safe altitude (m)
SAFETY_RADIUS = 50.0    # Maximum distance from takeoff point (m)

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

running = True
drone_armed = False
drone_ready = False
ai_control_active = False

# Thread locks
frame_lock = threading.Lock()
telemetry_lock = threading.Lock()
velocity_lock = threading.Lock()
action_buffer_lock = threading.Lock()

# Data storage
current_frame = None
current_telemetry = None
current_annotation = "Follow the target and maintain safe distance"
takeoff_position = None

# Control variables
current_vx = 0.0
current_vy = 0.0  
current_vz = 0.0
current_yaw_rate = 0.0

# Action buffer system - stores 10 fluid actions from AI
action_buffer = []
action_buffer_index = 0
last_ai_request_time = 0.0

# Queues
action_queue = queue.Queue(maxsize=5)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_observation(frame, telemetry, annotation):
    """Create observation dict with proper data types for AI model"""
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    position = np.array(telemetry['position'], dtype=np.float64)
    velocity = np.array(telemetry['velocity'], dtype=np.float64)
    orientation = np.array(telemetry['orientation'], dtype=np.float64)
    gimbal = np.array(telemetry['gimbal'], dtype=np.float64)
    
    observation = {
        "video.front_camera": frame[np.newaxis, ...],
        "state.position": position[np.newaxis, ...],
        "state.orientation": orientation[np.newaxis, ...],
        "state.velocity": velocity[np.newaxis, ...],
        "state.gimbal": gimbal[np.newaxis, ...],
        "annotation.human.task_description": [annotation]
    }
    
    return observation

def set_velocity(vx, vy, vz, yaw_rate=0.0):
    """Thread-safe velocity setting with safety limits"""
    global current_vx, current_vy, current_vz, current_yaw_rate
    
    # Apply safety limits
    vx = np.clip(vx, -MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY)
    vy = np.clip(vy, -MAX_LINEAR_VELOCITY, MAX_LINEAR_VELOCITY)
    vz = np.clip(vz, -MAX_ALTITUDE_VELOCITY, MAX_ALTITUDE_VELOCITY)
    yaw_rate = np.clip(yaw_rate, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
    
    with velocity_lock:
        current_vx = vx
        current_vy = vy
        current_vz = vz
        current_yaw_rate = yaw_rate

def get_velocity():
    """Thread-safe velocity reading"""
    with velocity_lock:
        return current_vx, current_vy, current_vz, current_yaw_rate

def add_actions_to_buffer(actions_dict):
    """Add AI actions to buffer for sequential execution"""
    global action_buffer, action_buffer_index
    
    with action_buffer_lock:
        # DEBUG: Print what fields are available in the AIq response
        print(f"🔍 AI Response Keys: {list(actions_dict.keys())}")
        
        # Extract all 10 actions and store them
        flight_control = np.array(actions_dict.get('action.flight_control', []))
        velocity_command = np.array(actions_dict.get('action.velocity_command', []))
        gimbal_command = np.array(actions_dict.get('action.gimbal', []))
        
        # DEBUG: Print extracted array shapes and some values
        print(f"🔍 Extracted arrays:")
        print(f"   flight_control shape: {flight_control.shape}, sample: {flight_control[:2] if len(flight_control) > 0 else 'empty'}")
        print(f"   velocity_command shape: {velocity_command.shape}, sample: {velocity_command[:2] if len(velocity_command) > 0 else 'empty'}")
        print(f"   gimbal_command shape: {gimbal_command.shape}, sample: {gimbal_command[:2] if len(gimbal_command) > 0 else 'empty'}")
        
        # Clear existing buffer
        action_buffer = []
        action_buffer_index = 0
        
        # Add all 10 actions to buffer
        num_actions = min(len(flight_control), len(velocity_command), len(gimbal_command))
        
        for i in range(num_actions):
            action = {
                'flight_control': flight_control[i] if len(flight_control) > i else [0, 0, 0, 0],
                'velocity_command': velocity_command[i] if len(velocity_command) > i else [0, 0, 0],
                'gimbal': gimbal_command[i] if len(gimbal_command) > i else [0, 0, 0],
                'timestep': i
            }
            action_buffer.append(action)
        
        print(f"🎬 Added {len(action_buffer)} fluid actions to buffer")
        return len(action_buffer)

def get_next_action_from_buffer():
    """Get the next action from buffer, returns None if buffer is empty"""
    global action_buffer, action_buffer_index
    
    with action_buffer_lock:
        if action_buffer_index < len(action_buffer):
            action = action_buffer[action_buffer_index]
            action_buffer_index += 1
            
            remaining = len(action_buffer) - action_buffer_index
            print(f"🎮 Executing action {action_buffer_index}/{len(action_buffer)} (remaining: {remaining})")
            
            return action
        else:
            # Buffer is empty
            return None

def is_action_buffer_empty():
    """Check if action buffer needs refilling"""
    with action_buffer_lock:
        return action_buffer_index >= len(action_buffer)

def safety_check(telemetry):
    """Check if current position is within safety limits"""
    if not telemetry or not takeoff_position:
        return False
    
    # Check altitude limits
    altitude = telemetry['position'][2]
    if altitude < MIN_ALTITUDE or altitude > MAX_ALTITUDE:
        print(f"⚠️  SAFETY: Altitude {altitude:.1f}m outside limits [{MIN_ALTITUDE}, {MAX_ALTITUDE}]")
        return False
    
    # Check distance from takeoff point
    dx = telemetry['position'][0] - takeoff_position[0]
    dy = telemetry['position'][1] - takeoff_position[1]
    distance = np.sqrt(dx*dx + dy*dy)
    
    if distance > SAFETY_RADIUS:
        print(f"⚠️  SAFETY: Distance {distance:.1f}m exceeds radius {SAFETY_RADIUS}m")
        return False
    
    return True

# =============================================================================
# DRONE CONTROLLER
# =============================================================================

class DroneController:
    def __init__(self, connection_string: str):
        self.master = None
        self.connected = False
        self.last_telemetry = None
        self.connect(connection_string)
    
    def connect(self, connection_string: str) -> bool:
        """Connect to drone via MAVLink"""
        try:
            print("🔗 Connecting to drone...")
            self.master = mavutil.mavlink_connection(connection_string)
            print("⏳ Waiting for heartbeat...")
            self.master.wait_heartbeat(timeout=10)
            print(f"✅ Connected to system {self.master.target_system}, component {self.master.target_component}")
            self.connected = True
            return True
        except Exception as e:
            print(f"❌ Failed to connect to drone: {e}")
            self.connected = False
            return False
    
    def get_telemetry(self) -> Optional[Dict[str, Any]]:
        """Get current drone telemetry"""
        if not self.connected:
            return self.last_telemetry
        
        try:
            pos_msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
            if pos_msg:
                att_msg = self.master.recv_match(type='ATTITUDE', blocking=False)
                
                telemetry = {
                    'position': [float(pos_msg.x), float(pos_msg.y), float(-pos_msg.z)],
                    'velocity': [float(pos_msg.vx), float(pos_msg.vy), float(-pos_msg.vz)],
                    'orientation': [0.0, 0.0, 0.0, 1.0],
                    'gimbal': [0.0, 0.0, 0.0],
                    'timestamp': time.time()
                }
                
                if att_msg:
                    roll, pitch, yaw = att_msg.roll, att_msg.pitch, att_msg.yaw
                    cr = np.cos(roll * 0.5)
                    sr = np.sin(roll * 0.5)
                    cp = np.cos(pitch * 0.5)
                    sp = np.sin(pitch * 0.5)
                    cy = np.cos(yaw * 0.5)
                    sy = np.sin(yaw * 0.5)
                    
                    w = cr * cp * cy + sr * sp * sy
                    x = sr * cp * cy - cr * sp * sy
                    y = cr * sp * cy + sr * cp * sy
                    z = cr * cp * sy - sr * sp * cy
                    
                    telemetry['orientation'] = [float(x), float(y), float(z), float(w)]
                
                self.last_telemetry = telemetry
                return telemetry
                
        except Exception as e:
            print(f"❌ Error getting telemetry: {e}")
        
        return self.last_telemetry
    
    def arm_and_offboard(self) -> bool:
        """ARM drone and switch to OFFBOARD mode"""
        global drone_armed, takeoff_position
        
        print("\n🚁 PREPARING FOR FLIGHT")
        
        # Get initial position
        telem = self.get_telemetry()
        if telem:
            takeoff_position = telem['position'].copy()
            print(f"📍 Takeoff position: X={takeoff_position[0]:.2f}, Y={takeoff_position[1]:.2f}, Z={takeoff_position[2]:.2f}")
        else:
            takeoff_position = [0.0, 0.0, 0.0]
            print("⚠️  Using default takeoff position")
        
        # Send initial setpoints
        print("📡 Sending initial setpoints...")
        for i in range(200):
            self.master.mav.set_position_target_local_ned_send(
                0, self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,  # Use position only
                takeoff_position[0], takeoff_position[1], -takeoff_position[2],
                0, 0, 0, 0, 0, 0, 0, 0
            )
            time.sleep(0.005)
        
        # ARM
        print("🔓 Arming drone...")
        for i in range(5):
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )
            time.sleep(0.2)
        
        # Switch to OFFBOARD
        print("🎮 Switching to OFFBOARD mode...")
        for i in range(5):
            self.master.set_mode_px4('OFFBOARD', 0, 0)
            time.sleep(0.2)
        
        drone_armed = True
        print("✅ Drone armed and ready for takeoff")
        return True
    
    def takeoff_to_altitude(self, target_altitude: float) -> bool:
        """Takeoff to specified altitude"""
        global drone_ready
        
        print(f"\n🚀 TAKING OFF TO {target_altitude}m")
        
        if not takeoff_position:
            print("❌ No takeoff position available")
            return False
        
        # Gradual takeoff
        altitudes = np.linspace(0.5, target_altitude, 8)
        
        for alt in altitudes:
            target_z = -alt  # NED frame (negative Z is up)
            print(f"⬆️  Climbing to {alt:.1f}m...")
            
            start_time = time.time()
            while time.time() - start_time < 5.0:  # 5 seconds per altitude step
                self.master.mav.set_position_target_local_ned_send(
                    0, self.master.target_system, self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    0b0000111111111000,  # Use position only
                    takeoff_position[0], takeoff_position[1], target_z,
                    0, 0, 0, 0, 0, 0, 0, 0
                )
                time.sleep(0.1)
            
            # Check altitude
            telem = self.get_telemetry()
            if telem:
                current_alt = telem['position'][2]
                error = abs(current_alt - alt)
                if error < HOVER_TOLERANCE:
                    print(f"✅ Reached {alt:.1f}m (actual: {current_alt:.2f}m)")
                else:
                    print(f"⚠️  At {current_alt:.2f}m (target: {alt:.1f}m)")
        
        # Final stabilization
        print(f"🎯 Stabilizing at {target_altitude}m...")
        start_time = time.time()
        stable_count = 0
        
        while time.time() - start_time < 10.0:  # 10 seconds max
            self.master.mav.set_position_target_local_ned_send(
                0, self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,
                takeoff_position[0], takeoff_position[1], -target_altitude,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            
            telem = self.get_telemetry()
            if telem:
                current_alt = telem['position'][2]
                if abs(current_alt - target_altitude) < HOVER_TOLERANCE:
                    stable_count += 1
                    if stable_count >= 30:  # 3 seconds stable (10Hz)
                        break
                else:
                    stable_count = 0
            
            time.sleep(0.1)
        
        drone_ready = True
        print(f"🎉 TAKEOFF COMPLETE - Drone ready at {target_altitude}m")
        
        return True
    
    def navigate_to_waypoints(self) -> bool:
        """Navigate using velocity commands before AI control"""
        global drone_ready
        
        print(f"\n🗺️  VELOCITY-BASED NAVIGATION")
        
        if not takeoff_position:
            print("❌ No takeoff position available")
            return False
        
        # Phase 1: Move forward 10 meters using velocity
        print(f"📍 Phase 1: Moving forward 10m using velocity control...")
        target_distance = 10.0
        velocity = 2.0  # 2 m/s forward velocity (faster for longer distance)
        move_time = target_distance / velocity  # 5 seconds

        print(f"   Setting forward velocity: {velocity} m/s for {move_time:.1f} seconds")
        start_time = time.time()
        
        while time.time() - start_time < move_time:
            # Send forward velocity command (X-axis positive = forward)
            self.master.mav.set_position_target_local_ned_send(
                0, self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000011111000111,  # Use velocity only
                0, 0, 0,  # Position (ignored)
                velocity, 0, 0,  # Velocity: forward, right, down
                0, 0, 0,  # Acceleration (ignored)
                0, 0  # Yaw, yaw_rate
            )
            
            # Show progress
            elapsed = time.time() - start_time
            distance_traveled = velocity * elapsed
            remaining = target_distance - distance_traveled
            
            telem = self.get_telemetry()
            if telem:
                current_pos = telem['position']
                actual_distance = current_pos[0] - takeoff_position[0]
                print(f"   Progress: {elapsed:.1f}s, Target: {distance_traveled:.1f}m, Actual: {actual_distance:.1f}m")
            
            time.sleep(0.1)  # 10Hz
        
        # Stop forward movement
        print("   Stopping forward movement...")
        for _ in range(10):  # 1 second of stop commands
            self.master.mav.set_position_target_local_ned_send(
                0, self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000011111000111,  # Use velocity only
                0, 0, 0,  # Position (ignored)
                0, 0, 0,  # Velocity: stop
                0, 0, 0,  # Acceleration (ignored)
                0, 0  # Yaw, yaw_rate
            )
            time.sleep(0.1)
        
        print("✅ Forward movement complete")
        
        # Phase 2: Move right 1 meter using velocity
        print(f"📍 Phase 2: Moving right 1m using velocity control...")
        target_distance = 2.0
        velocity = 1.0  # 1 m/s right velocity (slower for precision)
        move_time = target_distance / velocity  # 1 seconds

        print(f"   Setting right velocity: {velocity} m/s for {move_time:.1f} seconds")
        start_time = time.time()
        
        while time.time() - start_time < move_time:
            # Send right velocity command (Y-axis positive = right)
            self.master.mav.set_position_target_local_ned_send(
                0, self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000011111000111,  # Use velocity only
                0, 0, 0,  # Position (ignored)
                0, velocity, 0,  # Velocity: forward, right, down
                0, 0, 0,  # Acceleration (ignored)
                0, 0  # Yaw, yaw_rate
            )
            
            # Show progress
            elapsed = time.time() - start_time
            distance_traveled = velocity * elapsed
            remaining = target_distance - distance_traveled
            
            telem = self.get_telemetry()
            if telem:
                current_pos = telem['position']
                actual_distance = current_pos[1] - takeoff_position[1]
                print(f"   Progress: {elapsed:.1f}s, Target: {distance_traveled:.1f}m, Actual: {actual_distance:.1f}m")
            
            time.sleep(0.1)  # 10Hz
        
        # Stop right movement
        print("   Stopping right movement...")
        for _ in range(10):  # 1 second of stop commands
            self.master.mav.set_position_target_local_ned_send(
                0, self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000011111000111,  # Use velocity only
                0, 0, 0,  # Position (ignored)
                0, 0, 0,  # Velocity: stop
                0, 0, 0,  # Acceleration (ignored)
                0, 0  # Yaw, yaw_rate
            )
            time.sleep(0.1)
        
        print("✅ Right movement complete")
        
        # Final position check
        telem = self.get_telemetry()
        if telem:
            final_pos = telem['position']
            dx = final_pos[0] - takeoff_position[0]
            dy = final_pos[1] - takeoff_position[1]
            print(f"🎯 Final position relative to takeoff: X={dx:.2f}m, Y={dy:.2f}m")
            print(f"   Target was: X=10.0m, Y=1.0m")
            print(f"   Error: X={abs(dx-10.0):.2f}m, Y={abs(dy-1.0):.2f}m")
        
        print("🎯 Velocity-based navigation complete!")
        print("🤖 AI control will start in 3 seconds...")
        time.sleep(3)
        
        return True
    
    def land_and_disarm(self):
        """Emergency landing and disarm"""
        global running, drone_armed, drone_ready, ai_control_active
        
        print("\n🛬 INITIATING LANDING SEQUENCE")
        
        # Stop AI control
        ai_control_active = False
        drone_ready = False
        
        # Stop movement
        set_velocity(0.0, 0.0, 0.0, 0.0)
        time.sleep(1)
        
        # Land
        print("📉 Landing...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        
        time.sleep(10)  # Wait for landing
        
        # Disarm
        print("🔒 Disarming...")
        for i in range(3):
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            time.sleep(0.2)
        
        drone_armed = False
        print("✅ Landing complete")

# =============================================================================
# VIDEO CAPTURE
# =============================================================================

class VideoCapture:
    def __init__(self, video_url: str):
        self.video_url = video_url
        self.cap = None
        self.connected = False
        self.connect()
    
    def connect(self) -> bool:
        """Connect to video stream"""
        try:
            print(f"📹 Connecting to video stream...")
            self.cap = cv2.VideoCapture(self.video_url)
            
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    print(f"✅ Video connected - Frame size: {frame.shape}")
                    self.connected = True
                    return True
            
            print("❌ Failed to capture test frame")
            return False
            
        except Exception as e:
            print(f"❌ Video connection error: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current video frame"""
        if not self.connected or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                if frame.shape[:2][::-1] != TARGET_VIDEO_SIZE:
                    frame = cv2.resize(frame, TARGET_VIDEO_SIZE)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb.astype(np.uint8)
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
        
        return None
    
    def release(self):
        if self.cap:
            self.cap.release()

# =============================================================================
# AI MODEL CLIENT
# =============================================================================

class AIModelClient:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session = requests.Session()
        self.timeout = CONTROL_TIMEOUT
        self.request_count = 0
        self.success_count = 0
    
    def predict_action(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send observation to AI model and get predicted action"""
        self.request_count += 1
        
        try:
            payload = {"observation": observation}
            response = self.session.post(self.api_url, json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                action = response.json()
                self.success_count += 1
                print(f"🤖 AI Request #{self.request_count} - Success rate: {self.success_count/self.request_count:.1%}")
                
                # DEBUG: Print the exact AI response structure
                print(f"🔍 AI Response Debug:")
                print(f"   Type: {type(action)}")
                print(f"   Keys: {list(action.keys()) if isinstance(action, dict) else 'Not a dict'}")
                print(f"   Full response: {action}")
                
                return action
            else:
                print(f"❌ AI Request #{self.request_count} failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ AI Request #{self.request_count} timeout")
            return None
        except Exception as e:
            print(f"❌ AI Request #{self.request_count} error: {e}")
            return None

# =============================================================================
# MAIN THREADS
# =============================================================================

def telemetry_thread(drone_controller: DroneController):
    """Continuously collect telemetry data"""
    global current_telemetry, running
    
    while running:
        try:
            telemetry = drone_controller.get_telemetry()
            if telemetry:
                with telemetry_lock:
                    current_telemetry = telemetry
            time.sleep(0.05)  # 20Hz
        except Exception as e:
            print(f"❌ Telemetry error: {e}")
            time.sleep(0.1)

def video_thread(video_capture: VideoCapture):
    """Continuously capture video frames"""
    global current_frame, running
    
    while running:
        try:
            frame = video_capture.get_frame()
            if frame is not None:
                with frame_lock:
                    current_frame = frame
            time.sleep(0.033)  # ~30Hz
        except Exception as e:
            print(f"❌ Video error: {e}")
            time.sleep(0.1)

def ai_prediction_thread(ai_client: AIModelClient):
    """Get AI predictions and manage action buffer"""
    global running, current_frame, current_telemetry, current_annotation, ai_control_active
    global last_ai_request_time
    
    while running:
        try:
            if not ai_control_active:
                time.sleep(0.1)
                continue
            
            # Check if we need to request new actions from AI
            current_time = time.time()
            need_new_actions = (
                is_action_buffer_empty() or 
                (current_time - last_ai_request_time) > (ACTION_BUFFER_SIZE / ACTION_EXECUTION_RATE + 1.0)
            )
            
            if need_new_actions:
                # Get current data for AI request
                with frame_lock:
                    frame = current_frame.copy() if current_frame is not None else None
                
                with telemetry_lock:
                    telemetry = current_telemetry.copy() if current_telemetry is not None else None
                
                if frame is None or telemetry is None:
                    time.sleep(0.1)
                    continue
                
                # Safety check
                if not safety_check(telemetry):
                    print("🚨 SAFETY VIOLATION - Stopping AI control")
                    set_velocity(0.0, 0.0, 0.0, 0.0)
                    time.sleep(1.0)
                    continue
                
                # Create observation
                observation = create_observation(frame, telemetry, current_annotation)
                
                # Get AI prediction (10 fluid actions)
                print(f"🤖 Requesting new 10 fluid actions from AI...")
                action = ai_client.predict_action(observation)
                
                if action:
                    # Add all 10 actions to buffer
                    num_added = add_actions_to_buffer(action)
                    last_ai_request_time = current_time
                    print(f"✅ Added {num_added} actions to buffer")
                else:
                    print("❌ Failed to get actions from AI")
            
            # Wait before checking again (slower rate for AI requests)
            time.sleep(0.5)  # Check every 0.5 seconds
            
        except Exception as e:
            print(f"❌ AI prediction thread error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)

def action_execution_thread():
    """Execute actions from buffer at regular intervals"""
    global running, ai_control_active
    
    while running:
        try:
            if not ai_control_active:
                time.sleep(0.1)
                continue
            
            # Debug: Check buffer status
            buffer_empty = is_action_buffer_empty()
            with action_buffer_lock:
                buffer_size = len(action_buffer)
                buffer_index = action_buffer_index
            
            if buffer_size > 0:
                print(f"🔍 Buffer status: {buffer_size} actions, index={buffer_index}, empty={buffer_empty}")
            
            # Get next action from buffer
            current_action = get_next_action_from_buffer()
            
            if current_action:
                print(f"✅ Got action from buffer: timestep {current_action.get('timestep', 'unknown')}")
                # Get current telemetry for safety checks
                with telemetry_lock:
                    telemetry = current_telemetry.copy() if current_telemetry is not None else None
                
                if telemetry and safety_check(telemetry):
                    # Execute this specific action
                    execute_single_action(current_action, telemetry)
                else:
                    print("⚠️  Skipping action due to safety check failure")
                    set_velocity(0.0, 0.0, 0.0, 0.0)
            else:
                # No actions in buffer, hover
                if buffer_size == 0:
                    print("⏸️  No actions in buffer - hovering")
                set_velocity(0.0, 0.0, 0.0, 0.0)
            
            time.sleep(1.0 / ACTION_EXECUTION_RATE)  # 10Hz execution rate
            
        except Exception as e:
            print(f"❌ Action execution thread error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

def execute_single_action(action: Dict[str, Any], telemetry: Dict[str, Any]):
    """Execute a single fluid action from the buffer"""
    try:
        # Extract action components
        flight_control = action.get('flight_control', [0, 0, 0, 0])
        velocity_command = action.get('velocity_command', [0, 0, 0])
        gimbal = action.get('gimbal', [0, 0, 0])
        timestep = action.get('timestep', 0)
        
        # Convert to proper types
        flight_control = np.array(flight_control, dtype=np.float64)
        velocity_command = np.array(velocity_command, dtype=np.float64)
        gimbal = np.array(gimbal, dtype=np.float64)
        
        # Extract velocity commands (primary control)
        vx = vy = vz = yaw_rate = 0.0
        
        # Use velocity_command as primary control
        if len(velocity_command) >= 3:
            vx, vy, vz = velocity_command[:3]
            print(f"🔍 velocity_command extracted: vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f}")
        
        # Use flight_control for yaw if available (4th component)
        if len(flight_control) >= 4:
            yaw_rate = flight_control[3]  # Yaw rate
            print(f"🔍 flight_control[3] yaw_rate: {yaw_rate:.4f}")
        
        # Debug: Print raw action data
        print(f"🔍 Raw action data:")
        print(f"   flight_control: {flight_control}")
        print(f"   velocity_command: {velocity_command}")
        print(f"   gimbal: {gimbal}")
        
        # Apply safety constraints
        current_altitude = telemetry['position'][2]
        
        # Altitude safety
        if current_altitude <= MIN_ALTITUDE and vz < 0:
            vz = max(vz, -0.1)
        elif current_altitude >= MAX_ALTITUDE and vz > 0:
            vz = min(vz, 0.1)
        
        # Set velocity with safety limits
        set_velocity(vx, vy, vz, yaw_rate)
        
        print(f"🎮 Action {timestep+1}: vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f}, yaw={yaw_rate:.4f}")
        
    except Exception as e:
        print(f"❌ Single action execution error: {e}")
        import traceback
        traceback.print_exc()
        set_velocity(0.0, 0.0, 0.0, 0.0)

def execute_ai_action(action: Dict[str, Any], telemetry: Dict[str, Any]):
    """Convert AI action to drone velocity commands"""
    try:
        # DEBUG: Print detailed action analysis
        print(f"\n🔍 DEBUGGING AI ACTION:")
        print(f"   Action type: {type(action)}")
        print(f"   Action keys: {list(action.keys()) if isinstance(action, dict) else 'Not a dict'}")
        
        # Extract velocity commands from AI action
        vx = vy = vz = yaw_rate = 0.0
        extraction_method = "none"
        
        if isinstance(action, dict):
            print(f"   Dict keys: {list(action.keys())}")
            
            # Method 1: AI Model specific format - velocity_command
            if 'action.velocity_command' in action:
                velocity_command = action['action.velocity_command']
                print(f"   Method 1 - action.velocity_command shape: {np.array(velocity_command).shape}")
                
                # Take the first row or average of all rows
                vel_array = np.array(velocity_command)
                if len(vel_array.shape) == 2 and vel_array.shape[0] > 0:
                    # Take the first command or average all commands
                    vel_cmd = vel_array[0]  # Use first command
                    # vel_cmd = np.mean(vel_array, axis=0)  # Alternative: use average
                    
                    if len(vel_cmd) >= 3:
                        vx, vy, vz = vel_cmd[:3]
                        extraction_method = "action.velocity_command"
                        print(f"   ✅ Extracted from action.velocity_command: vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}")
            
            # Method 2: AI Model specific format - flight_control
            elif 'action.flight_control' in action and extraction_method == "none":
                flight_control = action['action.flight_control']
                print(f"   Method 2 - action.flight_control shape: {np.array(flight_control).shape}")
                
                fc_array = np.array(flight_control)
                if len(fc_array.shape) == 2 and fc_array.shape[0] > 0:
                    # Take the first command
                    fc_cmd = fc_array[0]
                    
                    if len(fc_cmd) >= 4:
                        vx, vy, vz, yaw_rate = fc_cmd[:4]
                        extraction_method = "action.flight_control"
                        print(f"   ✅ Extracted from action.flight_control: vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}, yaw={yaw_rate:.3f}")
            
            # Method 3: Standard format - Direct velocity commands
            elif 'linear_velocity' in action and 'angular_velocity' in action:
                linear_vel = action['linear_velocity']
                angular_vel = action['angular_velocity']
                
                print(f"   Method 3 - linear_velocity: {linear_vel}, angular_velocity: {angular_vel}")
                
                if len(linear_vel) >= 3:
                    vx, vy, vz = linear_vel[:3]
                if len(angular_vel) >= 3:
                    yaw_rate = angular_vel[2]  # Yaw rate around Z axis
                extraction_method = "linear_angular_velocity"
            
            # Method 4: Individual velocity components
            elif 'velocity_x' in action:
                vx = action.get('velocity_x', 0.0)
                vy = action.get('velocity_y', 0.0)
                vz = action.get('velocity_z', 0.0)
                yaw_rate = action.get('yaw_rate', 0.0)
                
                print(f"   Method 4 - individual components: vx={vx}, vy={vy}, vz={vz}, yaw_rate={yaw_rate}")
                extraction_method = "individual_components"
            
            # Method 5: Action array
            elif 'action' in action:
                action_array = np.array(action['action']).flatten()
                print(f"   Method 5 - action array: {action_array}")
                
                if len(action_array) >= 4:
                    vx, vy, vz, yaw_rate = action_array[:4]
                    extraction_method = "action_array"
                else:
                    print(f"   ⚠️  Action array too short: {len(action_array)} elements")
            
            # Method 6: Check for other numerical arrays
            else:
                print(f"   🔍 Checking other formats...")
                for key, value in action.items():
                    if isinstance(value, (list, np.ndarray)):
                        try:
                            arr = np.array(value)
                            print(f"     Key '{key}': shape={arr.shape}, dtype={arr.dtype}")
                            
                            # Try to extract from any numerical array with right dimensions
                            if len(arr.shape) >= 1:
                                arr_flat = arr.flatten()
                                if len(arr_flat) >= 4 and extraction_method == "none":
                                    vx, vy, vz, yaw_rate = arr_flat[:4]
                                    extraction_method = f"from_key_{key}"
                                    print(f"   ✅ Extracted from key '{key}': vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}, yaw_rate={yaw_rate:.3f}")
                                    break
                        except Exception as e:
                            print(f"     ❌ Failed to extract from key '{key}': {e}")
        
        elif isinstance(action, (list, np.ndarray)):
            # Direct array format [vx, vy, vz, yaw_rate]
            action_array = np.array(action).flatten()
            print(f"   Method 7 - direct array: {action_array}")
            
            if len(action_array) >= 4:
                vx, vy, vz, yaw_rate = action_array[:4]
                extraction_method = "direct_array"
            else:
                print(f"   ⚠️  Direct array too short: {len(action_array)} elements")
        
        print(f"   📊 EXTRACTION RESULT:")
        print(f"     Method used: {extraction_method}")
        print(f"     Raw values: vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f}, yaw_rate={yaw_rate:.4f}")
        
        # Apply safety constraints based on current position
        current_altitude = telemetry['position'][2]
        
        # Limit altitude changes
        if current_altitude <= MIN_ALTITUDE and vz < 0:
            vz = max(vz, -0.1)  # Slow descent near minimum
            print(f"   🛡️  Altitude safety: Limited vz to {vz} (near min altitude)")
        elif current_altitude >= MAX_ALTITUDE and vz > 0:
            vz = min(vz, 0.1)   # Slow ascent near maximum
            print(f"   🛡️  Altitude safety: Limited vz to {vz} (near max altitude)")
        
        # Set velocity with safety limits
        set_velocity(vx, vy, vz, yaw_rate)
        
        print(f"🎮 AI Control Applied: vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f}, yaw={yaw_rate:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Action execution error: {e}")
        import traceback
        traceback.print_exc()
        set_velocity(0.0, 0.0, 0.0, 0.0)  # Safe fallback

def control_thread(drone_controller: DroneController):
    """Send control commands to drone"""
    global running, drone_ready, ai_control_active
    
    while running:
        try:
            if drone_ready:
                vx, vy, vz, yaw_rate = get_velocity()
                
                # Debug: Print velocity commands being sent to drone
                if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vz) > 0.01 or abs(yaw_rate) > 0.01:
                    print(f"🚁 Sending to drone: vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f}, yaw_rate={yaw_rate:.4f}")
                
                # Send velocity commands using LOCAL_NED frame for world-relative control
                drone_controller.master.mav.set_position_target_local_ned_send(
                    0,  # time_boot_ms
                    drone_controller.master.target_system,
                    drone_controller.master.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # Use world frame instead of body frame
                    0b0000011111000111,  # Use velocity and yaw rate
                    0, 0, 0,  # Position (ignored)
                    vx, vy, vz,  # Velocity in world frame
                    0, 0, 0,  # Acceleration (ignored)
                    0, yaw_rate  # Yaw, yaw_rate
                )
            
            time.sleep(0.1)  # 10Hz control loop
            
        except Exception as e:
            print(f"❌ Control error: {e}")
            time.sleep(0.1)

def display_thread(video_capture: VideoCapture):
    """Display video feed with status information"""
    global running, current_annotation, ai_control_active
    
    cv2.namedWindow("AI Drone Control System", cv2.WINDOW_AUTOSIZE)
    
    while running:
        try:
            # Get display frame
            with frame_lock:
                if current_frame is not None:
                    display_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
                else:
                    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(display_frame, "Waiting for video...", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add status overlay
            y_pos = 30
            
            # System status
            if not drone_armed:
                status_color = (0, 255, 255)  # Yellow
                status_text = "INITIALIZING"
            elif not drone_ready:
                status_color = (0, 165, 255)  # Orange
                status_text = "TAKING OFF"
            elif ai_control_active:
                status_color = (0, 255, 0)    # Green
                status_text = "AI CONTROL ACTIVE"
            else:
                status_color = (255, 255, 0)  # Cyan
                status_text = "READY FOR AI"
            
            cv2.putText(display_frame, f"Status: {status_text}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            y_pos += 30
            
            # Telemetry info
            with telemetry_lock:
                if current_telemetry is not None:
                    pos = current_telemetry['position']
                    vel = current_telemetry['velocity']
                    
                    cv2.putText(display_frame, f"Pos: X={pos[0]:.1f} Y={pos[1]:.1f} Z={pos[2]:.1f}m",
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 25
                    
                    cv2.putText(display_frame, f"Vel: X={vel[0]:.1f} Y={vel[1]:.1f} Z={vel[2]:.1f}m/s",
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 25
            
            # Control info
            vx, vy, vz, yaw_rate = get_velocity()
            cv2.putText(display_frame, f"Control: vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} yaw={yaw_rate:.2f}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
            
            # Task description
            cv2.putText(display_frame, f"Task: {current_annotation[:40]}",
                       (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Instructions
            instructions = [
                "SPACE: Start/Stop AI Control",
                "ESC/Q: Emergency Land & Exit"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(display_frame, instruction, (10, display_frame.shape[0] - 30 + i * 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("AI Drone Control System", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
                print("\n🚨 EMERGENCY STOP REQUESTED")
                running = False
                break
            elif key == ord(' '):  # SPACE - Toggle AI control
                if drone_ready:
                    ai_control_active = not ai_control_active
                    if ai_control_active:
                        print("🤖 AI Control ACTIVATED")
                    else:
                        print("⏸️  AI Control PAUSED")
                        set_velocity(0.0, 0.0, 0.0, 0.0)  # Stop movement
                else:
                    print("⚠️  Drone not ready for AI control")
            
        except Exception as e:
            print(f"❌ Display error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main AI drone control function"""
    global running, current_annotation
    
    print("=" * 60)
    print("🤖 AI-CONTROLLED DRONE SYSTEM")
    print("=" * 60)
    print(f"🔗 AI Model API: {AI_MODEL_API_URL}")
    print(f"📹 Video Feed: {AIRSIM_VIDEO_URL}")
    print(f"📡 MAVLink: {MAVLINK_CONNECTION}")
    print(f"🎯 Takeoff Altitude: {TAKEOFF_ALTITUDE}m")
    print("=" * 60)
    
    # Get user task description
    try:
        user_input = input("\n📝 Enter task description (or press Enter for default): ").strip()
        if user_input:
            current_annotation = user_input
        print(f"📋 Task: {current_annotation}")
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        return
    
    print("\n🔧 Initializing components...")
    
    # Initialize components
    drone_controller = DroneController(MAVLINK_CONNECTION)
    if not drone_controller.connected:
        print("❌ Failed to connect to drone. Exiting...")
        return
    
    video_capture = VideoCapture(AIRSIM_VIDEO_URL)
    if not video_capture.connected:
        print("❌ Failed to connect to video stream. Exiting...")
        return
    
    ai_client = AIModelClient(AI_MODEL_API_URL)
    
    print("✅ All components initialized!")
    print("\n🚀 Starting flight sequence...")
    
    try:
        # Start all threads
        threads = [
            threading.Thread(target=telemetry_thread, args=(drone_controller,), daemon=True),
            threading.Thread(target=video_thread, args=(video_capture,), daemon=True),
            threading.Thread(target=ai_prediction_thread, args=(ai_client,), daemon=True),
            threading.Thread(target=action_execution_thread, daemon=True),
            threading.Thread(target=control_thread, args=(drone_controller,), daemon=True),
            threading.Thread(target=display_thread, args=(video_capture,), daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        # Flight sequence
        print("🔄 Phase 1: ARM and OFFBOARD")
        if not drone_controller.arm_and_offboard():
            print("❌ Failed to arm drone")
            return
        
        print("🔄 Phase 2: TAKEOFF")
        if not drone_controller.takeoff_to_altitude(TAKEOFF_ALTITUDE):
            print("❌ Failed to takeoff")
            return
        
        print("🔄 Phase 3: VELOCITY NAVIGATION")
        if not drone_controller.navigate_to_waypoints():
            print("❌ Failed velocity navigation")
            return
        
        print("🔄 Phase 4: AI CONTROL READY")
        print("💡 Press SPACE in the video window to start AI control")
        print("💡 Press ESC or Q to emergency land and exit")
        
        # Wait for threads to complete
        for thread in threads:
            thread.join()
    
    except KeyboardInterrupt:
        print("\n🚨 Keyboard interrupt received...")
    
    finally:
        print("\n🔄 Shutting down system...")
        running = False
        
        # Emergency landing
        if drone_controller.connected and drone_armed:
            drone_controller.land_and_disarm()
        
        # Cleanup
        video_capture.release()
        
        print("✅ Shutdown complete")
        print("=" * 60)

if __name__ == "__main__":
    main()
