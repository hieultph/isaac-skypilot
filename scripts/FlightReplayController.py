# Enhanced Drone Flight Replay from Sensor Data
# This script reads recorded sensor data and replays drone movement using velocity and orientation

import airsim
import pandas as pd
import time
import math
import numpy as np
import signal
import sys
from pathlib import Path
from pymavlink import mavutil
import threading

class FlightReplayController:
    def __init__(self):
        self.client = None
        self.mavlink_connection = None
        self.vehicle_name = "PX4"
        self.replay_speed = 1.0
        self.start_altitude = 5.0  # meters above ground
        self.control_frequency = 50.0  # Hz - how often to send commands (50-100 Hz for smooth flight)
        self.interrupted = False
        self.interrupt_count = 0
        self.mavlink_port = "udpin:0.0.0.0:14555"  # Default PX4 SITL port
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C interrupt gracefully."""
        self.interrupt_count += 1
        if self.interrupt_count == 1:
            print("\n⚠ Interrupt received, will stop after current operation completes...")
            self.interrupted = True
        elif self.interrupt_count == 2:
            print("\n⚠ Second interrupt - forcing immediate stop...")
            self.interrupted = True
        else:
            print("\n🛑 Multiple interrupts - emergency exit!")
            sys.exit(1)
        
    def connect(self):
        """Connect to AirSim and MAVLink."""
        try:
            # Connect to AirSim
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("✓ Connected to AirSim successfully.")
            
            # Connect to MAVLink
            try:
                print(f"🔗 Connecting to MAVLink at {self.mavlink_port}...")
                self.mavlink_connection = mavutil.mavlink_connection(self.mavlink_port)
                self.mavlink_connection.wait_heartbeat()
                print("✓ Connected to MAVLink successfully.")
                print(f"   • System ID: {self.mavlink_connection.target_system}")
                print(f"   • Component ID: {self.mavlink_connection.target_component}")
                
                # Check status
                self.check_mavlink_status()
                return True
                
            except Exception as mav_e:
                print(f"⚠ MAVLink connection failed: {mav_e}")
                print("   • Will use AirSim API only for takeoff")
                self.mavlink_connection = None
                return True
                
        except Exception as e:
            print(f"✗ AirSim connection failed: {e}")
            return False
    
    def load_data(self, file_path):
        """Load and validate sensor data from Excel file."""
        try:
            # Read Excel file, skipping header rows
            df = pd.read_excel(file_path, skiprows=4)  # Fixed: was 5, should be 4
            
            print(f"✓ Loaded {len(df)} data points from {file_path}")
            
            # Clean data - remove rows without frame numbers
            df = df.dropna(subset=['Frame'])
            df = df.reset_index(drop=True)
            
            # Check for required columns
            required_velocity = ['Video Time (s)', 'X (m/s)', 'Y (m/s)', 'Z (m/s)']
            drone_orientation = ['Drone Roll (deg)', 'Drone Pitch (deg)', 'Drone Yaw (deg)']
            
            missing_vel = [col for col in required_velocity if col not in df.columns]
            if missing_vel:
                print(f"✗ Missing velocity columns: {missing_vel}")
                print(f"Available columns: {list(df.columns)}")
                return None, False
                
            # Check if we have drone orientation data (second set of Roll/Pitch/Yaw)
            has_orientation = all(col in df.columns for col in drone_orientation)
            if not has_orientation:
                print("⚠ Warning: No drone orientation data found. Will use velocity only.")
                print(f"Looking for: {drone_orientation}")
                print(f"Available: {[col for col in df.columns if 'deg' in str(col)]}")
            else:
                print("✓ Found drone orientation data for replay.")
            
            return df, has_orientation
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None, False
    
    def check_mavlink_status(self):
        """Check MAVLink connection status and vehicle state."""
        if not self.mavlink_connection:
            return False
            
        try:
            print("📡 Checking MAVLink status...")
            
            # Get heartbeat
            msg = self.mavlink_connection.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
            if msg:
                armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                custom_mode = msg.custom_mode
                print(f"   • Vehicle Type: {msg.type}")
                print(f"   • Armed: {'Yes' if armed else 'No'}")
                print(f"   • Flight Mode: {custom_mode}")
                print(f"   • System Status: {msg.system_status}")
                return True
            else:
                print("   • No heartbeat received")
                return False
                
        except Exception as e:
            print(f"   • Status check error: {e}")
            return False
    
    def mavlink_arm_and_takeoff(self, target_altitude):
        """Arm and takeoff using MAVLink commands."""
        if not self.mavlink_connection:
            print("⚠ No MAVLink connection available")
            return False
            
        try:
            print("🚁 Using MAVLink for arm and takeoff...")
            
            # Wait for a valid heartbeat
            print("   • Waiting for heartbeat...")
            self.mavlink_connection.wait_heartbeat()
            print(f"   • Heartbeat received from system {self.mavlink_connection.target_system}")
            
            # Check current mode
            print("   • Checking current flight mode...")
            
            # Set mode to GUIDED
            print("   • Setting GUIDED mode...")
            mode = 'GUIDED'
            try:
                mode_id = self.mavlink_connection.mode_mapping()[mode]
                self.mavlink_connection.mav.set_mode_send(
                    self.mavlink_connection.target_system,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_id
                )
                print(f"   • GUIDED mode command sent (mode_id: {mode_id})")
            except Exception as mode_e:
                print(f"   • Mode setting error: {mode_e}, trying alternative...")
                # Try manual mode setting
                self.mavlink_connection.mav.command_long_send(
                    self.mavlink_connection.target_system,
                    self.mavlink_connection.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    0,  # confirmation
                    1,  # base mode
                    4,  # custom mode (GUIDED)
                    0, 0, 0, 0, 0
                )
            
            # Wait for mode change
            print("   • Waiting for mode change...")
            time.sleep(3)
            
            # Arm the vehicle
            print("   • Arming vehicle...")
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                1,  # arm (1 = arm, 0 = disarm)
                0, 0, 0, 0, 0, 0  # unused parameters
            )
            
            # Wait for arming confirmation
            print("   • Waiting for arm confirmation...")
            arm_timeout = 10
            arm_start = time.time()
            armed = False
            
            while time.time() - arm_start < arm_timeout:
                msg = self.mavlink_connection.recv_match(type='HEARTBEAT', blocking=False)
                if msg:
                    armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                    if armed:
                        print("   • Vehicle armed successfully!")
                        break
                time.sleep(0.5)
            
            if not armed:
                print("   • Warning: Arm confirmation not received, continuing anyway...")
            
            # Send takeoff command
            print(f"   • Sending takeoff command for {target_altitude}m...")
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,  # confirmation
                0,  # minimum pitch (if airframe is angle-limited)
                0,  # empty
                0,  # empty
                0,  # yaw angle (if magnetometer present)
                0,  # latitude (if global position)
                0,  # longitude (if global position)
                target_altitude  # altitude
            )
            
            print("   • Takeoff command sent, monitoring altitude...")
            
            # Monitor takeoff progress
            start_time = time.time()
            timeout = 60  # 60 seconds timeout for takeoff (more time)
            last_alt = 0
            progress_stall_count = 0
            
            while time.time() - start_time < timeout:
                try:
                    # Check for interrupt but don't stop during critical takeoff phase
                    if self.interrupted and (time.time() - start_time) > 10:  # Allow 10 seconds minimum
                        print("   • Takeoff interrupted by user")
                        break
                    
                    # Check altitude using AirSim
                    current_pose = self.client.simGetVehiclePose(self.vehicle_name)
                    current_alt = abs(current_pose.position.z_val)
                    
                    # Check altitude progress
                    if current_alt > last_alt + 0.2:  # Altitude is increasing (more threshold)
                        last_alt = current_alt
                        progress_stall_count = 0
                        print(f"   • Takeoff progress: {current_alt:.1f}m / {target_altitude}m")
                    else:
                        progress_stall_count += 1
                    
                    if current_alt >= target_altitude * 0.85:  # 85% of target altitude
                        print(f"✓ MAVLink takeoff completed at {current_alt:.1f}m")
                        return True
                    
                    # If altitude stalled for too long, send another takeoff command
                    if progress_stall_count > 10 and current_alt < target_altitude * 0.5:
                        print("   • Sending additional takeoff command...")
                        self.mavlink_connection.mav.command_long_send(
                            self.mavlink_connection.target_system,
                            self.mavlink_connection.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                            0, 0, 0, 0, 0, 0, 0, target_altitude
                        )
                        progress_stall_count = 0
                    
                    # Also check for MAVLink altitude messages
                    alt_msg = self.mavlink_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
                    if alt_msg:
                        rel_alt = alt_msg.relative_alt / 1000.0  # Convert mm to m
                        if rel_alt >= target_altitude * 0.85:
                            print(f"✓ MAVLink takeoff completed (MAVLink alt: {rel_alt:.1f}m)")
                            return True
                    
                    # Try using position control if takeoff stalls
                    if progress_stall_count > 20 and current_alt > 0.5:
                        print("   • Switching to position control for altitude...")
                        # Send position setpoint to reach target altitude
                        self.mavlink_connection.mav.set_position_target_local_ned_send(
                            0,  # time_boot_ms
                            self.mavlink_connection.target_system,
                            self.mavlink_connection.target_component,
                            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                            0b0000111111111000,  # ignore velocity and acceleration, control position
                            0, 0, -target_altitude,  # x, y, z (NED)
                            0, 0, 0,  # vx, vy, vz
                            0, 0, 0,  # afx, afy, afz
                            0, 0  # yaw, yaw_rate
                        )
                        progress_stall_count = 0
                        
                    time.sleep(1.0)  # Check every second
                    
                except Exception as monitor_e:
                    print(f"   • Monitoring error: {monitor_e}")
                    time.sleep(1.0)
                    continue
            
            print("⚠ MAVLink takeoff timeout, checking final altitude...")
            try:
                final_pose = self.client.simGetVehiclePose(self.vehicle_name)
                final_alt = abs(final_pose.position.z_val)
                if final_alt > 1.0:  # At least 1 meter off ground
                    print(f"✓ Takeoff appears successful at {final_alt:.1f}m")
                    return True
                else:
                    print(f"⚠ Takeoff may have failed, altitude: {final_alt:.1f}m")
                    return False
            except:
                print("⚠ Could not verify takeoff altitude")
                return False
            
        except Exception as e:
            print(f"⚠ MAVLink takeoff error: {e}")
            print("   • Falling back to AirSim takeoff")
            return False
    
    def prepare_drone(self):
        """Prepare drone for flight replay."""
        try:
            print("🚁 Preparing drone for replay...")
            
            # Enable API control
            self.client.enableApiControl(True, self.vehicle_name)
            print("✓ API control enabled")
            
            # Try MAVLink takeoff first
            mavlink_success = False
            if self.mavlink_connection:
                mavlink_success = self.mavlink_arm_and_takeoff(self.start_altitude)
            
            if not mavlink_success:
                print("🛫 Using AirSim takeoff as fallback...")
                
                # Set home position to current position (fixes GPS issue)
                try:
                    current_pose = self.client.simGetVehiclePose(self.vehicle_name)
                    # Get current GPS position or use default
                    current_gps = self.client.simGetGroundTruthKinematics(self.vehicle_name)
                    home_geo = airsim.GeoPoint()
                    home_geo.latitude = 47.641468  # Default Seattle latitude
                    home_geo.longitude = -122.140165  # Default Seattle longitude  
                    home_geo.altitude = 122.0  # Default altitude
                    self.client.simSetHomeGeoPoint(home_geo, self.vehicle_name)
                    print("✓ GPS home location set")
                except Exception as gps_e:
                    print(f"⚠ GPS home warning: {gps_e} (continuing anyway)")
                
                # Arm the drone
                try:
                    # Try to set a simple home position first
                    try:
                        current_pose = self.client.simGetVehiclePose(self.vehicle_name)
                        # Set home at current position with minimal GPS coordinates
                        home_geo = airsim.GeoPoint()
                        home_geo.latitude = 47.641468
                        home_geo.longitude = -122.140165
                        home_geo.altitude = 122.0
                        self.client.simSetHomeGeoPoint(home_geo, self.vehicle_name)
                    except:
                        pass  # Continue even if GPS home setting fails
                    
                    self.client.armDisarm(True, self.vehicle_name)
                    print("✓ Drone armed")
                except Exception as arm_e:
                    print(f"⚠ Arm warning: {arm_e} (trying alternative method)")
                    # Alternative: try to enable API control and proceed
                    try:
                        self.client.enableApiControl(True, self.vehicle_name)
                        print("✓ Alternative arming method used")
                    except:
                        print("⚠ Unable to arm drone, but continuing with API control")
                
                # Take off to a safe altitude
                print("🛫 Taking off...")
                try:
                    takeoff_result = self.client.takeoffAsync(vehicle_name=self.vehicle_name)
                    takeoff_result.join()
                    print("✓ Basic takeoff completed")
                except Exception as takeoff_e:
                    print(f"⚠ Takeoff issue: {takeoff_e}")
                    # Try moving to hover position as alternative
                    print("🛫 Moving to hover position...")
                    self.client.moveToPositionAsync(0, 0, -2, 5, vehicle_name=self.vehicle_name).join()
                    print("✓ Moved to hover position")
                
                # Move to safe replay starting altitude
                safe_altitude = -self.start_altitude  # Convert to NED coordinates (negative Z)
                print(f"⬆️ Moving to safe replay altitude ({self.start_altitude} meters)...")
                
                # Get current position to maintain X, Y coordinates
                current_pose = self.client.simGetVehiclePose(self.vehicle_name)
                start_x = current_pose.position.x_val
                start_y = current_pose.position.y_val
                
                # Move to safe altitude while maintaining horizontal position
                move_result = self.client.moveToPositionAsync(
                    start_x, start_y, safe_altitude, 
                    3,  # 3 m/s speed
                    vehicle_name=self.vehicle_name
                )
                move_result.join()
                
                print(f"✓ Positioned at safe altitude: ({start_x:.1f}, {start_y:.1f}, {self.start_altitude:.1f}m)")
            
            # Wait for stable hover
            print("⏳ Stabilizing...")
            time.sleep(3)
            
            # Verify final position
            final_pose = self.client.simGetVehiclePose(self.vehicle_name)
            final_alt = abs(final_pose.position.z_val)
            print(f"✓ Drone ready for replay at {final_alt:.1f}m altitude")
            
            return True
            
        except Exception as e:
            print(f"✗ Error preparing drone: {e}")
            return False
    
    def interpolate_data(self, df):
        """Interpolate recorded data to higher frequency for smoother control."""
        print(f"🔄 Interpolating data from 30Hz to {self.control_frequency}Hz for smoother flight...")
        
        # Original time points (30 fps)
        original_times = df['Video Time (s)'].values
        
        # Create new time points at higher frequency
        start_time = original_times[0]
        end_time = original_times[-1]
        dt = 1.0 / self.control_frequency  # Time step for high frequency
        new_times = np.arange(start_time, end_time + dt, dt)
        
        print(f"   • Original data points: {len(df)} at 30Hz")
        print(f"   • Interpolated points: {len(new_times)} at {self.control_frequency}Hz")
        
        # Prepare interpolated dataframe
        interpolated_data = {
            'Video Time (s)': new_times,
            'Frame': np.interp(new_times, original_times, df['Frame'].values)
        }
        
        # Interpolate velocity data
        velocity_columns = ['X (m/s)', 'Y (m/s)', 'Z (m/s)']
        for col in velocity_columns:
            if col in df.columns:
                interpolated_data[col] = np.interp(new_times, original_times, df[col].values)
            else:
                interpolated_data[col] = np.zeros(len(new_times))
        
        # Interpolate orientation data if available
        orientation_columns = ['Drone Roll (deg)', 'Drone Pitch (deg)', 'Drone Yaw (deg)']
        has_orientation = all(col in df.columns for col in orientation_columns)
        
        if has_orientation:
            for col in orientation_columns:
                # Handle angle wraparound for smooth interpolation
                angles = df[col].values
                # Unwrap angles to avoid 360° jumps
                unwrapped_angles = np.unwrap(np.radians(angles))
                interpolated_angles = np.interp(new_times, original_times, unwrapped_angles)
                # Convert back to degrees
                interpolated_data[col] = np.degrees(interpolated_angles)
        else:
            for col in orientation_columns:
                interpolated_data[col] = np.zeros(len(new_times))
        
        # Create interpolated dataframe
        interpolated_df = pd.DataFrame(interpolated_data)
        
        print(f"✓ Data interpolation completed")
        return interpolated_df, has_orientation
    
    def euler_to_quaternion(self, roll_deg, pitch_deg, yaw_deg):
        """Convert Euler angles (degrees) to quaternion."""
        # Convert to radians
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        
        # Calculate quaternion components
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return airsim.Quaternionr(x, y, z, w)
    
    def replay_flight(self, df, has_orientation):
        """Execute the flight replay using recorded data."""
        print(f"🎬 Starting flight replay...")
        
        # Interpolate data for smoother control
        interpolated_df, has_orientation = self.interpolate_data(df)
        
        print(f"📊 Interpolated frames: {len(interpolated_df)}, Speed: {self.replay_speed}x")
        print(f"🎛️ Control frequency: {self.control_frequency}Hz")
        print("Press Ctrl+C to stop replay anytime.\n")
        
        start_time = time.time()
        successful_commands = 0
        control_interval = 1.0 / self.control_frequency
        
        try:
            for index, row in interpolated_df.iterrows():
                # Check for interrupt signal
                if self.interrupted:
                    print("⚠ Stopping replay due to interrupt...")
                    break
                    
                # Calculate timing for replay speed
                current_time = time.time()
                expected_time = start_time + (row['Video Time (s)'] / self.replay_speed)
                
                # Wait for proper timing (much more frequent now)
                wait_time = expected_time - current_time
                if wait_time > 0:
                    time.sleep(wait_time)
                elif wait_time < -0.1:  # More tolerance since we're running faster
                    if index % 100 == 0:  # Only report occasionally
                        print(f"⚠ Timing lag at frame {row['Frame']:.0f}")
                    continue
                
                # Extract velocity data
                vel_x = float(row.get('X (m/s)', 0))
                vel_y = float(row.get('Y (m/s)', 0))
                vel_z = float(row.get('Z (m/s)', 0))
                
                # Handle invalid values
                if not all(math.isfinite(v) for v in [vel_x, vel_y, vel_z]):
                    vel_x = vel_y = vel_z = 0
                
                # Apply velocity command with shorter duration for responsive control
                try:
                    # Use control interval as duration for smooth continuous control
                    duration = control_interval * 1.5  # Slightly longer than interval for overlap
                    
                    # Send velocity command (non-blocking for better performance)
                    self.client.moveByVelocityAsync(
                        vel_x, vel_y, vel_z,
                        duration,
                        airsim.DrivetrainType.MaxDegreeOfFreedom,
                        airsim.YawMode(False, 0),  # Don't control yaw through velocity
                        vehicle_name=self.vehicle_name
                    )
                    
                    successful_commands += 1
                    
                    # Apply drone orientation if available (less frequently to avoid jitter)
                    if has_orientation and index % 5 == 0:  # Update orientation every 5th command (~10Hz)
                        drone_roll = float(row.get('Drone Roll (deg)', 0))
                        drone_pitch = float(row.get('Drone Pitch (deg)', 0))
                        drone_yaw = float(row.get('Drone Yaw (deg)', 0))
                        
                        # Check for valid orientation data
                        if all(math.isfinite(v) for v in [drone_roll, drone_pitch, drone_yaw]):
                            # Only apply significant orientation changes to avoid jitter
                            if abs(drone_roll) > 0.1 or abs(drone_pitch) > 0.1 or abs(drone_yaw) > 0.1:
                                try:
                                    orientation_quat = self.euler_to_quaternion(drone_roll, drone_pitch, drone_yaw)
                                    current_pose = self.client.simGetVehiclePose(self.vehicle_name)
                                    new_pose = airsim.Pose(current_pose.position, orientation_quat)
                                    self.client.simSetVehiclePose(new_pose, True, self.vehicle_name)
                                except Exception as ori_e:
                                    if index % 250 == 0:  # Less frequent error reporting
                                        print(f"⚠ Orientation error at frame {row['Frame']:.0f}: {ori_e}")
                    
                except Exception as e:
                    if index % 250 == 0:  # Less frequent error reporting due to higher frequency
                        print(f"⚠ Movement error at frame {row['Frame']:.0f}: {e}")
                    continue
                
                # Progress display less frequently due to higher update rate
                if index % (self.control_frequency * 2) == 0:  # Every 2 seconds
                    progress = (index / len(interpolated_df)) * 100
                    elapsed = time.time() - start_time
                    remaining = (elapsed / max(index, 1)) * (len(interpolated_df) - index)
                    
                    print(f"Progress: {progress:5.1f}% | "
                          f"Frame: {row['Frame']:6.1f} | "
                          f"Elapsed: {elapsed:5.1f}s | "
                          f"ETA: {remaining:5.1f}s | "
                          f"Vel: ({vel_x:5.2f}, {vel_y:5.2f}, {vel_z:5.2f})")
        
        except KeyboardInterrupt:
            print("\n⚠ Replay interrupted by user (Ctrl+C)")
        
        except Exception as e:
            print(f"\n✗ Replay error: {e}")
        
        finally:
            success_rate = (successful_commands / max(len(interpolated_df), 1)) * 100
            print(f"\n📈 Replay Summary:")
            print(f"   • Commands executed: {successful_commands}/{len(interpolated_df)}")
            print(f"   • Success rate: {success_rate:.1f}%")
            print(f"   • Control frequency achieved: ~{successful_commands/(time.time()-start_time):.1f}Hz")
    
    def mavlink_land(self):
        """Land using MAVLink commands."""
        if not self.mavlink_connection:
            return False
            
        try:
            print("🛬 Using MAVLink for landing...")
            
            # Send land command
            self.mavlink_connection.mav.command_long_send(
                self.mavlink_connection.target_system,
                self.mavlink_connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,  # confirmation
                0,  # abort altitude
                0,  # precision land mode
                0,  # empty
                0,  # yaw angle
                0,  # latitude
                0,  # longitude
                0   # altitude
            )
            
            # Wait for landing
            start_time = time.time()
            while time.time() - start_time < 20:  # 20 second timeout
                try:
                    current_pose = self.client.simGetVehiclePose(self.vehicle_name)
                    current_alt = abs(current_pose.position.z_val)
                    if current_alt < 0.5:  # Close to ground
                        print("✓ MAVLink landing completed")
                        return True
                    time.sleep(0.5)
                except:
                    time.sleep(0.5)
                    continue
            
            print("⚠ MAVLink landing timeout")
            return False
            
        except Exception as e:
            print(f"⚠ MAVLink landing error: {e}")
            return False
    
    def land_and_cleanup(self):
        """Safely land drone and cleanup."""
        try:
            print("\n🛬 Landing sequence...")
            
            # Stop all movement
            try:
                self.client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=self.vehicle_name).join()
                time.sleep(1)
            except KeyboardInterrupt:
                print("⚠ Landing interrupted, forcing stop...")
                # Force stop by setting velocity to zero without waiting
                try:
                    self.client.moveByVelocityAsync(0, 0, 0, 0.1, vehicle_name=self.vehicle_name)
                except:
                    pass
            
            # Try MAVLink landing first
            mavlink_land_success = False
            if self.mavlink_connection:
                mavlink_land_success = self.mavlink_land()
            
            if not mavlink_land_success:
                # Land the drone using AirSim
                try:
                    print("🛬 Using AirSim for landing...")
                    land_result = self.client.landAsync(vehicle_name=self.vehicle_name)
                    land_result.join()
                except KeyboardInterrupt:
                    print("⚠ Landing command interrupted")
                except Exception as land_e:
                    print(f"⚠ Landing warning: {land_e}")
            
            # Cleanup
            try:
                if self.mavlink_connection:
                    # Disarm using MAVLink
                    self.mavlink_connection.mav.command_long_send(
                        self.mavlink_connection.target_system,
                        self.mavlink_connection.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                        0,  # confirmation
                        0,  # disarm
                        0, 0, 0, 0, 0, 0  # unused parameters
                    )
                    print("✓ MAVLink disarm command sent")
                
                self.client.armDisarm(False, self.vehicle_name)
                self.client.enableApiControl(False, self.vehicle_name)
                print("✓ Landing and cleanup completed")
            except Exception as cleanup_e:
                print(f"⚠ Cleanup warning: {cleanup_e}")
            
        except Exception as e:
            print(f"⚠ Cleanup error: {e}")

def find_sensor_logs():
    """Find sensor log files in current directory."""
    current_dir = Path('.')
    sensor_files = list(current_dir.glob('*_sensor_log.xlsx'))
    return sorted(sensor_files, key=lambda f: f.stat().st_mtime, reverse=True)

def main():
    print("🚁 AirSim Flight Replay System")
    print("=" * 50)
    print("This script replays recorded drone flights using velocity and orientation data.\n")
    
    # Initialize controller
    controller = FlightReplayController()
    
    # Connect to AirSim
    if not controller.connect():
        print("❌ Cannot proceed without AirSim connection.")
        input("Press Enter to exit...")
        return
    
    # Find available sensor log files
    sensor_logs = find_sensor_logs()
    
    if not sensor_logs:
        print("📁 No sensor log files found in current directory.")
        sensor_file = input("Enter full path to sensor log Excel file: ").strip()
        if not Path(sensor_file).exists():
            print("❌ File not found.")
            return
    else:
        print(f"📁 Found {len(sensor_logs)} sensor log files:")
        for i, log_file in enumerate(sensor_logs[:5], 1):  # Show max 5 recent files
            file_time = time.ctime(log_file.stat().st_mtime)
            file_size = log_file.stat().st_size / 1024  # KB
            print(f"   {i}. {log_file.name}")
            print(f"      Created: {file_time}, Size: {file_size:.1f} KB")
        
        if len(sensor_logs) > 5:
            print(f"   ... and {len(sensor_logs) - 5} more files")
        
        print()
        choice = input(f"Select file (1-{min(5, len(sensor_logs))}) or enter custom path: ").strip()
        
        try:
            file_index = int(choice) - 1
            if 0 <= file_index < min(5, len(sensor_logs)):
                sensor_file = str(sensor_logs[file_index])
            else:
                raise ValueError()
        except ValueError:
            sensor_file = choice
            if not Path(sensor_file).exists():
                print("❌ File not found.")
                return
    
    # Configure replay speed
    try:
        speed_input = input(f"⚡ Replay speed multiplier (default 1.0): ").strip()
        if speed_input:
            controller.replay_speed = float(speed_input)
            if controller.replay_speed <= 0:
                raise ValueError("Speed must be positive")
    except ValueError:
        print("⚠ Invalid speed, using default 1.0")
        controller.replay_speed = 1.0
    
    # Configure starting altitude
    try:
        altitude_input = input(f"📏 Starting altitude in meters (default {controller.start_altitude}): ").strip()
        if altitude_input:
            controller.start_altitude = float(altitude_input)
            if controller.start_altitude <= 0:
                raise ValueError("Altitude must be positive")
    except ValueError:
        print(f"⚠ Invalid altitude, using default {controller.start_altitude}m")
    
    # Configure control frequency
    try:
        freq_input = input(f"🎛️ Control frequency in Hz (default {controller.control_frequency}, recommended 50-100): ").strip()
        if freq_input:
            controller.control_frequency = float(freq_input)
            if controller.control_frequency < 10 or controller.control_frequency > 200:
                print("⚠ Warning: Frequency outside recommended range (10-200 Hz)")
    except ValueError:
        print(f"⚠ Invalid frequency, using default {controller.control_frequency}Hz")
    
    # Configure MAVLink port (optional)
    mavlink_input = input(f"🔗 MAVLink connection (default {controller.mavlink_port}, or 'none' to skip): ").strip()
    if mavlink_input.lower() == 'none':
        controller.mavlink_connection = None
        print("⚠ MAVLink disabled, using AirSim API only")
    elif mavlink_input:
        controller.mavlink_port = mavlink_input
    
    # Load and validate data
    print(f"\n📊 Loading data from: {Path(sensor_file).name}")
    result = controller.load_data(sensor_file)
    
    if result[0] is None:
        print("❌ Failed to load sensor data.")
        return
    
    df, has_orientation = result
    
    # Display data summary
    print(f"\n📋 Flight Data Summary:")
    print(f"   • Total frames: {len(df)}")
    print(f"   • Duration: {df['Video Time (s)'].max():.1f} seconds")
    print(f"   • Replay duration: {df['Video Time (s)'].max() / controller.replay_speed:.1f} seconds")
    print(f"   • Has drone orientation: {'Yes' if has_orientation else 'No'}")
    print(f"   • Replay speed: {controller.replay_speed}x")
    print(f"   • Starting altitude: {controller.start_altitude}m")
    print(f"   • Control frequency: {controller.control_frequency}Hz")
    
    # Max velocity check
    max_vel = max(
        abs(df['X (m/s)'].max()), abs(df['Y (m/s)'].max()), abs(df['Z (m/s)'].max())
    )
    if max_vel > 10:  # Safety check
        print(f"⚠ Warning: High velocity detected ({max_vel:.1f} m/s)")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Replay cancelled for safety.")
            return
    
    # Final confirmation
    print(f"\n🎬 Ready to replay flight!")
    confirm = input("Start replay? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Replay cancelled.")
        return
    
    # Execute the replay
    try:
        if controller.prepare_drone():
            controller.replay_flight(df, has_orientation)
        else:
            print("❌ Failed to prepare drone for replay.")
    finally:
        controller.land_and_cleanup()
    
    print("\n🎉 Replay session completed!")
    try:
        input("Press Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")

if __name__ == '__main__':
    main()
