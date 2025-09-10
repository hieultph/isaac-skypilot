import cv2
import time
import math
import threading
from pymavlink import mavutil
import numpy as np

# ===== THÔNG SỐ CAMERA =====
FOCAL_LENGTH_MM = 8       # Tiêu cự (mm)
SENSOR_HEIGHT_MM = 8      # Chiều cao cảm biến (mm)
REAL_HEIGHT_MM = 180      # Chiều cao thật của vật thể (mm) - Điều chỉnh theo đối tượng cần track

# ===== THÔNG SỐ ĐIỀU KHIỂN DRONE =====
TARGET_DISTANCE_MIN = 50  # Khoảng cách tối thiểu (cm)
TARGET_DISTANCE_MAX = 60  # Khoảng cách tối đa (cm)
TARGET_DISTANCE_OPTIMAL = 55  # Khoảng cách tối ưu (cm)
FIXED_ALTITUDE = 3.0      # Độ cao cố định (m)

# Tốc độ điều khiển - TINH CHỈNH để giảm dao động
MAX_XY_VELOCITY = 1.0     # Tốc độ tối đa theo X,Y (m/s)
MAX_Z_VELOCITY = 0.5      # Tốc độ tối đa theo Z (m/s)
MAX_YAW_RATE = 0.08       # Tốc độ yaw tối đa (rad/s) - Tăng nhẹ từ 0.05
CENTERING_GAIN = 1.0      # Hệ số điều khiển căn giữa
DISTANCE_GAIN = 0.8       # Hệ số điều khiển khoảng cách

# Ngưỡng điều khiển - THÊM DEAD ZONE để tránh dao động
YAW_ERROR_THRESHOLD = 60  # Ngưỡng lỗi X để bắt đầu yaw (pixel)
YAW_STOP_THRESHOLD = 25   # Ngưỡng để dừng yaw (pixel)
MOVE_ERROR_THRESHOLD = 15 # Ngưỡng lỗi Y để tiến tới sau khi yaw xong (pixel)
STABLE_FRAMES_REQUIRED = 3  # Số frame ổn định cần thiết để chuyển state

# DEAD ZONE - vùng "đủ tốt" để không cần điều chỉnh
DEAD_ZONE_X = 40          # Vùng chết cho X (pixel) - không yaw khi lỗi < 40px
DEAD_ZONE_Y = 25          # Vùng chết cho Y (pixel) - không di chuyển khi lỗi < 25px

# ===== ĐƯỜNG DẪN VIDEO =====
video_path = "http://10.5.9.53:5000/video_feed"

# ===== BIẾN TOÀN CỤC với THREAD LOCK =====
velocity_lock = threading.Lock()  # Lock để đồng bộ hóa

running = True
drone_ready = False       # Drone đã lên 3m chưa
roi_mode = False          # Đang chọn ROI
tracking_active = False   # Đang tracking
current_mode = "POSITION"
current_target_altitude = FIXED_ALTITUDE
current_target_x = 0.0
current_target_y = 0.0

# Velocity variables - protected by lock
current_vx = 0.0          # Vận tốc X (m/s)
current_vy = 0.0          # Vận tốc Y (m/s) 
current_vz = 0.0          # Vận tốc Z (m/s)
current_yaw_rate = 0.0    # Tốc độ yaw (rad/s)

# Tracking variables
tracker = None
bbox = None
selecting = False
ix, iy = -1, -1
object_center_x = 0
object_center_y = 0
current_distance = 0
frame = None

                # Tracking state để tránh dao động yaw  
yaw_state = "IDLE"  # IDLE, YAWING, MOVING, LOCKED
yaw_stable_count = 0  # Đếm số frame ổn định để tránh giật lag
lock_stable_count = 0  # Đếm frame khi đã lock target

# Debug counter
velocity_set_count = 0
velocity_read_count = 0

# ===== KẾT NỐI MAVLINK =====
try:
    print("Đang kết nối tới PX4 SITL...")
    master = mavutil.mavlink_connection('udpin:0.0.0.0:14555')
    print("Đang chờ HEARTBEAT...")
    master.wait_heartbeat()
    print(f"Heartbeat nhận được từ system {master.target_system}, component {master.target_component}")
    time.sleep(1)
    print("Đã kết nối thành công với UAV")
except Exception as e:
    print(f"Lỗi kết nối MAVLink: {e}")
    exit()

def set_velocity(vx, vy, vz, yaw_rate=0.0):
    """Thread-safe function để set velocity và yaw rate"""
    global current_vx, current_vy, current_vz, current_yaw_rate, velocity_set_count
    
    with velocity_lock:
        current_vx = vx
        current_vy = vy
        current_vz = vz
        current_yaw_rate = yaw_rate
        velocity_set_count += 1
        print(f"SET VELOCITY #{velocity_set_count}: Vx={vx:.3f}, Vy={vy:.3f}, Vz={vz:.3f}, YawRate={yaw_rate:.3f}")

def get_velocity():
    """Thread-safe function để đọc velocity và yaw rate"""
    global velocity_read_count
    
    with velocity_lock:
        vx = current_vx
        vy = current_vy
        vz = current_vz
        yaw_rate = current_yaw_rate
        velocity_read_count += 1
        print(f"READ VELOCITY #{velocity_read_count}: Vx={vx:.3f}, Vy={vy:.3f}, Vz={vz:.3f}, YawRate={yaw_rate:.3f}")
        return vx, vy, vz, yaw_rate

def get_telemetry():
    """Lấy telemetry từ drone"""
    try:
        # Clear old messages
        while master.recv_match(blocking=False):
            pass
        
        # Get fresh position
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=0.5)
        if msg:
            return {
                'x': msg.x,
                'y': msg.y,
                'z': -msg.z,  # Convert NED to altitude
                'vx': msg.vx,
                'vy': msg.vy,
                'vz': -msg.vz
            }
    except:
        pass
    return None

def send_setpoint_thread():
    """Thread gửi setpoint liên tục - Sử dụng BODY FRAME cho di chuyển"""
    global running, current_mode, drone_ready
    
    while running:
        try:
            if tracking_active and drone_ready:
                # Tracking mode - đọc velocity và yaw rate thread-safe
                vx, vy, vz, yaw_rate = get_velocity()
                
                if abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vz) > 0.01 or abs(yaw_rate) > 0.01:
                    print(f"DRONE ACTIVE! Vx={vx:.3f}, Vy={vy:.3f}, Vz={vz:.3f}, YawRate={yaw_rate:.3f}")
                else:
                    print(f"Drone hovering")
                
                # SỬ DỤNG BODY FRAME để di chuyển theo hướng drone
                master.mav.set_position_target_local_ned_send(
                    0,  # time_boot_ms
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_FRAME_BODY_NED,  # THAY ĐỔI: Dùng BODY frame thay vì LOCAL
                    0b0000011111000011,  # type_mask: Use vx, vy, Z position và yaw_rate
                    0, 0, -FIXED_ALTITUDE,  # Position Z cố định 3m, X,Y ignored
                    vx, vy, 0,  # Velocity trong BODY frame: VX=tiến/lùi, VY=trái/phải
                    0, 0, 0,  # Acceleration (ignored)
                    0, yaw_rate  # Yaw (ignored), yaw_rate (used)
                )
                
            else:
                # Position mode - giữ vị trí cố định
                master.mav.set_position_target_local_ned_send(
                    0,
                    master.target_system,
                    master.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    0b0000111111111000,  # Use position only
                    current_target_x, current_target_y, -current_target_altitude,
                    0, 0, 0,  # Velocity (ignored)
                    0, 0, 0,  # Acceleration (ignored)
                    0, 0  # Yaw (ignored)
                )
            
            time.sleep(0.1)  # 10Hz
            
        except Exception as e:
            if running:
                print(f"Lỗi setpoint: {e}")
            time.sleep(0.1)

def arm_and_offboard():
    """ARM drone và chuyển sang OFFBOARD mode"""
    global current_target_x, current_target_y, current_target_altitude
    
    print("\nCHUẨN BỊ BAY")
    
    # Get initial position
    telem = get_telemetry()
    if telem:
        current_target_x = telem['x']
        current_target_y = telem['y']
        current_target_altitude = 0.5
        print(f"  Vị trí ban đầu: X={telem['x']:.2f}, Y={telem['y']:.2f}")
    else:
        current_target_x = 0
        current_target_y = 0
        current_target_altitude = 0.5
    
    # Send initial setpoints
    print("  Gửi setpoint ban đầu...")
    for i in range(200):
        master.mav.set_position_target_local_ned_send(
            0, master.target_system, master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,
            current_target_x, current_target_y, -0.5,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        time.sleep(0.005)
    
    # ARM
    print("  ARM drone...")
    for i in range(5):
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        time.sleep(0.2)
    
    # Switch to OFFBOARD
    print("  Chuyển sang OFFBOARD mode...")
    for i in range(5):
        master.set_mode_px4('OFFBOARD', 0, 0)
        time.sleep(0.2)
    
    print("Sẵn sàng bay\n")
    return True

def takeoff_and_hold():
    """Cất cánh lên 3m và giữ vị trí cố định"""
    global current_target_altitude, drone_ready
    
    print("CẤT CÁNH LÊN 3M VÀ GIỮ VỊ TRÍ CỐ ĐỊNH")
    
    # Lấy vị trí hiện tại để giữ X, Y cố định
    telem = get_telemetry()
    if telem:
        current_target_x = telem['x']
        current_target_y = telem['y']
        print(f"  Giữ vị trí X={current_target_x:.2f}, Y={current_target_y:.2f}")
    
    # Cất cánh từng bước
    altitudes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for alt in altitudes:
        current_target_altitude = alt
        print(f"  Lên {alt:.1f}m...")
        time.sleep(2)
        
        # Kiểm tra độ cao
        telem = get_telemetry()
        if telem:
            error = abs(telem['z'] - alt)
            if error < 0.3:
                print(f"    Đạt {alt:.1f}m (thực: {telem['z']:.2f}m)")
    
    # Cố định tại 3m
    current_target_altitude = FIXED_ALTITUDE
    print(f"  Cố định tại độ cao {FIXED_ALTITUDE}m")
    time.sleep(3)  # Chờ ổn định
    
    drone_ready = True
    print("Drone đã sẵn sàng tại độ cao 3m")
    print("Nhấn 'S' để bắt đầu chọn đối tượng ROI")
    print("="*50)
    
    return True

def draw_roi(event, x, y, flags, param):
    """Callback cho việc chọn ROI bằng chuột"""
    global ix, iy, selecting, bbox, tracker, frame, roi_mode

    if not roi_mode:  # Chỉ cho phép chọn ROI khi ở chế độ ROI
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        frame2 = frame.copy()
        cv2.rectangle(frame2, (ix, iy), (x, y), (255, 0, 0), 2)
        cv2.putText(frame2, "Selecting ROI...", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Drone Object Tracking", frame2)

    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        w, h = x - ix, y - iy
        if w > 20 and h > 20:  # Đảm bảo ROI đủ lớn
            bbox = (ix, iy, w, h)
            
            # Khởi tạo tracker
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            print(f"Đã chọn đối tượng: ROI {bbox}")
            
            # Reset velocity và state khi chọn lại
            set_velocity(0.0, 0.0, 0.0, 0.0)
            global yaw_state, yaw_stable_count, lock_stable_count
            yaw_state = "IDLE"
            yaw_stable_count = 0
            lock_stable_count = 0
            
            # Tự động bắt đầu tracking
            roi_mode = False
            global tracking_active
            tracking_active = True
            print("BẮT ĐẦU TRACKING ĐỐI TƯỢNG!")
        else:
            print("ROI quá nhỏ, chọn lại!")

def calculate_distance(object_height_px, image_height_px):
    """Tính khoảng cách từ đối tượng đến camera"""
    if object_height_px <= 0:
        return 0
    
    # Quy đổi sang mm trên cảm biến
    object_height_mm_on_sensor = object_height_px * (SENSOR_HEIGHT_MM / image_height_px)
    
    # Tính khoảng cách
    distance_mm = (REAL_HEIGHT_MM * FOCAL_LENGTH_MM) / object_height_mm_on_sensor
    distance_cm = distance_mm / 10
    
    return distance_cm

def video_processing_thread():
    """Thread xử lý video và tracking"""
    global frame, tracking_active, object_center_x, object_center_y, current_distance
    global tracker, bbox, roi_mode, drone_ready
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video feed")
        return
    
    cv2.namedWindow("Drone Object Tracking")
    cv2.setMouseCallback("Drone Object Tracking", draw_roi)
    
    print("Video feed đã sẵn sàng")
    
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Vẽ khung giữa màn hình
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        
        # Crosshair
        cv2.line(frame, (frame_center_x-30, frame_center_y), (frame_center_x+30, frame_center_y), (0, 255, 0), 2)
        cv2.line(frame, (frame_center_x, frame_center_y-30), (frame_center_x, frame_center_y+30), (0, 255, 0), 2)
        
        # Hiển thị trạng thái
        info_y = 30
        
        if not drone_ready:
            cv2.putText(frame, "DRONE PREPARING...", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif roi_mode:
            cv2.putText(frame, "ROI MODE - Click and drag to select object", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif tracking_active and tracker is not None:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Tính tâm đối tượng
                object_center_x = x + w // 2
                object_center_y = y + h // 2
                
                # Vẽ tâm đối tượng
                cv2.circle(frame, (object_center_x, object_center_y), 8, (0, 0, 255), -1)
                
                # Vẽ đường từ tâm frame đến tâm object
                cv2.line(frame, (frame_center_x, frame_center_y), 
                        (object_center_x, object_center_y), (255, 0, 0), 2)
                
                # Tính khoảng cách
                current_distance = calculate_distance(h, frame_height)
                
                # DEBUG: Kiểm tra trước khi gọi control
                print(f"BEFORE Control: object_center=({object_center_x},{object_center_y}), tracking_active={tracking_active}, tracker={tracker is not None}")
                
                # Điều khiển drone - viết trực tiếp ở đây để tránh scope issues
                
                # Lấy kích thước frame
                frame_height_local, frame_width_local = frame.shape[:2]
                frame_center_x_local = frame_width_local // 2
                frame_center_y_local = frame_height_local // 2
                
                # Tính lỗi vị trí (pixel)
                error_x_local = object_center_x - frame_center_x_local
                error_y_local = object_center_y - frame_center_y_local
                
                print(f"Direct Control: Center=({frame_center_x_local},{frame_center_y_local}), Error=({error_x_local:+d},{error_y_local:+d})")
                
                # STATE MACHINE với DEAD ZONE và LOCK TARGET
                global yaw_state, yaw_stable_count, lock_stable_count
                
                vx = 0.0
                vy = 0.0
                vz = 0.0
                yaw_rate = 0.0
                
                # Kiểm tra xem tâm khung hình có nằm trong bounding box không
                def is_center_in_bbox():
                    # Lấy bbox hiện tại từ tracker
                    if tracker is None or bbox is None:
                        return False
                    
                    # Tính tọa độ bounding box
                    x, y, w, h = [int(v) for v in bbox]
                    bbox_left = x
                    bbox_right = x + w  
                    bbox_top = y
                    bbox_bottom = y + h
                    
                    # Kiểm tra tâm khung hình có nằm trong bbox không
                    center_in_bbox = (bbox_left <= frame_center_x_local <= bbox_right and 
                                     bbox_top <= frame_center_y_local <= bbox_bottom)
                    
                    if center_in_bbox:
                        print(f"CENTER IN BBOX: Frame center ({frame_center_x_local},{frame_center_y_local}) inside bbox [{bbox_left},{bbox_top},{bbox_right},{bbox_bottom}]")
                    
                    return center_in_bbox
                
                # Tính toán yaw_rate với exponential smoothing
                def calculate_smooth_yaw_rate(error_x):
                    max_error = 150
                    normalized_error = max(-1.0, min(1.0, error_x / max_error))
                    base_rate = 0.04
                    if abs(error_x) < 40:
                        decay_factor = abs(error_x) / 40.0
                        return normalized_error * base_rate * decay_factor
                    else:
                        return normalized_error * base_rate
                
                # KIỂM TRA TARGET LOCK trước
                center_in_target = is_center_in_bbox()
                in_dead_zone = abs(error_x_local) <= DEAD_ZONE_X and abs(error_y_local) <= DEAD_ZONE_Y
                
                # STATE MACHINE với TARGET LOCK
                if yaw_state == "LOCKED":
                    # Đã lock target - chỉ ra khỏi lock nếu sai lệch quá lớn
                    if center_in_target or in_dead_zone:
                        lock_stable_count += 1
                        print(f"STATE: LOCKED - stable for {lock_stable_count} frames (center_in_bbox={center_in_target}, in_dead_zone={in_dead_zone})")
                        # Giữ nguyên vận tốc = 0
                    else:
                        # Sai lệch quá lớn - cần điều chỉnh lại
                        lock_stable_count = 0
                        if abs(error_x_local) > YAW_ERROR_THRESHOLD:
                            yaw_state = "YAWING"
                            yaw_rate = calculate_smooth_yaw_rate(error_x_local)
                            print(f"STATE: LOCKED -> YAWING (large error_x={error_x_local})")
                        elif abs(error_y_local) > MOVE_ERROR_THRESHOLD * 2:  # Threshold cao hơn để tránh dao động
                            yaw_state = "MOVING"
                            vx = -0.2 if error_y_local > 0 else 0.2  # Chậm hơn khi đã gần target
                            print(f"STATE: LOCKED -> MOVING (large error_y={error_y_local})")
                        else:
                            yaw_state = "IDLE"
                            print("STATE: LOCKED -> IDLE (minor adjustment needed)")
                
                elif yaw_state == "IDLE":
                    yaw_stable_count = 0
                    lock_stable_count = 0
                    
                    # Kiểm tra xem có cần lock không
                    if center_in_target:
                        yaw_state = "LOCKED"
                        print("STATE: IDLE -> LOCKED (center in bounding box)")
                    elif in_dead_zone:
                        yaw_state = "LOCKED" 
                        print("STATE: IDLE -> LOCKED (in dead zone)")
                    elif abs(error_x_local) > YAW_ERROR_THRESHOLD:
                        yaw_state = "YAWING"
                        yaw_rate = calculate_smooth_yaw_rate(error_x_local)
                        print(f"STATE: IDLE -> YAWING (error_x={error_x_local})")
                    elif abs(error_y_local) > MOVE_ERROR_THRESHOLD:
                        yaw_state = "MOVING"
                        vx = -0.3 if error_y_local > 0 else 0.3
                        print(f"STATE: IDLE -> MOVING (error_y={error_y_local})")
                    else:
                        print("STATE: IDLE - Centered, hovering")
                
                elif yaw_state == "YAWING":
                    lock_stable_count = 0
                    if center_in_target:
                        yaw_state = "LOCKED"
                        yaw_rate = 0.0
                        print("STATE: YAWING -> LOCKED (center reached target)")
                    elif abs(error_x_local) > YAW_STOP_THRESHOLD:
                        yaw_stable_count = 0
                        yaw_rate = calculate_smooth_yaw_rate(error_x_local)
                        print(f"STATE: YAWING continues (error_x={error_x_local})")
                    else:
                        yaw_stable_count += 1
                        if yaw_stable_count >= STABLE_FRAMES_REQUIRED:
                            yaw_state = "IDLE"
                            yaw_rate = 0.0
                            yaw_stable_count = 0
                            print("STATE: YAWING -> IDLE (stabilized)")
                        else:
                            yaw_rate = calculate_smooth_yaw_rate(error_x_local) * 0.5
                            print(f"STATE: YAWING (stabilizing {yaw_stable_count}/{STABLE_FRAMES_REQUIRED})")
                
                elif yaw_state == "MOVING":
                    yaw_stable_count = 0
                    lock_stable_count = 0
                    
                    if center_in_target:
                        yaw_state = "LOCKED"
                        vx = 0.0
                        print("STATE: MOVING -> LOCKED (center reached target)")
                    elif abs(error_x_local) > YAW_ERROR_THRESHOLD:
                        yaw_state = "YAWING"
                        vx = 0.0
                        yaw_rate = calculate_smooth_yaw_rate(error_x_local)
                        print(f"STATE: MOVING -> YAWING (error_x={error_x_local} too large)")
                    elif abs(error_y_local) > MOVE_ERROR_THRESHOLD:
                        vx = -0.3 if error_y_local > 0 else 0.3
                        print(f"STATE: MOVING continues (error_y={error_y_local})")
                    else:
                        yaw_state = "IDLE"
                        vx = 0.0
                        print("STATE: MOVING -> IDLE (centered)")
                
                # Chỉ điều chỉnh altitude khi LOCKED và ổn định
                if yaw_state == "LOCKED" and lock_stable_count > 10 and current_distance > 0:
                    distance_error = current_distance - TARGET_DISTANCE_OPTIMAL
                    if abs(distance_error) > 20:  # Tolerance cao hơn
                        if current_distance < TARGET_DISTANCE_MIN:
                            vz = -0.05  # Rất chậm
                            print(f"LOCKED ALTITUDE: UP - distance={current_distance:.1f}cm")
                        elif current_distance > TARGET_DISTANCE_MAX:
                            vz = 0.05
                            print(f"LOCKED ALTITUDE: DOWN - distance={current_distance:.1f}cm")
                
                # Set velocity với lock system
                set_velocity(vx, vy, vz, yaw_rate)
                
                # DEBUG: Đọc lại để verify
                vx_check, vy_check, vz_check, yaw_check = get_velocity()
                print(f"AFTER Control: velocity_check=({vx_check:.3f}, {vy_check:.3f}, {vz_check:.3f}, {yaw_check:.3f})")
                
                # Hiển thị thông tin tracking
                cv2.putText(frame, f"TRACKING ACTIVE", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                info_y += 35
                # Hiển thị trạng thái tracking
                tracking_status = "ACTIVE" if tracking_active else "INACTIVE"
                cv2.putText(frame, f"Tracking: {tracking_status}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if tracking_active else (0, 0, 255), 2)
                cv2.putText(frame, f"Distance: {current_distance:.1f} cm", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                info_y += 30
                error_x = object_center_x - frame_center_x
                error_y = object_center_y - frame_center_y
                cv2.putText(frame, f"Error: X={error_x:+4d}, Y={error_y:+4d}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                info_y += 25
                vx_display, vy_display, vz_display, yaw_display = get_velocity()
                cv2.putText(frame, f"Control: Vx={vx_display:.2f} Vy={vy_display:.2f} Vz={vz_display:.2f}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                           
                info_y += 20
                cv2.putText(frame, f"YawRate: {yaw_display:.2f} rad/s", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Kiểm tra khoảng cách
                info_y += 25
                if TARGET_DISTANCE_MIN <= current_distance <= TARGET_DISTANCE_MAX:
                    cv2.putText(frame, "DISTANCE: OPTIMAL", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif current_distance < TARGET_DISTANCE_MIN and current_distance > 0:
                    cv2.putText(frame, "DISTANCE: TOO CLOSE", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif current_distance > TARGET_DISTANCE_MAX:
                    cv2.putText(frame, "DISTANCE: TOO FAR", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            else:
                cv2.putText(frame, "TRACKING LOST - Press R to reselect", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                tracking_active = False
        else:
            cv2.putText(frame, "READY - Press S to select object", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Hiển thị hướng dẫn phím
        instructions = [
            "S: Select object (ROI)",
            "R: Reselect object", 
            "ESC: Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (frame_width - 250, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Drone Object Tracking", frame)
        
        # Xử lý phím
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s') or key == ord('S'):  # Bắt đầu chọn ROI
            if drone_ready and not tracking_active:
                roi_mode = True
                print("Chế độ chọn ROI - Nhấp và kéo để chọn đối tượng")
        elif key == ord('r') or key == ord('R'):  # Chọn lại đối tượng
            if drone_ready:
                print("Chọn lại đối tượng...")
                tracker = None
                bbox = None
                tracking_active = False
                roi_mode = True
                # Reset velocity
                set_velocity(0.0, 0.0, 0.0, 0.0)
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main program - Workflow theo yêu cầu"""
    global running
    
    try:
        print("DRONE OBJECT TRACKING SYSTEM")
        print("="*50)
        print("Workflow:")
        print("  1. Drone tự động bay lên 3m và cố định")
        print("  2. Nhấn 'S' để chọn đối tượng ROI")
        print("  3. Drone tự động bám theo đối tượng")
        print("  4. Điều khiển bằng Velocity Control")
        print(f"  5. Duy trì khoảng cách {TARGET_DISTANCE_MIN}-{TARGET_DISTANCE_MAX}cm")
        print("="*50)
        
        # Start setpoint thread
        setpoint_thread = threading.Thread(target=send_setpoint_thread, daemon=True)
        setpoint_thread.start()
        print("Hệ thống điều khiển đã khởi động")
        
        # Start video processing thread
        video_thread = threading.Thread(target=video_processing_thread, daemon=True)
        video_thread.start()
        print("Video processing đã khởi động")
        
        # BƯỚC 1: ARM và cất cánh lên 3m cố định
        if arm_and_offboard():
            time.sleep(1)
            
            if takeoff_and_hold():
                # BƯỚC 2-4: Chờ user chọn ROI và tự động tracking
                print("Hệ thống sẵn sàng - Chờ lệnh từ user...")
                
                # Keep main thread alive
                try:
                    while running:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\nDỪNG KHẨN CẤP!")
        
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        # Cleanup
        running = False
        time.sleep(1)
        
        print("\nHạ cánh...")
        # Reset velocity trước khi land
        set_velocity(0.0, 0.0, 0.0, 0.0)
        
        # Land
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        
        time.sleep(5)
        
        # Disarm
        print("Disarm drone...")
        for i in range(3):
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            time.sleep(0.2)
        
        print("\nKết thúc chương trình an toàn")
        print("="*50)

if __name__ == "__main__":
    main()