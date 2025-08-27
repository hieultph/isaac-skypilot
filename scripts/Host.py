# Phiên bản đa luồng để cải thiện hiệu suất streaming.
# Luồng nền sẽ lấy ảnh từ AirSim, luồng chính sẽ phục vụ client.

import airsim
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
import time
import threading

# --- Khởi tạo Ứng dụng Flask ---
app = Flask(__name__)

# --- Biến Toàn cục ---
AIRSIM_CLIENT = None
# Tên camera mặc định.
CAMERA_NAME = "0"
# Loại ảnh để chụp.
IMAGE_TYPE = airsim.ImageType.Scene

# --- Biến cho việc chia sẻ khung hình giữa các luồng ---
latest_frame_lock = threading.Lock()
latest_frame_jpeg = None
stop_thread = threading.Event()

def connect_to_airsim():
    """Kết nối với client AirSim và xử lý các lỗi có thể xảy ra."""
    global AIRSIM_CLIENT
    try:
        print("Đang kết nối với AirSim...")
        AIRSIM_CLIENT = airsim.MultirotorClient()
        AIRSIM_CLIENT.confirmConnection()
        print("Đã kết nối thành công với AirSim.")
        return True
    except Exception as e:
        print(f"Lỗi khi kết nối với AirSim: {e}")
        print("Vui lòng đảm bảo mô phỏng AirSim đang chạy và có thể truy cập.")
        AIRSIM_CLIENT = None
        return False

def airsim_frame_updater_thread():
    """
    Luồng nền (background thread) chuyên để lấy và xử lý khung hình từ AirSim.
    """
    global latest_frame_jpeg, latest_frame_lock, stop_thread

    print("[Thread] Luồng cập nhật khung hình đã bắt đầu.")
    while not stop_thread.is_set():
        try:
            # Kiểm tra kết nối AirSim
            if not (AIRSIM_CLIENT and AIRSIM_CLIENT.ping()):
                print("[Thread] Mất kết nối, đang thử kết nối lại...")
                if not connect_to_airsim():
                    time.sleep(2) # Chờ 2 giây trước khi thử lại
                    continue
            
            # Yêu cầu ảnh không nén từ AirSim
            request = airsim.ImageRequest(CAMERA_NAME, IMAGE_TYPE, False, False)
            responses = AIRSIM_CLIENT.simGetImages([request])
            
            if not (responses and responses[0].image_data_uint8 and responses[0].width > 0):
                time.sleep(0.01)
                continue

            response = responses[0]
            
            # Logic xử lý video đã được xác minh là chính xác
            img_np = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_bgr = img_np.reshape(response.height, response.width, 3)


            if img_bgr is None:
                continue

            # Mã hóa khung hình thành JPEG
            ret, buffer = cv2.imencode('.jpg', img_bgr)
            if not ret:
                continue
            
            # Cập nhật khung hình mới nhất một cách an toàn
            with latest_frame_lock:
                latest_frame_jpeg = buffer.tobytes()

        except Exception as e:
            print(f"[Thread] Đã xảy ra lỗi: {e}")
            time.sleep(1)
    
    print("[Thread] Luồng cập nhật khung hình đã dừng.")

def generate_frames_for_client():
    """
    Hàm generator này chỉ lấy khung hình đã xử lý sẵn và gửi cho client.
    """
    global latest_frame_jpeg, latest_frame_lock
    
    while True:
        with latest_frame_lock:
            frame_to_send = latest_frame_jpeg

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
        



@app.route('/video_feed')
def video_feed():
    """Đường dẫn phát luồng video."""
    return Response(generate_frames_for_client(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Trang chủ hiển thị luồng video."""
    return render_template_string("""
        <html>
            <head>
                <title>AirSim Threaded Streamer</title>
                <style>
                    body { font-family: sans-serif; text-align: center; background-color: #222; color: #eee; }
                    img { border: 2px solid #555; max-width: 90vw; max-height: 80vh; }
                </style>
            </head>
            <body>
                <h1>AirSim Threaded Streamer</h1>
                <img src="/video_feed" alt="AirSim Video Feed">
            </body>
        </html>
    """)

if __name__ == '__main__':
    if connect_to_airsim():
        # Khởi tạo và bắt đầu luồng nền
        updater_thread = threading.Thread(target=airsim_frame_updater_thread, daemon=True)
        updater_thread.start()
        
        # Chạy máy chủ Flask
        print("[Main] Bắt đầu chạy máy chủ Flask...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
        # Khi máy chủ dừng (ví dụ: Ctrl+C), báo cho luồng nền dừng lại
        print("[Main] Máy chủ đã dừng, ra hiệu cho luồng nền kết thúc.")
        stop_thread.set()
        updater_thread.join(timeout=2) # Chờ luồng nền kết thúc
        print("[Main] Chương trình đã thoát.")
    else:
        print("[Main] Không thể khởi động máy chủ Flask vì không kết nối được với AirSim.")
