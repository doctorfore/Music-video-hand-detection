import cv2
import numpy as np
import mediapipe as mp

# 初始化 MediaPipe Hands（如果后面要用到手势检测）
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


def get_camera():
    """
    自动寻找第一个可用摄像头，优先尝试 index=1 (MacBook 内置摄像头)。
    """
    for i in [1, 0, 2, 3, 4]:
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"✅ Using camera index {i}")
            return cap
    raise RuntimeError(" No available camera found!")


def detect_piano_keyboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    keyboard_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        # 简单规则：长方形且在画面下半部分
        if aspect_ratio > 3 and y + h > frame.shape[0] * 0.6:
            keyboard_contours.append((x, y, w, h))

    return keyboard_contours


def extract_white_keys(keyboard_region, frame):
    x, y, w, h = keyboard_region
    keyboard_roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(keyboard_roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white_keys = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x_key, y_key, w_key, h_key = cv2.boundingRect(contour)
            white_keys.append({
                'x': x + x_key,
                'y': y + y_key,
                'w': w_key,
                'h': h_key,
                'note': None,
                'active': False
            })
    return white_keys


def detect_black_keys(keyboard_region, frame, white_keys):
    x, y, w, h = keyboard_region
    keyboard_roi = frame[y:y+h, x:x+w]

    hsv = cv2.cvtColor(keyboard_roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_keys = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x_key, y_key, w_key, h_key = cv2.boundingRect(contour)
            black_keys.append({
                'x': x + x_key,
                'y': y + y_key,
                'w': w_key,
                'h': h_key,
                'note': None,
                'active': False
            })
    return black_keys


# ------------------- 主程序 -------------------
if __name__ == "__main__":
    cap = get_camera()  # 自动检测并打开可用摄像头

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Can't receive frame. Exiting...")
            break

        # 检测钢琴键盘区域
        keyboards = detect_piano_keyboard(frame)
        for (x, y, w, h) in keyboards:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            white_keys = extract_white_keys((x, y, w, h), frame)
            black_keys = detect_black_keys((x, y, w, h), frame, white_keys)

            # 画白键
            for key in white_keys:
                cv2.rectangle(frame, (key['x'], key['y']),
                              (key['x']+key['w'], key['y']+key['h']),
                              (255, 255, 255), 1)

            # 画黑键
            for key in black_keys:
                cv2.rectangle(frame, (key['x'], key['y']),
                              (key['x']+key['w'], key['y']+key['h']),
                              (0, 0, 0), 1)

        cv2.imshow("Piano Keyboard Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
