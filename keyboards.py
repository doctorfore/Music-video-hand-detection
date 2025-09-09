import cv2
import numpy as np
import mediapipe as mp

# 初始化MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open up the camera

def detect_piano_keyboard(frame):
    """
    自动检测图像中的钢琴键盘区域
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 形态学操作增强键盘边缘
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keyboard_contours = []
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 过滤太小的轮廓
        if area < 5000:
            continue
            
        # 计算轮廓的边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 筛选可能是键盘的轮廓（宽大于高，位于图像底部）
        aspect_ratio = w / h
        if aspect_ratio > 3 and y + h > frame.shape[0] * 0.6:  # 宽高比大且位于底部
            keyboard_contours.append((x, y, w, h))
    
    return keyboard_contours

def extract_white_keys(keyboard_region, frame):
    """
    从检测到的键盘区域中提取白键
    """
    x, y, w, h = keyboard_region
    keyboard_roi = frame[y:y+h, x:x+w]
    
    # 转换为HSV颜色空间，更容易分离黑白
    hsv = cv2.cvtColor(keyboard_roi, cv2.COLOR_BGR2HSV)
    
    # 检测白色区域（白键）
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找白键轮廓
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    white_keys = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 过滤小轮廓
            x_key, y_key, w_key, h_key = cv2.boundingRect(contour)
            # 转换回全局坐标
            white_keys.append({
                'x': x + x_key,
                'y': y + y_key,
                'w': w_key,
                'h': h_key,
                'note': None,  # 可以后续映射音符
                'active': False
            })
    
    return white_keys

def detect_black_keys(keyboard_region, frame, white_keys):
    """
    检测黑键（位于白键之间的上方）
    """
    x, y, w, h = keyboard_region
    keyboard_roi = frame[y:y+h, x:x+w]
    
    # 检测黑色区域
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