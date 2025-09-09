import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open up the camera

def detect_piano_keyboard(frame):
    """
    """
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