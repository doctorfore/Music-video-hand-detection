import cv2, mediapipe as mp

# MediaPipe hands model 21 nodes index
FINGER_TIPS = {
    "Thumb": 4,     
    "Index": 8,     
    "Middle": 12,   
    "Ring": 16,     
    "Pinky": 20     
}

cap = cv2.VideoCapture(0)  # Open up the camera
hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils

# Add the data counter
hand_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            hand_count += 1
            draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

            # Detection direction of the index finger ：IF tip(8) is above pip(6) 
            pts = [(lm.x, lm.y) for lm in hand.landmark]

            # Print finger tip coordinates
            print(f"\n--- Hand {hand_count} ---")
            for finger_name, tip_index in FINGER_TIPS.items():
                
                if len(pts) > tip_index:
                    x, y = pts[tip_index]

                    # pixel coordinates ?? if we need to use it
                    h, w, _ = frame.shape
                    px = int(x * w)
                    py = int(y * h)
                    
                    print(f"{finger_name} Tip: ({x:.3f}, {y:.3f}) [Pixel: ({px}, {py})]")

            # Safty Check for the hand landmark
            if len(pts) > 6 :
                if pts[8][1] < pts[6][1]:
                    cv2.putText(frame, "Hands UP", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    cv2.putText(frame, "Hands DOWN", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break


cap.release()
cv2.destroyAllWindows()