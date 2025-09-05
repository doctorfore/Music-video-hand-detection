import cv2, mediapipe as mp

cap = cv2.VideoCapture(0)  # Open up the camera
hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

            # Detection direction of the index finger ：IF tip(8) is above pip(6) 
            pts = [(lm.x, lm.y) for lm in hand.landmark]
            if pts[8][1] < pts[6][1]:
                cv2.putText(frame, "Hands UP", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "Hands DOWN", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break


cap.release()
cv2.destroyAllWindows()