import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Capture from webcam
cap = cv2.VideoCapture(0)

# Smoothing setup
prev_x, prev_y = 0, 0
smoothening = 7

def fingers_up(lm_list):
    fingers = []

    # Thumb
    fingers.append(lm_list[4][1] < lm_list[3][1])  # Adjust for flipped image

    # Fingers
    for tip in [8, 12, 16, 20]:
        fingers.append(lm_list[tip][2] < lm_list[tip - 2][2])
    
    return fingers  # [thumb, index, middle, ring, pinky]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_img)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if lm_list:
                # Get finger states
                fingers = fingers_up(lm_list)

                # Mouse Movement (Index finger tip)
                x1, y1 = lm_list[8][1:]
                screen_x = np.interp(x1, (0, w), (0, screen_w))
                screen_y = np.interp(y1, (0, h), (0, screen_h))
                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Left Click (Index + Middle fingers close together and both up)
                x2, y2 = lm_list[12][1:]
                dist_click = math.hypot(x2 - x1, y2 - y1)
                if fingers[1] and fingers[2] and dist_click < 25:
                    pyautogui.click()
                    time.sleep(0.3)

                # Right Click (Thumb + Pinky close)
                x_thumb, y_thumb = lm_list[4][1:]
                x_pinky, y_pinky = lm_list[20][1:]
                dist_right = math.hypot(x_thumb - x_pinky, y_thumb - y_pinky)
                if fingers[0] and fingers[4] and dist_right < 40:
                    pyautogui.click(button='right')
                    time.sleep(0.3)

                # Scroll (Fist)
                if fingers == [False, False, False, False, False]:
                    pyautogui.scroll(-30)
                    time.sleep(0.3)

    cv2.imshow("Hand Tracking Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
