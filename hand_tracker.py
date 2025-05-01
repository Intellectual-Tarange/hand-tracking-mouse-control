import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# Helpers
def calculate_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def fingers_up(landmarks):
    tips = [4, 8, 12, 16, 20]
    pips = [2, 6, 10, 14, 18]
    return [landmarks[tip].y < landmarks[pip].y for tip, pip in zip(tips, pips)]

# State variables
prev_x, prev_y = 0, 0
move_smoothness = 0.2
last_click_time = 0
click_delay = 0.5
last_scroll_time = 0
scroll_delay = 0.5
dragging = False
current_mode = "Mouse"
last_toggle_time = 0
toggle_delay = 1.0

# Main loop
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        frame_height, frame_width, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                # Landmark positions
                index_tip = landmarks[8]
                thumb_tip = landmarks[4]
                pinky_tip = landmarks[20]

                x_index = int(index_tip.x * frame_width)
                y_index = int(index_tip.y * frame_height)
                x_thumb = int(thumb_tip.x * frame_width)
                y_thumb = int(thumb_tip.y * frame_height)
                x_pinky = int(pinky_tip.x * frame_width)
                y_pinky = int(pinky_tip.y * frame_height)

                # Get finger states
                states = fingers_up(landmarks)

                # Mouse Mode: Move mouse with index finger
                if current_mode == "Mouse":
                    screen_x = int(index_tip.x * screen_width)
                    screen_y = int(index_tip.y * screen_height)
                    smooth_x = prev_x + (screen_x - prev_x) * move_smoothness
                    smooth_y = prev_y + (screen_y - prev_y) * move_smoothness
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y

                    # Drag & drop
                    pinch_distance = calculate_distance((x_index, y_index), (x_thumb, y_thumb))
                    if pinch_distance < 20:
                        if not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                    else:
                        if dragging:
                            pyautogui.mouseUp()
                            dragging = False

                    # Left click (thumb + index)
                    if pinch_distance < 20 and (time.time() - last_click_time) > click_delay:
                        pyautogui.click()
                        last_click_time = time.time()

                    # Right click (thumb + pinky)
                    right_click_distance = calculate_distance((x_thumb, y_thumb), (x_pinky, y_pinky))
                    if right_click_distance < 40 and (time.time() - last_click_time) > click_delay:
                        pyautogui.rightClick()
                        last_click_time = time.time()

                    # Scroll with fist
                    tip_ids = [8, 12, 16, 20]
                    pip_ids = [6, 10, 14, 18]
                    is_fist = all(landmarks[tip].y > landmarks[pip].y for tip, pip in zip(tip_ids, pip_ids))
                    if is_fist and (time.time() - last_scroll_time) > scroll_delay:
                        pyautogui.scroll(-40)
                        last_scroll_time = time.time()

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display mode
        cv2.putText(frame, f"Mode: {current_mode}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show
        cv2.imshow("Hand Mouse Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
