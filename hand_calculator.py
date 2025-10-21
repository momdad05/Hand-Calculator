import cv2
import numpy as np
import math
import time
import mediapipe as mp

# Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Button class
class Button:
    def __init__(self, pos, w, h, val):
        self.pos, self.w, self.h, self.val = pos, w, h, val

    def draw(self, img, hover=False):
        x, y = self.pos
        if hover:
            color, text_color = (0, 255, 255), (0, 0, 0)
        else:
            color, text_color = (50, 50, 50), (255, 255, 255)

        # Button rectangle
        cv2.rectangle(img, (x, y), (x + self.w, y + self.h), color, cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + self.w, y + self.h), (255, 255, 255), 2)

        # Button text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 1.2, 2
        (tw, th), _ = cv2.getTextSize(self.val, font, font_scale, thickness)
        text_x = x + (self.w - tw) // 2
        text_y = y + (self.h + th) // 2
        cv2.putText(img, self.val, (text_x, text_y), font, font_scale, text_color, thickness)

    def is_hover(self, x, y):
        bx, by = self.pos
        return bx <= x <= bx + self.w and by <= y <= by + self.h


# Calculator button layout
keys = [
    ["7", "8", "9", "+"],
    ["4", "5", "6", "-"],
    ["1", "2", "3", "*"],
    ["C", "0", "=", "/"],
    ["(", ")", "DEL", "."]
]

button_list = [Button((j*100+50, i*100+150), 80, 80, keys[i][j])
               for i in range(len(keys)) for j in range(len(keys[i]))]

expression, last_result = "", ""
last_click, delay = 0, 0.6

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # Calculator window
    calc = np.zeros((700, 600, 3), np.uint8)

    # Last result
    if last_result:
        cv2.putText(calc, f"Last: {last_result}", (60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Current expression
    cv2.rectangle(calc, (50, 70), (500, 130), (0, 0, 0), cv2.FILLED)
    cv2.putText(calc, expression, (60, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

    # Draw buttons
    for b in button_list:
        b.draw(calc)

    hovered = None
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        lm_list = [(int(p.x*w), int(p.y*h)) for p in lm.landmark]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # Thumb and index fingertip
        x1, y1 = lm_list[4]
        x2, y2 = lm_list[8]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        length = math.hypot(x2-x1, y2-y1)

        # Scale to calculator window
        calc_x, calc_y = int((cx/w)*600), int((cy/h)*700)

        # Hover detection
        for b in button_list:
            if b.is_hover(calc_x, calc_y):
                hovered = b
                b.draw(calc, hover=True)
                break

        # Click detection
        if hovered and length < 40 and time.time()-last_click > delay:
            val = hovered.val
            if val == "C":
                expression = ""
            elif val == "=":
                try:
                    last_result = expression
                    expression = str(eval(expression))
                except Exception:
                    expression = "Err"
            elif val == "DEL":
                expression = expression[:-1]
            else:
                expression += val
            last_click = time.time()

    # Blend calculator with webcam
    calc_resized = cv2.resize(calc, (frame.shape[1], frame.shape[0]))
    blended = cv2.addWeighted(frame, 0.6, calc_resized, 0.4, 0)

    cv2.imshow("Virtual Calculator", blended)
    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:  # ESC or q
        break

cap.release()
cv2.destroyAllWindows()
