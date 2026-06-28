import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from pynput.keyboard import Controller, Key
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions as MpBaseOptions

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)

CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 540
SMOOTHING = 5
CLICK_DISTANCE = 35
PRESS_DISTANCE = 35
CLICK_COOLDOWN = 0.4
PRESS_COOLDOWN = 0.6
SCROLL_SENSITIVITY = 4
FRAME_MARGIN = 100
TRAIL_LENGTH = 14         
RIPPLE_DURATION = 0.35     
pyautogui.FAILSAFE = False
keyboard_ctrl = Controller()

KEY_ROWS = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "BACKSPACE", "ENTER"],
]


COLOR_BG_PANEL = (28, 22, 36)        
COLOR_KEY_IDLE = (54, 42, 68)
COLOR_KEY_HOVER = (255, 184, 60)      
COLOR_KEY_PRESS = (90, 230, 140)       
COLOR_TEXT = (240, 240, 245)
COLOR_ACCENT = (130, 90, 240)           
COLOR_TRAIL = (255, 184, 60)
COLOR_SCROLL = (90, 220, 255)
COLOR_LEFT_CLICK = (90, 230, 140)
COLOR_RIGHT_CLICK = (255, 90, 110)


THUMB_TIP = 4
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18


def finger_is_up(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def build_keys(frame_w, frame_h):
    keys = []
    key_h = 55
    top_margin = 50
    gap = 4
    for row_idx, row in enumerate(KEY_ROWS):
        if row_idx < 4:
            key_w = frame_w // len(row)
            for col_idx, label in enumerate(row):
                x = col_idx * key_w
                y = top_margin + row_idx * key_h
                keys.append({"label": label, "rect": (x + gap, y + gap, key_w - 2 * gap, key_h - 2 * gap)})
        else:
            widths = [frame_w // 2, frame_w // 4, frame_w // 4]
            x = 0
            y = top_margin + 4 * key_h
            for label, wd in zip(row, widths):
                keys.append({"label": label, "rect": (x + gap, y + gap, wd - 2 * gap, key_h - 2 * gap)})
                x += wd
    return keys, top_margin + 5 * key_h


def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def press_key(label, typed_text):
    if label == "SPACE":
        keyboard_ctrl.press(" ")
        keyboard_ctrl.release(" ")
        typed_text += " "
    elif label == "BACKSPACE":
        keyboard_ctrl.press(Key.backspace)
        keyboard_ctrl.release(Key.backspace)
        typed_text = typed_text[:-1]
    elif label == "ENTER":
        keyboard_ctrl.press(Key.enter)
        keyboard_ctrl.release(Key.enter)
        typed_text += "\n"
    else:
        keyboard_ctrl.press(label.lower())
        keyboard_ctrl.release(label.lower())
        typed_text += label.lower()
    return typed_text


def draw_rounded_rect(img, top_left, bottom_right, color, radius=14, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 0:
        r = 0

    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_glow_circle(img, center, radius, color, intensity=0.5):
    glow_layer = img.copy()
    for i in range(3, 0, -1):
        alpha = intensity * (i / 3) * 0.35
        r = int(radius * (1 + i * 0.5))
        cv2.circle(glow_layer, center, r, color, -1)
        img[:] = cv2.addWeighted(glow_layer, alpha, img, 1 - alpha, 0)


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


ensure_model_downloaded()

HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode

landmarker_options = HandLandmarkerOptions(
    base_options=MpBaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
landmarker = HandLandmarker.create_from_options(landmarker_options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),
]


def draw_hand_skeleton(frame, landmarks_px, color_dots=(130, 90, 240), color_lines=(200, 200, 255)):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_px[a], landmarks_px[b], color_lines, 1, cv2.LINE_AA)
    for pt in landmarks_px:
        cv2.circle(frame, pt, 3, color_dots, -1, cv2.LINE_AA)


screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

keys, keyboard_height = build_keys(FRAME_W, FRAME_H)


prev_x, prev_y = 0, 0
is_dragging = False
last_left_click_time = 0
last_right_click_time = 0
last_scroll_y = None

keyboard_open = False
last_press_time = 0
last_pressed_label = None
typed_text = ""

cursor_trail = deque(maxlen=TRAIL_LENGTH)   
ripples = []                                 
mode_transition_time = time.time()            
frame_timestamp_ms = 0                         

print("Animated virtual mouse + keyboard running.")
print("  'k' = toggle keyboard overlay | 'c' = clear typed text | 'q' = quit")

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from webcam. Check CAM_INDEX.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame = cv2.convertScaleAbs(frame, alpha=0.85, beta=-10)

    overlay = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_timestamp_ms += 33 
    results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    hover_label = None
    index_px = None
    now = time.time()

    if results.hand_landmarks:
        lm = results.hand_landmarks[0]  
        landmarks_px = [(int(p.x * w), int(p.y * h)) for p in lm]
        draw_hand_skeleton(frame, landmarks_px, color_dots=COLOR_ACCENT, color_lines=(200, 200, 255))

        index_px = (int(lm[INDEX_TIP].x * w), int(lm[INDEX_TIP].y * h))
        thumb_px = (int(lm[THUMB_TIP].x * w), int(lm[THUMB_TIP].y * h))
        middle_px = (int(lm[MIDDLE_TIP].x * w), int(lm[MIDDLE_TIP].y * h))

        thumb_index_dist = math.hypot(thumb_px[0] - index_px[0], thumb_px[1] - index_px[1])
        thumb_middle_dist = math.hypot(thumb_px[0] - middle_px[0], thumb_px[1] - middle_px[1])

        if keyboard_open:
            for key in keys:
                if point_in_rect(index_px[0], index_px[1], key["rect"]):
                    hover_label = key["label"]
                    break

            if hover_label and thumb_index_dist < PRESS_DISTANCE:
                if now - last_press_time > PRESS_COOLDOWN:
                    typed_text = press_key(hover_label, typed_text)
                    last_press_time = now
                    last_pressed_label = hover_label
                    for key in keys:
                        if key["label"] == hover_label:
                            kx, ky, kw, kh = key["rect"]
                            ripples.append({
                                "center": (kx + kw // 2, ky + kh // 2),
                                "start": now,
                                "max_r": max(kw, kh),
                            })
                            break

            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False

        else:
            index_up = finger_is_up(lm, INDEX_TIP, INDEX_PIP)
            middle_up = finger_is_up(lm, MIDDLE_TIP, MIDDLE_PIP)
            ring_up = finger_is_up(lm, RING_TIP, RING_PIP)
            pinky_up = finger_is_up(lm, PINKY_TIP, PINKY_PIP)

            if index_up and middle_up and not ring_up and not pinky_up:
                avg_y = (index_px[1] + middle_px[1]) / 2
                if last_scroll_y is not None:
                    delta = last_scroll_y - avg_y
                    if abs(delta) > 2:
                        pyautogui.scroll(int(delta * SCROLL_SENSITIVITY))
                last_scroll_y = avg_y
                cv2.putText(frame, "SCROLL", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_SCROLL, 2, cv2.LINE_AA)
            else:
                last_scroll_y = None

                if index_up:
                    target_x = np.interp(index_px[0], (FRAME_MARGIN, w - FRAME_MARGIN), (0, screen_w))
                    target_y = np.interp(index_px[1], (FRAME_MARGIN, h - FRAME_MARGIN), (0, screen_h))
                    curr_x = prev_x + (target_x - prev_x) / SMOOTHING
                    curr_y = prev_y + (target_y - prev_y) / SMOOTHING
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                    cursor_trail.append(index_px)

                if thumb_index_dist < CLICK_DISTANCE:
                    if not is_dragging and now - last_left_click_time > CLICK_COOLDOWN:
                        pyautogui.mouseDown()
                        is_dragging = True
                        last_left_click_time = now
                    cv2.putText(frame, "LEFT CLICK / DRAG", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_LEFT_CLICK, 2, cv2.LINE_AA)
                    draw_glow_circle(frame, index_px, 18, COLOR_LEFT_CLICK, intensity=0.6)
                else:
                    if is_dragging:
                        pyautogui.mouseUp()
                        is_dragging = False

                if thumb_middle_dist < CLICK_DISTANCE and now - last_right_click_time > CLICK_COOLDOWN:
                    pyautogui.click(button="right")
                    last_right_click_time = now

                if now - last_right_click_time < 0.25:
                    cv2.putText(frame, "RIGHT CLICK", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_RIGHT_CLICK, 2, cv2.LINE_AA)
                    draw_glow_circle(frame, middle_px, 18, COLOR_RIGHT_CLICK, intensity=0.6)

    if not keyboard_open and len(cursor_trail) > 1:
        pts = list(cursor_trail)
        for i in range(1, len(pts)):
            t = i / len(pts)
            color = lerp_color((40, 30, 50), COLOR_TRAIL, t)
            thickness = max(1, int(t * 6))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
        if index_px and not keyboard_open:
            draw_glow_circle(frame, index_px, 10, COLOR_TRAIL, intensity=0.5)
            cv2.circle(frame, index_px, 6, COLOR_TRAIL, -1, cv2.LINE_AA)
    if keyboard_open:
        panel = overlay.copy()
        draw_rounded_rect(panel, (0, 0), (w, keyboard_height + 10), COLOR_BG_PANEL, radius=0)

        for key in keys:
            x, y, kw, kh = key["rect"]
            label = key["label"]
            is_hover = label == hover_label
            time_since_press = now - last_press_time
            is_pressed = label == last_pressed_label and time_since_press < 0.18

            if is_pressed:
                color = COLOR_KEY_PRESS
            elif is_hover:
                color = COLOR_KEY_HOVER
            else:
                color = COLOR_KEY_IDLE

            draw_rounded_rect(panel, (x, y), (x + kw, y + kh), color, radius=10)

            if is_hover:
                draw_glow_circle(panel, (x + kw // 2, y + kh // 2), max(kw, kh) // 2, COLOR_KEY_HOVER, intensity=0.35)

            font_scale = 0.42 if len(label) > 3 else 0.75
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            tx = x + (kw - text_size[0]) // 2
            ty = y + (kh + text_size[1]) // 2
            text_color = (20, 20, 20) if (is_hover or is_pressed) else COLOR_TEXT
            cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

        frame = cv2.addWeighted(panel, 0.88, frame, 0.12, 0)
        still_active = []
        for r in ripples:
            age = now - r["start"]
            if age < RIPPLE_DURATION:
                progress = age / RIPPLE_DURATION
                radius = int(r["max_r"] * 0.5 * progress + 8)
                alpha = 1 - progress
                ripple_layer = frame.copy()
                cv2.circle(ripple_layer, r["center"], radius, COLOR_KEY_PRESS, 3, cv2.LINE_AA)
                frame[:] = cv2.addWeighted(ripple_layer, alpha, frame, 1 - alpha, 0)
                still_active.append(r)
        ripples = still_active
        bar_top = h - 44
        draw_rounded_rect(frame, (8, bar_top), (w - 8, h - 8), (18, 14, 24), radius=10)
        display_text = typed_text[-60:] if typed_text else "Start typing..."
        text_color = COLOR_TEXT if typed_text else (120, 110, 130)
        cv2.putText(frame, display_text, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)
    banner_age = now - mode_transition_time
    banner_alpha = max(0, 1 - banner_age / 0.6) if banner_age < 0.6 else 0
    mode_text = "KEYBOARD MODE — press 'k' to close" if keyboard_open else "MOUSE MODE — press 'k' for keyboard"
    banner_color = COLOR_KEY_HOVER if keyboard_open else COLOR_ACCENT

    text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    if not keyboard_open:
        draw_rounded_rect(frame, (4, 4), (text_size[0] + 24, 34), (20, 16, 28), radius=8)
    cv2.putText(frame, mode_text, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2, cv2.LINE_AA)

    if banner_alpha > 0:
        flash_layer = frame.copy()
        cv2.rectangle(flash_layer, (0, 0), (w, h), banner_color, 6)
        frame[:] = cv2.addWeighted(flash_layer, banner_alpha * 0.5, frame, 1 - banner_alpha * 0.5, 0)

    cv2.putText(frame, "q: quit", (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    cv2.imshow("Virtual Mouse + Keyboard", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("q"):
        break
    elif key_pressed == ord("k"):
        keyboard_open = not keyboard_open
        mode_transition_time = time.time()
        cursor_trail.clear()
        if is_dragging:
            pyautogui.mouseUp()
            is_dragging = False
    elif key_pressed == ord("c"):
        typed_text = ""

cap.release()
cv2.destroyAllWindows()
landmarker.close()