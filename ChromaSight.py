import cv2
import numpy as np
import pyttsx3
import threading
import time
from collections import Counter, deque

# --- 1. YOUR ORIGINAL DATASET (UNALTERED) ---
color_dataset = {
    "Alice Blue": {"bgr": (255, 248, 240), "desc": "A very pale tint of Blue."},
    "Antique White": {"bgr": (215, 235, 250), "desc": "A warm, aged shade of White."},
    "Aqua": {"bgr": (255, 255, 0), "desc": "A bright, neon Blue-Green."},
    "Aquamarine": {"bgr": (212, 255, 127), "desc": "A soft, minty shade of Green."},
    "Azure": {"bgr": (255, 255, 240), "desc": "A crisp, cool hint of Blue."},
    "Beige": {"bgr": (220, 245, 245), "desc": "A light, sandy shade of Brown."},
    "Bisque": {"bgr": (196, 228, 255), "desc": "A pale, creamy Orange."},
    "Black": {"bgr": (0, 0, 0), "desc": "A solid, deep Black."},
    "Blanched Almond": {"bgr": (205, 235, 255), "desc": "A soft, almond shade of White."},
    "Blue": {"bgr": (255, 0, 0), "desc": "A bright, solid Blue."},
    "Blue Violet": {"bgr": (226, 43, 138), "desc": "A vibrant mix of Blue and Purple."},
    "Brown": {"bgr": (42, 42, 165), "desc": "A solid, earthy Brown."},
    "Burly Wood": {"bgr": (135, 184, 222), "desc": "A medium, tan shade of Brown."},
    "Cadet Blue": {"bgr": (160, 158, 95), "desc": "A dusty, greyish Blue."},
    "Chartreuse": {"bgr": (0, 255, 127), "desc": "A sharp, neon Yellow-Green."},
    "Chocolate": {"bgr": (30, 105, 210), "desc": "A rich, dark shade of Brown."},
    "Coral": {"bgr": (80, 127, 255), "desc": "A bright, pinkish Orange."},
    "Cornflower Blue": {"bgr": (237, 149, 100), "desc": "A soft, medium Blue."},
    "Cornsilk": {"bgr": (220, 248, 255), "desc": "A very light, creamy Yellow."},
    "Crimson": {"bgr": (60, 20, 220), "desc": "A deep, powerful Red."},
    "Cyan": {"bgr": (255, 255, 0), "desc": "A bright, electric Blue."},
    "Dark Blue": {"bgr": (139, 0, 0), "desc": "A deep, navy shade of Blue."},
    "Dark Cyan": {"bgr": (139, 139, 0), "desc": "A dark, muted Blue-Green."},
    "Dark Goldenrod": {"bgr": (11, 134, 184), "desc": "A dark, metallic Yellow-Brown."},
    "Dark Gray": {"bgr": (169, 169, 169), "desc": "A heavy, dark Gray."},
    "Dark Green": {"bgr": (0, 100, 0), "desc": "A very deep, forest Green."},
    "Dark Khaki": {"bgr": (107, 183, 189), "desc": "A dark, olive-toned Yellow."},
    "Dark Magenta": {"bgr": (139, 0, 139), "desc": "A dark, solid Purple."},
    "Dark Olive Green": {"bgr": (47, 107, 85), "desc": "A dark, muted Green."},
    "Dark Orange": {"bgr": (0, 140, 255), "desc": "A deep, burnt Orange."},
    "Dark Orchid": {"bgr": (204, 50, 153), "desc": "A deep, vibrant Purple."},
    "Dark Red": {"bgr": (0, 0, 139), "desc": "A dark, wine-like Red."},
    "Dark Salmon": {"bgr": (122, 150, 233), "desc": "A deep, tanned Pink."},
    "Dark Sea Green": {"bgr": (143, 188, 143), "desc": "A muted, mossy Green."},
    "Dark Slate Blue": {"bgr": (139, 61, 72), "desc": "A deep, stony Purple-Blue."},
    "Dark Slate Gray": {"bgr": (79, 79, 47), "desc": "A very dark, cold Gray."},
    "Dark Turquoise": {"bgr": (209, 206, 0), "desc": "A dark, oceanic Blue-Green."},
    "Dark Violet": {"bgr": (211, 0, 148), "desc": "A deep, royal Purple."},
    "Deep Pink": {"bgr": (147, 20, 255), "desc": "A strong, dark shade of Pink."},
    "Deep Sky Blue": {"bgr": (255, 191, 0), "desc": "A bright, intense Blue."},
    "Dim Gray": {"bgr": (105, 105, 105), "desc": "A dark, heavy Gray."},
    "Dodger Blue": {"bgr": (255, 144, 30), "desc": "A vivid, sporty Blue."},
    "Fire Brick": {"bgr": (34, 34, 178), "desc": "A dark, brownish Red."},
    "Floral White": {"bgr": (240, 250, 255), "desc": "A warm, floral White."},
    "Forest Green": {"bgr": (34, 139, 34), "desc": "A solid, dark Green."},
    "Fuchsia": {"bgr": (255, 0, 255), "desc": "A bright, neon Purple."},
    "Gainsboro": {"bgr": (220, 220, 220), "desc": "A very light, neutral Gray."},
    "Ghost White": {"bgr": (255, 248, 248), "desc": "A crisp, cool White."},
    "Gold": {"bgr": (0, 215, 255), "desc": "A metallic, shiny Yellow."},
    "Goldenrod": {"bgr": (32, 165, 218), "desc": "A warm, earthy Yellow."},
    "Gray": {"bgr": (128, 128, 128), "desc": "A neutral, solid Gray."},
    "Green": {"bgr": (0, 128, 0), "desc": "A solid, primary Green."},
    "Green Yellow": {"bgr": (47, 255, 173), "desc": "A bright neon Yellow-Green."},
    "Honeydew": {"bgr": (240, 255, 240), "desc": "A very pale, minty White."},
    "Hot Pink": {"bgr": (180, 105, 255), "desc": "A loud, vibrant Pink."},
    "Indian Red": {"bgr": (92, 92, 205), "desc": "A muted, earthy Red."},
    "Indigo": {"bgr": (130, 0, 75), "desc": "A very dark, inky Purple."},
    "Ivory": {"bgr": (240, 255, 240), "desc": "A bright, creamy White."},
    "Khaki": {"bgr": (140, 230, 240), "desc": "A light, tan Yellow."},
    "Lavender": {"bgr": (250, 230, 230), "desc": "A soft, floral Purple."},
    "Lavender Blush": {"bgr": (245, 240, 255), "desc": "A pale, pinkish White."},
    "Lawn Green": {"bgr": (0, 252, 124), "desc": "A bright, vibrant Green."},
    "Lemon Chiffon": {"bgr": (205, 250, 255), "desc": "A soft, lemon Yellow."},
    "Light Blue": {"bgr": (230, 216, 173), "desc": "A soft, pale Blue."},
    "Light Coral": {"bgr": (128, 128, 240), "desc": "A soft, pinkish Red."},
    "Light Cyan": {"bgr": (255, 255, 224), "desc": "A very pale Blue-Green."},
    "Light Goldenrod Yellow": {"bgr": (210, 250, 250), "desc": "A pale, sunny Yellow."},
    "Light Gray": {"bgr": (211, 211, 211), "desc": "A soft, light Gray."},
    "Light Green": {"bgr": (144, 238, 144), "desc": "A soft, pastel Green."},
    "Light Pink": {"bgr": (193, 182, 255), "desc": "A delicate, pale Pink."},
    "Light Salmon": {"bgr": (122, 160, 255), "desc": "A light, pinkish Orange."},
    "Light Sea Green": {"bgr": (170, 178, 32), "desc": "A bright, minty Green."},
    "Light Sky Blue": {"bgr": (250, 206, 135), "desc": "A soft, airy Blue."},
    "Light Slate Gray": {"bgr": (153, 136, 119), "desc": "A muted, blue-toned Gray."},
    "Light Steel Blue": {"bgr": (222, 196, 176), "desc": "A cold, metallic Blue."},
    "Light Yellow": {"bgr": (224, 255, 255), "desc": "A very pale, bright Yellow."},
    "Lime": {"bgr": (0, 255, 0), "desc": "A pure, neon Green."},
    "Lime Green": {"bgr": (50, 205, 50), "desc": "A solid, grassy Green."},
    "Linen": {"bgr": (230, 240, 250), "desc": "A very pale, warm White."},
    "Magenta": {"bgr": (255, 0, 255), "desc": "A vivid, reddish Purple."},
    "Maroon": {"bgr": (0, 0, 128), "desc": "A dark, brownish Red."},
    "Medium Aquamarine": {"bgr": (170, 205, 102), "desc": "A medium, minty Green."},
    "Medium Blue": {"bgr": (205, 0, 0), "desc": "A solid, medium Blue."},
    "Medium Orchid": {"bgr": (211, 85, 186), "desc": "A medium, floral Purple."},
    "Medium Purple": {"bgr": (219, 112, 147), "desc": "A soft, balanced Purple."},
    "Medium Sea Green": {"bgr": (113, 179, 60), "desc": "A medium, oceanic Green."},
    "Medium Slate Blue": {"bgr": (238, 104, 123), "desc": "A bright Purple-Blue."},
    "Medium Spring Green": {"bgr": (154, 250, 0), "desc": "A bright, medium Green."},
    "Medium Turquoise": {"bgr": (204, 209, 72), "desc": "A medium, tropical Blue-Green."},
    "Medium Violet Red": {"bgr": (133, 21, 199), "desc": "A vibrant, pinkish Red."},
    "Midnight Blue": {"bgr": (112, 25, 25), "desc": "A dark, near-black Blue."},
    "Mint Cream": {"bgr": (250, 255, 245), "desc": "A very pale, fresh White."},
    "Misty Rose": {"bgr": (225, 228, 255), "desc": "A very pale, dusty Pink."},
    "Moccasin": {"bgr": (181, 228, 255), "desc": "A warm, creamy Orange."},
    "Navajo White": {"bgr": (173, 222, 255), "desc": "A warm, sandy White."},
    "Navy": {"bgr": (128, 0, 0), "desc": "A solid, dark Blue."},
    "Old Lace": {"bgr": (230, 245, 253), "desc": "A warm, antique White."},
    "Olive": {"bgr": (0, 128, 128), "desc": "A muted, brownish Green."},
    "Olive Drab": {"bgr": (35, 142, 107), "desc": "A dark, olive Green."},
    "Orange": {"bgr": (0, 165, 255), "desc": "A bright, solid Orange."},
    "Orange Red": {"bgr": (0, 69, 255), "desc": "A glowing, reddish Orange."},
    "Orchid": {"bgr": (214, 112, 218), "desc": "A bright, flowery Purple."},
    "Pale Goldenrod": {"bgr": (170, 232, 238), "desc": "A pale, creamy Yellow."},
    "Pale Green": {"bgr": (152, 251, 152), "desc": "A soft, pale Green."},
    "Pale Turquoise": {"bgr": (238, 238, 175), "desc": "A soft, pale Blue-Green."},
    "Pale Violet Red": {"bgr": (147, 112, 219), "desc": "A dusty, muted Pink."},
    "Papaya Whip": {"bgr": (213, 239, 255), "desc": "A very pale, creamy Orange."},
    "Peach Puff": {"bgr": (185, 218, 255), "desc": "A soft, light Orange."},
    "Peru": {"bgr": (63, 133, 205), "desc": "A warm, sandy Brown."},
    "Pink": {"bgr": (203, 192, 255), "desc": "A soft, classic Pink."},
    "Plum": {"bgr": (221, 160, 221), "desc": "A dusty, fruit-like Purple."},
    "Powder Blue": {"bgr": (230, 224, 176), "desc": "A soft, pale Blue."},
    "Purple": {"bgr": (128, 0, 128), "desc": "A solid, primary Purple."},
    "Red": {"bgr": (0, 0, 255), "desc": "A solid, primary Red."},
    "Rosy Brown": {"bgr": (143, 143, 188), "desc": "A muted, pinkish Brown."},
    "Royal Blue": {"bgr": (225, 105, 41), "desc": "A majestic, deep Blue."},
    "Saddle Brown": {"bgr": (19, 69, 139), "desc": "A deep, leathery Brown."},
    "Salmon": {"bgr": (114, 128, 250), "desc": "A warm, pinkish Orange."},
    "Sandy Brown": {"bgr": (96, 164, 244), "desc": "A light, sandy Brown."},
    "Sea Green": {"bgr": (87, 139, 46), "desc": "A dark, teal-like Green."},
    "Sea Shell": {"bgr": (238, 245, 255), "desc": "A very pale, pinkish White."},
    "Sienna": {"bgr": (45, 82, 160), "desc": "An earthy, reddish Brown."},
    "Silver": {"bgr": (192, 192, 192), "desc": "A light, shiny Gray."},
    "Sky Blue": {"bgr": (235, 206, 135), "desc": "A clear, light Blue."},
    "Slate Blue": {"bgr": (205, 90, 106), "desc": "A dusty, greyish Purple."},
    "Slate Gray": {"bgr": (144, 128, 112), "desc": "A muted, blue-toned Gray."},
    "Snow": {"bgr": (250, 250, 255), "desc": "A pure, bright White."},
    "Spring Green": {"bgr": (127, 255, 0), "desc": "A bright, neon Green."},
    "Steel Blue": {"bgr": (180, 130, 70), "desc": "A metallic, cold Blue."},
    "Tan": {"bgr": (140, 180, 210), "desc": "A light, neutral Brown."},
    "Teal": {"bgr": (128, 128, 0), "desc": "A deep, dark Blue-Green."},
    "Thistle": {"bgr": (216, 191, 216), "desc": "A pale, greyish Purple."},
    "Tomato": {"bgr": (71, 99, 255), "desc": "A bright, reddish Orange."},
    "Turquoise": {"bgr": (208, 224, 64), "desc": "A bright, tropical Blue-Green."},
    "Violet": {"bgr": (238, 130, 238), "desc": "A bright, solid Purple."},
    "Wheat": {"bgr": (179, 222, 245), "desc": "A soft, light Brown."},
    "White": {"bgr": (255, 255, 255), "desc": "A pure, solid White."},
    "White Smoke": {"bgr": (245, 245, 245), "desc": "A very pale, smoky White."},
    "Yellow": {"bgr": (0, 255, 255), "desc": "A solid, primary Yellow."},
    "Yellow Green": {"bgr": (50, 205, 154), "desc": "A solid, grassy Green."}
}

# --- 2. ENGINE INITIALIZATION ---
def nothing(x): pass
cv2.namedWindow("Assistive Spectacle Engine")
cv2.createTrackbar("Voice Delay", "Assistive Spectacle Engine", 5, 30, nothing)
cv2.createTrackbar("Smoothing", "Assistive Spectacle Engine", 15, 50, nothing)
cv2.createTrackbar("Voice Speed", "Assistive Spectacle Engine", 160, 300, nothing)

hist_l = deque(maxlen=8)
hist_r = deque(maxlen=8)

# KNN Prep
knn_names = list(color_dataset.keys())
bgr_raw = np.array([v["bgr"] for v in color_dataset.values()], dtype=np.uint8).reshape(-1, 1, 3)
lab_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(lab_raw, cv2.ml.ROW_SAMPLE, np.arange(len(knn_names)).astype(np.float32))

speech_lock = threading.Lock()
last_speak_time = time.time()

def speak_worker(text, speed):
    if speech_lock.locked(): return
    with speech_lock:
        engine = pyttsx3.init()
        engine.setProperty('rate', speed)
        engine.say(text)
        engine.runAndWait()

def apply_pattern(mask_roi, pattern_type):
    h_r, w_r = mask_roi.shape[:2]
    if pattern_type == "stripes":
        for i in range(0, h_r + w_r, 10):
            cv2.line(mask_roi, (i, 0), (0, i), (255, 255, 255), 1)
    elif pattern_type == "dots":
        for i in range(0, h_r, 10):
            for j in range(0, w_r, 10):
                cv2.circle(mask_roi, (j, i), 1, (255, 255, 255), -1)

def get_color_lab(roi):
    avg_bgr = np.mean(roi, axis=(0, 1)).astype(np.uint8)
    avg_lab = cv2.cvtColor(np.uint8([[avg_bgr]]), cv2.COLOR_BGR2LAB)[0][0]
    px_lab_float = avg_lab.reshape(1, 3).astype(np.float32)
    _, res, _, _ = knn.findNearest(px_lab_float, k=1)
    return knn_names[int(res[0][0])], avg_lab, avg_bgr

# --- 3. MAIN LOOP ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    display = frame.copy()

    delay_val = cv2.getTrackbarPos("Voice Delay", "Assistive Spectacle Engine")
    smooth_val = max(1, cv2.getTrackbarPos("Smoothing", "Assistive Spectacle Engine"))
    speed_val = max(50, cv2.getTrackbarPos("Voice Speed", "Assistive Spectacle Engine"))

    if 'buffer_l' not in locals() or len(buffer_l) != smooth_val:
        buffer_l = deque(maxlen=smooth_val)
        buffer_r = deque(maxlen=smooth_val)

    # Wide ROIs
    roi_l_rect = (h//4, 3*h//4, w//8, 3*w//8)
    roi_r_rect = (h//4, 3*h//4, 5*w//8, 7*w//8)
    
    roi_l = frame[roi_l_rect[0]:roi_l_rect[1], roi_l_rect[2]:roi_l_rect[3]]
    roi_r = frame[roi_r_rect[0]:roi_r_rect[1], roi_r_rect[2]:roi_r_rect[3]]

    name_l, lab_l, bgr_l = get_color_lab(roi_l)
    name_r, lab_r, bgr_r = get_color_lab(roi_r)

    buffer_l.append(name_l)
    buffer_r.append(name_r)
    stable_l = Counter(buffer_l).most_common(1)[0][0]
    stable_r = Counter(buffer_r).most_common(1)[0][0]

    # Delta E (Perceptual Distance) Logic
    dist = np.linalg.norm(lab_l.astype(np.float32) - lab_r.astype(np.float32))
    match_status = "DISTINCT" if dist > 25 else "SIMILAR"

    # Voice & History Trigger
    now = time.time()
    if delay_val > 0 and (now - last_speak_time >= delay_val):
        msg = f"Left is {stable_l}. Right is {stable_r}. These colors are {match_status}."
        threading.Thread(target=speak_worker, args=(msg, speed_val), daemon=True).start()
        hist_l.append(color_dataset[stable_l]['bgr'])
        hist_r.append(color_dataset[stable_r]['bgr'])
        last_speak_time = now

    # --- UI RENDERING ---
    # 1. Patterns (Hash overlays for redundant identification)
    pattern_overlay = np.zeros_like(roi_l)
    apply_pattern(pattern_overlay, "stripes")
    display[roi_l_rect[0]:roi_l_rect[1], roi_l_rect[2]:roi_l_rect[3]] = cv2.addWeighted(roi_l, 0.8, pattern_overlay, 0.2, 0)
    
    pattern_overlay_r = np.zeros_like(roi_r)
    apply_pattern(pattern_overlay_r, "dots")
    display[roi_r_rect[0]:roi_r_rect[1], roi_r_rect[2]:roi_r_rect[3]] = cv2.addWeighted(roi_r, 0.8, pattern_overlay_r, 0.2, 0)

    # 2. Divider & Perceptual Match Indicator
    cv2.line(display, (w//2, 0), (w//2, h), (0, 255, 0), 1)
    match_color = (0, 255, 0) if match_status == "DISTINCT" else (0, 255, 255)
    cv2.putText(display, f"CONTRAST: {match_status}", (w//2 - 60, 30), 0, 0.5, match_color, 2)

    # 3. History Swatches
    for i, bgr in enumerate(hist_l):
        y = 20 + (i * 50)
        cv2.rectangle(display, (10, y), (50, y+40), [int(c) for c in bgr], -1)
        cv2.rectangle(display, (10, y), (50, y+40), (255, 255, 255), 1)
    for i, bgr in enumerate(hist_r):
        y = 20 + (i * 50)
        cv2.rectangle(display, (w-60, y), (w-20, y+40), [int(c) for c in bgr], -1)
        cv2.rectangle(display, (w-60, y), (w-20, y+40), (255, 255, 255), 1)

    # 4. Green Written Descriptions (Locked to bottom)
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

    sat_l = "Vivid" if lab_l[1]**2 + lab_l[2]**2 > 1000 else "Muted"
    sat_r = "Vivid" if lab_r[1]**2 + lab_r[2]**2 > 1000 else "Muted"

    cv2.putText(display, f"L: {stable_l} ({sat_l})", (70, h-65), 0, 0.6, (0, 255, 0), 2)
    cv2.putText(display, color_dataset[stable_l]['desc'], (70, h-35), 0, 0.45, (0, 200, 0), 1)
    cv2.putText(display, f"R: {stable_r} ({sat_r})", (w//2 + 20, h-65), 0, 0.6, (0, 255, 0), 2)
    cv2.putText(display, color_dataset[stable_r]['desc'], (w//2 + 20, h-35), 0, 0.45, (0, 200, 0), 1)

    cv2.imshow("Assistive Spectacle Engine", display)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
