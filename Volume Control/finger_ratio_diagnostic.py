"""
Finger Ratio Diagnostic
------------------------
Shows the live (fingertip-to-wrist / knuckle-to-wrist) ratio for each of
the 4 fingers, so you can read real numbers off your own hand and pick
correct FIST_RATIO / PALM_RATIO values for volume.py.

Try: relaxed hand, tight fist, full open palm -- note the ratio ranges.

Controls:
    q  -> quit
"""

import math
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky"]
FINGER_PAIRS = [(8, 5), (12, 9), (16, 13), (20, 17)]
WRIST = 0

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        wrist = landmarks[WRIST]
        wx, wy = wrist.x * w, wrist.y * h

        ratio_values = []

        for i, (tip_idx, base_idx) in enumerate(FINGER_PAIRS):
            tip, base = landmarks[tip_idx], landmarks[base_idx]
            tip_dist = math.hypot(tip.x * w - wx, tip.y * h - wy)
            base_dist = math.hypot(base.x * w - wx, base.y * h - wy)
            ratio = tip_dist / base_dist if base_dist > 0 else 0
            ratio_values.append(ratio)

            average_ratio = sum(ratio_values) / len(ratio_values)
            cv2.putText(
            frame, f"Average: {average_ratio:.2f}", (20, 40 + len(FINGER_PAIRS) * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

    cv2.imshow("Finger Ratio Diagnostic", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
