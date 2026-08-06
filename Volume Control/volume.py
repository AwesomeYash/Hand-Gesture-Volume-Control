"""
Hand Gesture Volume Control
----------------------------
Pinch your thumb and index finger together to lower the volume.
Spread them apart to raise the volume.

Controls:
    q  -> quit the program
"""

import math
import cv2
import mediapipe as mp
import numpy as np

from pycaw.pycaw import AudioUtilities


# ---------------------------------------------------------------------------
# Windows system volume setup (pycaw)
# ---------------------------------------------------------------------------
def get_volume_interface():
    # Newer pycaw versions wrap the device in an AudioDevice object and expose
    # the endpoint volume control directly as a property, rather than requiring
    # a manual COM .Activate() call.
    speakers = AudioUtilities.GetSpeakers()
    volume = speakers.EndpointVolume
    return volume


volume_interface = get_volume_interface()
vol_range = volume_interface.GetVolumeRange()   # (min_dB, max_dB, step)

# NOT USED: The volume range in dB is not used directly, but we store it for reference.
MIN_VOL = vol_range[0]
MAX_VOL = vol_range[1]

# ---------------------------------------------------------------------------
# MediaPipe hand detection setup
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,  # 0 = lite model, much faster, small accuracy tradeoff
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ---------------------------------------------------------------------------
# Distance range calibration
# These are pixel distances between thumb tip and index tip.
# Adjust MIN_DIST / MAX_DIST if it feels too sensitive or not sensitive enough.
# ---------------------------------------------------------------------------
MIN_DIST = 25    # fingers touching -> 0% volume
MAX_DIST = 200   # fingers fully spread -> 100% volume


# Alternate Fucntion to math.hypot() to avoid using sqrt and squares, which can be computationally expensive.
"""
def safe_hypot(x, y):
    # 1. Take absolute values
    ax = abs(x)
    ay = abs(y)
    
    # 2. Identify the larger and smaller number
    if ax < ay:
        ax, ay = ay, ax
        
    # 3. Handle the zero case immediately to avoid DivisionByZero
    if ax == 0.0:
        return 0.0
        
    # 4. Factor out the larger number: ax * sqrt(1 + (ay/ax)^2)
    return ax * ((1.0 + (ay / ax) ** 2) ** 0.5)
"""


def main():
    # CAP_DSHOW is generally faster/more responsive than the default backend
    # on Windows. Lower resolution than before (still plenty for hand
    # tracking) and a buffer size of 1 so we always grab the newest frame
    # instead of processing a backlog of stale ones.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check that it's connected and not in use.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        h, w, _ = frame.shape

        # Color conversion: OpenCV uses BGR internally; MediaPipe expects RGB. Easy thing to forget when combining CV libraries.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)                      

        vol_percent = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark

                # Landmark 4 = thumb tip, Landmark 8 = index finger tip
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]

                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                # Draw markers and connecting line
                cv2.circle(frame, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 12, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

                # Distance between the two fingertips
                distance = math.hypot(x2 - x1, y2 - y1)
                # Alternative way to compute distance without using math.hypot (which uses sqrt and squares):
                # distance = safe_hypot(x2 - x1, y2 - y1)

                # Map distance -> volume percentage (0-100)
                vol_percent = np.interp(distance, [MIN_DIST, MAX_DIST], [0, 100])
                vol_percent = float(np.clip(vol_percent, 0, 100))

                # Map percentage -> the volume scalar pycaw expects (0.0 - 1.0)
                volume_interface.SetMasterVolumeLevelScalar(vol_percent / 100, None)

                # Visual feedback: turn the connecting line green when very close
                if distance < MIN_DIST + 10:
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

        # --- Draw the volume bar UI ---
        bar_x, bar_y, bar_w, bar_h = 50, 150, 35, 300
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 3)

        if vol_percent is not None:
            filled_h = int(np.interp(vol_percent, [0, 100], [bar_h, 0]))
            cv2.rectangle(
                frame,
                (bar_x, bar_y + filled_h),
                (bar_x + bar_w, bar_y + bar_h),
                (0, 255, 0),
                cv2.FILLED,
            )
            cv2.putText(
                frame, f'{int(vol_percent)} %', (bar_x - 10, bar_y + bar_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
            )
        else:
            # Show current system volume when no hand is detected
            current = int(volume_interface.GetMasterVolumeLevelScalar() * 100)
            filled_h = int(np.interp(current, [0, 100], [bar_h, 0]))
            cv2.rectangle(
                frame,
                (bar_x, bar_y + filled_h),
                (bar_x + bar_w, bar_y + bar_h),
                (100, 100, 100),
                cv2.FILLED,
            )
            cv2.putText(
                frame, "No hand", (bar_x - 20, bar_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        cv2.putText(
            frame, "Press 'q' to quit", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        cv2.imshow("Hand Gesture Volume Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()