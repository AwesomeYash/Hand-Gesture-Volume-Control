"""
Radial (Rotation) Volume Control
----------------------------------
Pinch your thumb and index finger together and HOLD to grip the dial.
While gripped, rotate your hand around your palm -- clockwise raises
volume, counterclockwise lowers it. Release the pinch to let go.

This uses relative angle change (not absolute position), so there's no
fixed "zero" position and no wraparound/orientation problem: you can
grip, rotate, release, re-grip anywhere, and keep turning like an
actual knob.

Controls:
    q  -> quit
"""

import math
import cv2
import mediapipe as mp
import numpy as np

from pycaw.pycaw import AudioUtilities


def get_volume_interface():
    speakers = AudioUtilities.GetSpeakers()
    return speakers.EndpointVolume


volume_interface = get_volume_interface()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

GRIP_THRESHOLD = 40       # thumb-index pixel distance below this -> gripped
ROTATION_SENSITIVITY = 0.4  # degrees of rotation -> % volume change


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    is_gripped = False
    last_angle = None
    vol_percent = volume_interface.GetMasterVolumeLevelScalar() * 100

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # Palm anchor: midpoint of wrist and middle-finger knuckle
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            anchor_x = (wrist.x + middle_mcp.x) / 2 * w
            anchor_y = (wrist.y + middle_mcp.y) / 2 * h

            # Cursor: middle fingertip
            cursor = landmarks[12]
            cursor_x, cursor_y = cursor.x * w, cursor.y * h

            # Grip check: thumb-index pinch distance
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            pinch_dist = math.hypot(
                (thumb_tip.x - index_tip.x) * w,
                (thumb_tip.y - index_tip.y) * h,
            )
            currently_gripped = pinch_dist < GRIP_THRESHOLD

            angle = math.degrees(math.atan2(cursor_y - anchor_y, cursor_x - anchor_x))

            if currently_gripped:
                if not is_gripped:
                    # Just gripped this frame -- record starting angle,
                    # don't apply a volume change yet (no prior angle to
                    # compare against).
                    last_angle = angle
                else:
                    delta = angle - last_angle
                    delta = (delta + 180) % 360 - 180  # normalize to [-180, 180]
                    vol_percent = np.clip(vol_percent + delta * ROTATION_SENSITIVITY, 0, 100)
                    volume_interface.SetMasterVolumeLevelScalar(vol_percent / 100, None)
                    last_angle = angle
                is_gripped = True

                cv2.line(frame, (int(anchor_x), int(anchor_y)), (int(cursor_x), int(cursor_y)), (0, 255, 0), 3)
                cv2.circle(frame, (int(cursor_x), int(cursor_y)), 10, (0, 255, 0), cv2.FILLED)
            else:
                is_gripped = False
                last_angle = None

            cv2.putText(
                frame, "GRIPPED" if is_gripped else "Pinch to grip",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if is_gripped else (200, 200, 200), 2,
            )

        cv2.putText(
            frame, f"Volume: {int(vol_percent)}%", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, "Press 'q' to quit", (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        cv2.imshow("Radial Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()