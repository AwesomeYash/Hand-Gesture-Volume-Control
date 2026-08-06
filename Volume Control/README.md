# Hand Gesture Volume Control (Windows)

Control your PC's system volume by pinching your thumb and index finger
in front of your webcam. Move them apart to raise volume, bring them
together to lower it.

## How it works

1. **OpenCV** captures live video from your webcam.
2. **MediaPipe Hands** detects 21 landmark points on your hand in each frame.
3. The script measures the pixel distance between your **thumb tip** (landmark 4)
   and **index fingertip** (landmark 8).
4. That distance is mapped to a 0–100% range and sent to Windows via **pycaw**,
   which talks to the Windows Core Audio API to set the master volume.

## Setup

1. Make sure you have **Python 3.9–3.11** installed (MediaPipe doesn't yet
   support the very latest Python versions on all platforms — 3.10 is a safe bet).
   Check with:
   ```
   python --version
   ```

2. (Recommended) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the program:
   ```
   python volume_control.py
   ```

5. A window will open showing your webcam feed with hand landmarks drawn on it,
   plus a volume bar on the left. Pinch/spread your thumb and index finger to
   control the volume. Press **q** to quit.

## Troubleshooting

- **"Could not open webcam"** — close any other app using the camera (Zoom, Teams,
  another instance of this script), or try changing `cv2.VideoCapture(0)` to
  `cv2.VideoCapture(1)` in `volume_control.py` if you have multiple cameras.
- **Volume feels too sensitive / not sensitive enough** — tweak the `MIN_DIST`
  and `MAX_DIST` constants near the top of `volume_control.py`. Larger `MAX_DIST`
  means you need to spread your fingers further apart to reach 100%.
- **ImportError on pycaw/comtypes** — these are Windows-only libraries; this
  project will not work on macOS or Linux without swapping out the volume-control
  section.
- **MediaPipe install fails** — this usually means your Python version is too new.
  Try Python 3.10 or 3.11 specifically.

## Next steps / ideas to extend this

- Add a second gesture (e.g., a fist) to **mute/unmute**.
- Use multiple fingers to control other things — brightness, media play/pause,
  scrolling, or mouse movement.
- Add a smoothing filter (moving average) on the distance value so the volume
  doesn't jitter as much.
- Support `max_num_hands=2` and use the second hand for a different control.
