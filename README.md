# Facial Expression Control for TurtleBot 4

A ROS 2 package that recognises the operator's facial expression from a webcam
and drives a TurtleBot 4 along a simple trajectory for each expression.
The camera view and the recognition result are shown in real time.

**Platform:** Ubuntu 24.04 · ROS 2 Jazzy · Gazebo Harmonic · Python 3.12

| Expression | Robot action | Command |
|---|---|---|
| Happy | Curve to the left | v = 0.15 m/s, ω = +0.5 rad/s (radius 0.30 m), while the expression is held |
| Sad | Curve to the right | v = 0.15 m/s, ω = −0.5 rad/s, while the expression is held |
| Neutral | Stop | v = 0, ω = 0 |
| Angry | Back off slightly | v = −0.10 m/s until odometry reports 0.15 m |
| Surprise | One full turn in place | ω = 1.0 rad/s until odometry reports 2π rad |
| Fear | Move forward slightly | v = +0.10 m/s until odometry reports 0.15 m |

## How it works

```
webcam ─► emotion_detector_node ── /emotion/label (10 Hz) ──► emotion_motion_node ── /cmd_vel (20 Hz) ──► TurtleBot 4
            │ face detection + expression model                 │ motion primitives        ◄── /odom ──┘
            │ temporal filter (EMA + hold time)                 │ odometry feedback
            └─ live window ◄──────────────── /emotion/action ───┘
```

**Recognition (`emotion_detector_node`).** Frames are captured on the main
thread and classified on a worker thread that always takes the newest frame,
so a slow model lowers the inference rate, not the video frame rate. Backends
share one interface:

| Backend | Model | Notes |
|---|---|---|
| `hsemotion` (default) | EfficientNet-B0 trained on AffectNet, ONNX | Most robust on a real webcam, ~10 ms per face on CPU |
| `deepface` | DeepFace emotion CNN (FER2013, 48×48) | Requires TensorFlow |
| `ferplus` | FER+ CNN (ONNX model zoo) via OpenCV DNN | No TensorFlow needed |
| `fer` | FER package | Optional, `pip install fer` |
| `none` | Face detection only | For testing the UI with manual keys |

`backend: auto` picks HSEmotion, then DeepFace, then FER+, depending on what
is installed. Scores are reduced to the six target classes (disgust and
contempt are dropped), smoothed with an exponential moving average, and a new
label is accepted only after it has led with ≥ 40 % confidence for 0.6 s.

**Control (`emotion_motion_node`).** Happy, sad and neutral are *continuous*:
the robot follows the label as long as it is held. Angry, surprise and fear
are *one-shot* manoeuvres that run to their odometry target (with a time-out)
and are edge-triggered, so holding the expression does not repeat them.
Rotation is integrated from unwrapped yaw differences, so a full turn reads
2π instead of wrapping to zero.

**Safety.** No face for 1 s → stop. No label for 1 s (detector stopped) →
watchdog stop. `SPACE` in the camera window → latched e-stop. Every node
sends a final zero-velocity command on shutdown, and the command is
republished at 20 Hz because the Create 3 base drops commands older than
0.5 s.

**Per-user calibration.** Expression models are biased against some classes
(fear and sad in particular). `emotion_calibrate` records ~3 s of score
vectors for each of the six expressions (no images are stored) and fits one
gain per class that maximises balanced accuracy on that data. The detector
loads the result at start-up.

## Installation

```bash
sudo apt install -y python3-venv python3-colcon-common-extensions \
  ros-jazzy-nav2-minimal-tb4-sim ros-jazzy-ros-gz

git clone https://github.com/VTongg159/hri-turtlebot4-emotion-control.git
cd hri-turtlebot4-emotion-control

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
scripts/download_models.sh          # HSEmotion + FER+ ONNX models (checksummed)

source /opt/ros/jazzy/setup.bash
python -m colcon build --symlink-install
```

Build with `python -m colcon` inside the virtual environment so that the
executables use the interpreter with the Python dependencies. NumPy stays
below 2.0 for compatibility with the ROS 2 Jazzy binaries. TensorFlow and
DeepFace are only needed for `backend:=deepface`.

## Usage

```bash
source .venv/bin/activate
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Gazebo TurtleBot 4 + webcam detector + motion controller
ros2 launch tb4_emotion_control emotion_control.launch.py start_sim:=true

# Real TurtleBot 4 (same ROS_DOMAIN_ID as the robot)
ros2 launch tb4_emotion_control emotion_control.launch.py
```

| Launch argument | Default | Description |
|---|---|---|
| `start_sim` | `false` | Start Gazebo with a TurtleBot 4 on an open floor |
| `headless` | `false` | Run Gazebo without its GUI |
| `backend` | `auto` | `auto`, `hsemotion`, `deepface`, `ferplus`, `fer`, `none` |
| `camera_source` | `0` | Webcam index or a video file |
| `show_window` | `true` | Show the live recognition window |
| `cmd_vel_stamped` | `false` | Publish `TwistStamped` instead of `Twist` |

Keys in the camera window:

| Key | Action |
|---|---|
| `1`–`6` | Manual override: happy, sad, neutral, angry, surprise, fear |
| `A` / `0` | Back to automatic recognition |
| `SPACE` | Toggle the e-stop |
| `Q` / `Esc` | Quit (the robot stops) |

All tunable values (speeds, distances, filter settings, time-outs) are in
[`config/emotion_control.yaml`](src/tb4_emotion_control/config/emotion_control.yaml).

### Calibrate to your face (recommended)

Stop the launch first, since the tool needs the webcam:

```bash
ros2 run tb4_emotion_control emotion_calibrate            # all six expressions
ros2 run tb4_emotion_control emotion_calibrate --only fear,sad
```

For each expression, make the face, press `SPACE` once and hold it until the
bar fills. The tool prints per-class recall before and after calibration and
writes `~/.ros/tb4_emotion_calibration.yaml`.

### Topics

| Topic | Type | Direction |
|---|---|---|
| `/emotion/label` | `std_msgs/String` | detector → motion: stable label, 10 Hz |
| `/emotion/scores` | `std_msgs/String` (JSON) | detector: raw and smoothed scores |
| `/emotion/estop` | `std_msgs/Bool` (latched) | detector → motion |
| `/emotion/override` | `std_msgs/String` | manual label, or `auto` |
| `/emotion/action` | `std_msgs/String` (JSON) | motion → detector: action, v, ω, progress |
| `/emotion/debug_image` | `sensor_msgs/Image` | detector overlay (`publish_image:=true`) |
| `/cmd_vel` | `geometry_msgs/Twist` | motion → robot |
| `/odom` | `nav_msgs/Odometry` | robot → motion |

## Testing and demo recording

```bash
# Unit tests: score normalisation, filter, state machine, odometry, calibration
python -m pytest -q src/tb4_emotion_control/test

# Gazebo + scripted labels sent straight to the motion node; plots /odom
scripts/run_sim_sequence.sh

# Demo video and figures without screen capture
scripts/record_demo.sh             # scripted expressions (no face needed)
scripts/record_demo.sh --live      # on-screen prompts; the robot follows your face
```

`record_demo.sh` renders two fixed cameras in the Gazebo world offscreen,
blurs everything except the operator's face in the recorded overlay, and
composes a 1920×1080 video with per-step captions, the odometry trail and the
velocity readout. `scripts/finalize_demo.py` then adds title and result
cards, encodes H.264 and measures every manoeuvre from odometry. The
simulation-side nodes run on simulation time, so the manoeuvres and the video
speed stay correct when the host is busy. Output goes to `evidence/`.

## Repository layout

```
src/tb4_emotion_control/
  tb4_emotion_control/
    emotion_detector_node.py   webcam → /emotion/label, live window
    emotion_motion_node.py     /emotion/label → /cmd_vel
    backends.py                HSEmotion / DeepFace / FER / FER+ behind one interface
    emotions.py                label set and score normalisation
    smoother.py                EMA, confidence gate, hold time, face-lost timeout
    motion.py                  emotion → motion primitive table and state machine
    odometry.py                travelled distance and unwrapped rotation
    calibration.py             per-user class gains
    emotion_calibrate.py       calibration tool
    overlay.py                 live window drawing (Vietnamese text via PIL)
    demo_graphics.py           fonts, colours and labels for the video and figures
    emotion_sequence.py        scripted labels or live prompts
    demo_recorder.py           offscreen cameras + overlay → demo video
    trajectory_recorder.py     /odom → CSV
    ros_utils.py               orderly SIGINT/SIGTERM shutdown
  launch/    emotion_control.launch.py, sim.launch.py
  config/    emotion_control.yaml
  worlds/    emotion_arena.sdf (open floor, 1 m grid, two demo cameras)
  test/      test_logic.py
scripts/     download_models.sh, record_demo.sh, finalize_demo.py,
             run_sim_sequence.sh, plot_trajectory.py
```

## Acknowledgements

- HSEmotion: A. V. Savchenko, *face-emotion-recognition*
  ([GitHub](https://github.com/HSE-asavchenko/face-emotion-recognition)).
- FER+: Microsoft, via the [ONNX Model Zoo](https://github.com/onnx/models).
- [DeepFace](https://github.com/serengil/deepface) by S. Serengil.
- TurtleBot 4 simulation from `nav2_minimal_tb4_sim`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
