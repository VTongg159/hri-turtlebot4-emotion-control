#!/usr/bin/env bash
# Record the lab-08 demo video and report figures without screen capture.
#
#   scripts/record_demo.sh            scripted: expressions are sent to the
#                                     detector as overrides (no face needed)
#   scripts/record_demo.sh --live     live: the detector window shows
#                                     "Hãy thể hiện: …" prompts and the robot
#                                     follows the real recognition result
#
# Options: --out DIR (default evidence/demo or evidence/demo_live),
#          --backend NAME (default auto: DeepFace if installed, else FER+).
# Output:  DIR/demo.mp4, DIR/snapshots/, DIR/trajectory.{csv,png},
#          DIR/figures/ + DIR/figures/results.tex (copied into the report by
#          scripts/finalize_demo.py --report).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=scripted
OUT=""
BACKEND=auto
while [[ $# -gt 0 ]]; do
  case "$1" in
    --live) MODE=live ;;
    --out) OUT="$2"; shift ;;
    --backend) BACKEND="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -z "$OUT" ]] && OUT="$ROOT/evidence/$([[ $MODE == live ]] && echo demo_live || echo demo)"
OUT="$(realpath -m "$OUT")"
rm -rf "$OUT" && mkdir -p "$OUT"

set +u
source /opt/ros/jazzy/setup.bash
source "$ROOT/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-62}"
PARAMS="$ROOT/src/tb4_emotion_control/config/emotion_control.yaml"

if [[ $MODE == live ]]; then
  SEQUENCE="neutral:5,happy:10,neutral:4,sad:10,neutral:4,angry:7,neutral:4,fear:7,neutral:4,surprise:10,neutral:4"
  SEQ_ARGS=(-p mode:=prompt)
  SHOW_WINDOW=true
  # prompts and video follow the person (wall clock); the robot runs on simulation time
  HUMAN_CLOCK=(-p use_sim_time:=false)
  DET_EXTRA=()
else
  SEQUENCE="neutral:3,happy:9,neutral:2,sad:9,neutral:2,angry:4,neutral:2,fear:4,neutral:2,surprise:9,neutral:3"
  SEQ_ARGS=(-p mode:=label -p topic:=/emotion/override)
  SHOW_WINDOW=false
  # everything on simulation time: steps and video stay true to the robot's speed even
  # when the host is busy (Gazebo then runs slower than real time)
  HUMAN_CLOCK=(-p use_sim_time:=true)
  # the robot must not react to whoever sits at the webcam before the script starts
  DET_EXTRA=(-p start_manual:=neutral)
fi

# Every node runs in its own process group so the whole tree can be stopped.
groups=()
start() {  # start <log name> <command...>
  local log="$OUT/$1.log"; shift
  setsid "$@" >"$log" 2>&1 &
  groups+=($!)
  disown  # no "Killed" job notices from bash when the group is stopped
}
stop_all() {
  for pg in "${groups[@]}"; do kill -INT -- "-$pg" 2>/dev/null || true; done
  for _ in $(seq 20); do
    alive=0
    for pg in "${groups[@]}"; do kill -0 -- "-$pg" 2>/dev/null && alive=1; done
    [[ $alive == 0 ]] && break
    sleep 0.5
  done
  for pg in "${groups[@]}"; do kill -KILL -- "-$pg" 2>/dev/null || true; done
  groups=()
}
trap stop_all EXIT

wait_topic() {  # wait_topic <topic> <type> <timeout s>
  for _ in $(seq "$3"); do
    timeout 3 ros2 topic echo --once "$1" "$2" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "timeout waiting for $1" >&2
  return 1
}

echo "[1/4] Gazebo (offscreen demo cameras)"
start sim ros2 launch tb4_emotion_control sim.launch.py headless:=true
wait_topic /odom nav_msgs/msg/Odometry 90
wait_topic /demo_cam/perspective sensor_msgs/msg/Image 60

echo "[2/4] detector ($BACKEND) + motion + recorders"
# Cap TensorFlow's thread pools so DeepFace does not starve Gazebo (keeps real-time factor ~1).
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=2
start detector ros2 run tb4_emotion_control emotion_detector_node --ros-args --params-file "$PARAMS" \
  -p backend:="$BACKEND" -p show_window:=$SHOW_WINDOW -p publish_image:=true "${DET_EXTRA[@]}"
SIM_CLOCK=(-p use_sim_time:=true)
start motion ros2 run tb4_emotion_control emotion_motion_node --ros-args --params-file "$PARAMS" "${SIM_CLOCK[@]}"
start trajectory ros2 run tb4_emotion_control trajectory_recorder --ros-args -p output:="$OUT/trajectory.csv" \
  "${SIM_CLOCK[@]}"
wait_topic /emotion/debug_image sensor_msgs/msg/Image 120
start demo_recorder ros2 run tb4_emotion_control demo_recorder --ros-args -p output_dir:="$OUT" "${HUMAN_CLOCK[@]}"
sleep 3

echo "[3/4] sequence ($MODE)"
ros2 run tb4_emotion_control emotion_sequence --ros-args -p sequence:="$SEQUENCE" "${SEQ_ARGS[@]}" \
  "${HUMAN_CLOCK[@]}" \
  2>&1 | tee "$OUT/sequence.log"
sleep 2
stop_all
trap - EXIT

echo "[4/4] video, figures"
"$ROOT/.venv/bin/python" "$ROOT/scripts/finalize_demo.py" "$OUT" --mode "$MODE"
grep -- "->" "$OUT/motion.log" | sed 's/.*emotion_motion\]: /  /'
echo "done: $OUT/demo.mp4"
