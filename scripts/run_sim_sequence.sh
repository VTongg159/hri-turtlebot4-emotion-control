#!/usr/bin/env bash
# Quick simulation check without a webcam: Gazebo TB4 + emotion_motion_node +
# trajectory_recorder + a scripted label sequence sent straight to the motion
# node, then the odometry plot and a per-segment summary.
#
#   scripts/run_sim_sequence.sh [output_dir]
# For the full demo video (detector, cameras, figures) use scripts/record_demo.sh.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(realpath -m "${1:-$ROOT/evidence/sim_sequence}")"
mkdir -p "$OUT"

set +u
source /opt/ros/jazzy/setup.bash
source "$ROOT/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-58}"
SIM_CLOCK=(-p use_sim_time:=true)

groups=()
start() {  # start <log name> <command...>; each node gets its own process group
  local log="$OUT/$1.log"; shift
  setsid "$@" >"$log" 2>&1 &
  groups+=($!)
  disown
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

start sim ros2 launch tb4_emotion_control sim.launch.py headless:=true
echo "waiting for /odom ..."
for _ in $(seq 90); do
  timeout 3 ros2 topic echo --once /odom nav_msgs/msg/Odometry >/dev/null 2>&1 && break
  sleep 1
done
start motion ros2 run tb4_emotion_control emotion_motion_node --ros-args \
  --params-file "$ROOT/src/tb4_emotion_control/config/emotion_control.yaml" "${SIM_CLOCK[@]}"
start recorder ros2 run tb4_emotion_control trajectory_recorder --ros-args \
  -p output:="$OUT/trajectory.csv" "${SIM_CLOCK[@]}"
sleep 2

ros2 run tb4_emotion_control emotion_sequence --ros-args "${SIM_CLOCK[@]}" 2>&1 | tee "$OUT/sequence.log"
sleep 2
stop_all
trap - EXIT

"$ROOT/.venv/bin/python" "$ROOT/scripts/plot_trajectory.py" "$OUT/trajectory.csv" "$OUT/trajectory.png"
grep -- "->" "$OUT/motion.log" | sed 's/.*emotion_motion\]: /  /'
