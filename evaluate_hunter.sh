#!/bin/bash

# Hunter Locomotion Evaluation Script
# This script provides easy access to different locomotion evaluation scenarios

CHECKPOINT_PATH="logs/FullTrainingHunter/20250715_134736-FullTrainingHunter-locomotion-hunter/model_83000.pt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
HUMANOIDVERSE_DIR="$SCRIPT_DIR"

# Function to display usage
usage() {
    echo "Usage: $0 [scenario] [options]"
    echo ""
    echo "Available scenarios:"
    echo "  forward     - Forward locomotion at constant velocity"
    echo "  backward    - Backward locomotion at constant velocity" 
    echo "  left        - Left lateral movement"
    echo "  right       - Right lateral movement"
    echo "  rotate_cw   - Clockwise rotation"
    echo "  rotate_ccw  - Counter-clockwise rotation"
    echo "  static      - Static balance test"
    echo "  manual      - Manual control mode (use WASD keys + Q/E for rotation)"
    echo "  push        - Original lateral push evaluation"
    echo ""
    echo "Options:"
    echo "  --checkpoint PATH    - Path to model checkpoint (default: $CHECKPOINT_PATH)"
    echo "  --headless          - Run in headless mode"
    echo "  --num_envs N        - Number of environments (default: 1)"
    echo "  --dry-run           - Print command without executing"
    echo "  --help              - Show this help message"
    echo ""
    echo "Keyboard controls during evaluation:"
    echo "  W/S    - Forward/Backward movement"
    echo "  A/D    - Left/Right movement" 
    echo "  Q/E    - Counter-clockwise/Clockwise rotation"
    echo "  X      - Stop all movement"
    echo "  1-4    - Force control (left/right hand up/down)"
    echo ""
    exit 1
}

# Parse command line arguments
SCENARIO=""
CHECKPOINT="logs/FullTrainingHunter/20250715_134736-FullTrainingHunter-locomotion-hunter/model_83000.pt"
HEADLESS=""
NUM_ENVS=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --checkpoint=*)
            CHECKPOINT="${arg#*=}"
            shift
            ;;
        --headless)
            HEADLESS="headless=True"
            shift
            ;;
        --num_envs=*)
            NUM_ENVS="num_envs=${arg#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        --*)
            echo "Unknown option: $arg"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$SCENARIO" ]; then
                SCENARIO=$arg
            else
                echo "Multiple scenarios specified. Only one scenario allowed."
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done# Check if scenario is provided
if [ -z "$SCENARIO" ]; then
    echo "Error: No scenario specified"
    usage
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    echo "Please provide a valid checkpoint path using --checkpoint"
    exit 1
fi

# Source conda environment
echo "Activating isaaclab environment..."
source ~/miniconda3/bin/activate isaaclab

# Build command based on scenario
case $SCENARIO in
    forward)
        echo "Running Forward Locomotion Scenario..."
        python "$HUMANOIDVERSE_DIR/humanoidverse/eval_agent.py" \
            +checkpoint="$CHECKPOINT" \
            +simulator=isaacsim \
            +exp=locomotion \
            ++env.config.max_episode_length_s=60 \
            ++env.config.locomotion_command_ranges.lin_vel_x="[0.5,0.5]" \
            ++env.config.locomotion_command_ranges.lin_vel_y="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.ang_vel_yaw="[0.0,0.0]" \
            ++env.config.locomotion_command_resampling_time=1000.0 \
            num_envs="$NUM_ENVS" \
            $HEADLESS_FLAG
        ;;
    backward)
        echo "Running Backward Locomotion Scenario..."
        python "$HUMANOIDVERSE_DIR/humanoidverse/eval_agent.py" \
            +checkpoint="$CHECKPOINT" \
            +simulator=isaacsim \
            +exp=locomotion \
            ++env.config.max_episode_length_s=60 \
            ++env.config.locomotion_command_ranges.lin_vel_x="[-0.5,-0.5]" \
            ++env.config.locomotion_command_ranges.lin_vel_y="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.ang_vel_yaw="[0.0,0.0]" \
            ++env.config.locomotion_command_resampling_time=1000.0 \
            num_envs="$NUM_ENVS" \
            $HEADLESS_FLAG
        ;;
    left)
        echo "Running Left Lateral Locomotion Scenario..."
        python "$HUMANOIDVERSE_DIR/humanoidverse/eval_agent.py" \
            +checkpoint="$CHECKPOINT" \
            +simulator=isaacsim \
            +exp=locomotion \
            ++env.config.max_episode_length_s=60 \
            ++env.config.locomotion_command_ranges.lin_vel_x="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.lin_vel_y="[0.5,0.5]" \
            ++env.config.locomotion_command_ranges.ang_vel_yaw="[0.0,0.0]" \
            ++env.config.locomotion_command_resampling_time=1000.0 \
            num_envs="$NUM_ENVS" \
            $HEADLESS_FLAG
        ;;
    right)
        echo "Running Right Lateral Locomotion Scenario..."
        python "$HUMANOIDVERSE_DIR/humanoidverse/eval_agent.py" \
            +checkpoint="$CHECKPOINT" \
            +simulator=isaacsim \
            +exp=locomotion \
            ++env.config.max_episode_length_s=60 \
            ++env.config.locomotion_command_ranges.lin_vel_x="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.lin_vel_y="[-0.5,-0.5]" \
            ++env.config.locomotion_command_ranges.ang_vel_yaw="[0.0,0.0]" \
            ++env.config.locomotion_command_resampling_time=1000.0 \
            num_envs="$NUM_ENVS" \
            $HEADLESS_FLAG
        ;;
    rotate_cw)
        echo "Running Clockwise Rotation Scenario..."
        python "$HUMANOIDVERSE_DIR/humanoidverse/eval_agent.py" \
            +checkpoint="$CHECKPOINT" \
            +simulator=isaacsim \
            +exp=locomotion \
            ++env.config.max_episode_length_s=60 \
            ++env.config.locomotion_command_ranges.lin_vel_x="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.lin_vel_y="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.ang_vel_yaw="[-0.5,-0.5]" \
            ++env.config.locomotion_command_resampling_time=1000.0 \
            num_envs="$NUM_ENVS" \
            $HEADLESS_FLAG
        ;;
    rotate_ccw)
        echo "Running Counter-Clockwise Rotation Scenario..."
        python "$HUMANOIDVERSE_DIR/humanoidverse/eval_agent.py" \
            +checkpoint="$CHECKPOINT" \
            +simulator=isaacsim \
            +exp=locomotion \
            ++env.config.max_episode_length_s=60 \
            ++env.config.locomotion_command_ranges.lin_vel_x="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.lin_vel_y="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.ang_vel_yaw="[0.5,0.5]" \
            ++env.config.locomotion_command_resampling_time=1000.0 \
            num_envs="$NUM_ENVS" \
            $HEADLESS_FLAG
        ;;
    static)
        echo "Running Static Balance Scenario..."
        python "$HUMANOIDVERSE_DIR/humanoidverse/eval_agent.py" \
            +checkpoint="$CHECKPOINT" \
            +simulator=isaacsim \
            +exp=locomotion \
            ++env.config.max_episode_length_s=60 \
            ++env.config.locomotion_command_ranges.lin_vel_x="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.lin_vel_y="[0.0,0.0]" \
            ++env.config.locomotion_command_ranges.ang_vel_yaw="[0.0,0.0]" \
            ++env.config.locomotion_command_resampling_time=1000.0 \
            num_envs="$NUM_ENVS" \
            $HEADLESS_FLAG
        ;;
    manual)
        echo "Running Manual Control Mode..."
        echo "Use WASD keys for movement, Q/E for rotation, X to stop"
        python "$HUMANOIDVERSE_DIR/humanoidverse/eval_agent.py" \
            +checkpoint="$CHECKPOINT" \
            +simulator=isaacsim \
            +env.config.max_episode_length_s=10000 \
            +env.config.locomotion_command_resampling_time=10000 \
            num_envs="$NUM_ENVS" \
            $HEADLESS_FLAG
        ;;
    push)
        echo "Running Original Lateral Push Evaluation..."
        python "$HUMANOIDVERSE_DIR/humanoidverse/eval_agent.py" \
            +checkpoint="$CHECKPOINT" \
            +simulator=isaacsim \
            +domain_rand.push_robots=True \
            +domain_rand.max_push_vel_xy=0.7 \
            +domain_rand.push_interval_s=[3,8] \
            num_envs="$NUM_ENVS" \
            $HEADLESS_FLAG
        ;;
    *)
        echo "Error: Unknown scenario '$SCENARIO'"
        usage
        ;;
esac
