#!/bin/bash

# Constants
SEED=1
CKPT_DIR="/p/yufeng/qc/exp/qc/highway_qcfql/highway-state/highway_lanefilter_3e6_h9"
CKPT_STEP=2400000
CLASSIFIER_PATH="data/h9_lane_classifier.pkl"
EVAL_EPISODES=10
HORIZON_LENGTH=9

# Arrays to loop over
DIRECTIONS=("right" "left")
STRENGTHS=(0.4 0.3 0.2 0.1)

# Loop over directions
for direction in "${DIRECTIONS[@]}"; do
    # Loop over strengths
    for strength in "${STRENGTHS[@]}"; do
        
        echo "=========================================================="
        echo "Running Intervention: Direction=$direction, Strength=$strength"
        echo "=========================================================="

        # Run the command
        XLA_FLAGS="" MUJOCO_GL=egl python intervention_direction_scaler.py \
            --seed=$SEED \
            --ckpt_dir="$CKPT_DIR" \
            --ckpt_step=$CKPT_STEP \
            --classifier_path="$CLASSIFIER_PATH" \
            --intervention_direction="$direction" \
            --intervention_strength=$strength \
            --eval_episodes=$EVAL_EPISODES \
            --video=True \
            --horizon_length=$HORIZON_LENGTH

    done
done

echo "All experiments finished."