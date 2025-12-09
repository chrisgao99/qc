#!/bin/bash

CKPT_DIR="/p/yufeng/qc/exp/qc/highway_qcfql/highway-state/sd00020251128_120115"
VIDEO_DIR_BASE="/p/yufeng/qc/videos/highway_lcfilter_3e6"
OUTFILE="eval_all_ckpts.log"

echo "==== Starting batch evaluation ====" | tee -a $OUTFILE
echo "Time: $(date)" | tee -a $OUTFILE

for STEP in $(seq 2000000 200000 4000000); do
    echo "----------------------------------------" | tee -a $OUTFILE
    echo "Evaluating checkpoint step: $STEP" | tee -a $OUTFILE
    echo "Start time: $(date)" | tee -a $OUTFILE
    echo "----------------------------------------" | tee -a $OUTFILE

    VIDEO_DIR="${VIDEO_DIR_BASE}/${STEP}"
    mkdir -p "$VIDEO_DIR"

    # Run evaluation
    XLA_FLAGS="" MUJOCO_GL=egl python eval_highway_ckpt.py \
        --seed=1 \
        --agent=agents/acfql.py \
        --ckpt_dir="$CKPT_DIR" \
        --ckpt_step="$STEP" \
        --horizon_length=7 \
        --eval_episodes=50 \
        --video_episodes=5 \
        --video_dir="$VIDEO_DIR" \
        >> $OUTFILE 2>&1

    echo "Finished step $STEP at $(date)" | tee -a $OUTFILE
done

echo "==== All evaluations complete ====" | tee -a $OUTFILE
echo "Finish Time: $(date)" | tee -a $OUTFILE
