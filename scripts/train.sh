#!/bin/bash

DATA_DIR="polyhaven_dataset"
BATCH_SIZE=4
NUM_WORKERS=4
OUTPUT_DIR="outputs/polyhaven_training"
CUBE_SIZE=512
NUM_EPOCHS=30
LEARNING_RATE=8e-5
GRADIENT_ACCUM=4
EVAL_INTERVAL=3
SAVE_INTERVAL=5
PREDICTION_TYPE="v"
FOV=95.0
OVERLAP=2.5
PRECISION="fp16"
SEED=42

# Parameter summary
echo "==============================================="
echo "        CubeDiff Training Configuration        "
echo "==============================================="
echo "Data Directory:         $DATA_DIR"
echo "Output Directory:       $OUTPUT_DIR"
echo "Batch Size:             $BATCH_SIZE"
echo "Workers:                $NUM_WORKERS"
echo "Cube Size:              $CUBE_SIZE"
echo "Epochs:                 $NUM_EPOCHS"
echo "Learning Rate:          $LEARNING_RATE"
echo "Gradient Accumulation:  $GRADIENT_ACCUM"
echo "Evaluation Interval:    $EVAL_INTERVAL epochs"
echo "Save Interval:          $SAVE_INTERVAL epochs"
echo "Prediction Type:        $PREDICTION_TYPE"
echo "Field of View:          $FOV degrees"
echo "Overlap:                $OVERLAP degrees"
echo "Mixed Precision:        $PRECISION"
echo "Random Seed:            $SEED"
echo "==============================================="

echo "Starting training..."
echo ""

accelerate launch --mixed_precision $PRECISION train_epoch_logging.py \
  --data_dir $DATA_DIR \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --output_dir $OUTPUT_DIR \
  --cube_size $CUBE_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --gradient_accumulation_steps $GRADIENT_ACCUM \
  --eval_interval $EVAL_INTERVAL \
  --save_interval $SAVE_INTERVAL \
  --prediction_type $PREDICTION_TYPE \
  --fov $FOV \
  --overlap $OVERLAP \
  --seed $SEED

TRAINING_STATUS=$?

if [ $TRAINING_STATUS -eq 0 ]; then
  echo ""
  echo "==============================================="
  echo "Training completed successfully!"
  echo "==============================================="
else
  echo ""
  echo "==============================================="
  echo "Training failed with error code $TRAINING_STATUS"
  echo "Check logs for details"
  echo "==============================================="
fi
