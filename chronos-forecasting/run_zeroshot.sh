#!/bin/bash

# Example script to run zero-shot evaluation with Chronos-2
# Modify the parameters below according to your needs

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory containing all datasets
BASE_DIR="/path/to/your/preprocessed_dataset/test_dataset"

# Output directory for results
OUTPUT_DIR="./zeroshot_results"

# Model name (default: amazon/chronos-2)
MODEL_NAME="amazon/chronos-2"

# Context lengths to evaluate (space-separated)
CONTEXT_LENGTHS="12 48 96 144 192 288"

# Prediction length (number of time steps to predict)
PREDICTION_LENGTH=18

# Step size for rolling window
STEP_SIZE=1

# Device to use (cuda or cpu)
DEVICE="cuda"

# =============================================================================
# SINGLE DATASET EVALUATION
# =============================================================================

# Example 1: Evaluate a single dataset
# Uncomment and modify the line below to run
# python run_zeroshot_evaluation.py \
#     --data_dir "${BASE_DIR}/BIG_IDEA_LAB" \
#     --output_dir "${OUTPUT_DIR}/BIG_IDEA_LAB" \
#     --model_name "${MODEL_NAME}" \
#     --context_lengths ${CONTEXT_LENGTHS} \
#     --prediction_length ${PREDICTION_LENGTH} \
#     --step_size ${STEP_SIZE} \
#     --device ${DEVICE} \
#     --dataset_name "BIG_IDEA_LAB"

# =============================================================================
# MULTIPLE DATASETS EVALUATION
# =============================================================================

# Example 2: Evaluate multiple datasets in a loop
# List of datasets to process
DATASETS=(
    "BIG_IDEA_LAB"
    "ShanghaiT1DM"
    "ShanghaiT2DM"
    "CGMacros"
    "UCHTT1DM"
    "1_Hall2018"
    "2_D1NAMO"
    "3_cOLAS2019"
    "14_HUPA-UCM"
    "17_T1DM-UOM"
    "18_Bris-T1D Open"
    "19_AZT1D"
)

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Processing dataset: ${DATASET}"
    echo "========================================"
    
    DATA_PATH="${BASE_DIR}/${DATASET}"
    
    # Check if dataset directory exists
    if [ ! -d "${DATA_PATH}" ]; then
        echo "Warning: Dataset directory '${DATA_PATH}' does not exist. Skipping..."
        continue
    fi
    
    # Run evaluation
    python run_zeroshot_evaluation.py \
        --data_dir "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}/${DATASET}" \
        --model_name "${MODEL_NAME}" \
        --context_lengths ${CONTEXT_LENGTHS} \
        --prediction_length ${PREDICTION_LENGTH} \
        --step_size ${STEP_SIZE} \
        --device ${DEVICE} \
        --dataset_name "${DATASET}"
    
    echo ""
    echo "Completed: ${DATASET}"
    echo ""
done

echo "========================================"
echo "All evaluations completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"

# =============================================================================
# ADDITIONAL EXAMPLES
# =============================================================================

# Example 3: Run with specific context lengths only
# python run_zeroshot_evaluation.py \
#     --data_dir "${BASE_DIR}/BIG_IDEA_LAB" \
#     --output_dir "${OUTPUT_DIR}/BIG_IDEA_LAB_quick" \
#     --context_lengths 48 96 144 \
#     --device ${DEVICE}

# Example 4: Run on CPU
# python run_zeroshot_evaluation.py \
#     --data_dir "${BASE_DIR}/BIG_IDEA_LAB" \
#     --output_dir "${OUTPUT_DIR}/BIG_IDEA_LAB_cpu" \
#     --device cpu

# Example 5: Run with custom prediction length and step size
# python run_zeroshot_evaluation.py \
#     --data_dir "${BASE_DIR}/BIG_IDEA_LAB" \
#     --output_dir "${OUTPUT_DIR}/BIG_IDEA_LAB_custom" \
#     --prediction_length 24 \
#     --step_size 6 \
#     --device ${DEVICE}
