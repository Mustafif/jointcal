#!/bin/bash

# Run the model evaluation on a specific saved model with random_data.csv

# Set variables
MODEL_PATH="saved_models/scalable_hn_dataset_250x60_20250820/model.pt"
DATA_PATH="random_data.csv"
OUTPUT_PATH="evaluation_results.csv"

# Check if the model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Available models:"
    find saved_models -name "model.pt" | sort
    exit 1
fi

# Check if the data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found at $DATA_PATH"
    exit 1
fi

# Run the evaluation script
echo "Running evaluation with:"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_PATH"
echo "  Output: $OUTPUT_PATH"
echo ""

python evaluate_model.py --model "$MODEL_PATH" --data "$DATA_PATH" --output "$OUTPUT_PATH"

# Check if the evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_PATH"

    # Display the first few rows of the results
    echo ""
    echo "First few rows of the results:"
    head -n 5 "$OUTPUT_PATH"
else
    echo ""
    echo "Evaluation failed."
fi
