#!/bin/bash

# Script to archive model files in logs subdirectories
# Keeps only the highest iteration model and zips the rest

set -e

LOGS_DIR="$(dirname "$0")"
cd "$LOGS_DIR"

echo "Starting model archival process in: $(pwd)"

# Function to process a single directory
process_directory() {
    local dir="$1"
    echo "Processing directory: $dir"
    
    cd "$dir"
    
    # Find all .pt model files
    model_files=($(ls -1 model_*.pt 2>/dev/null | sort -V))
    
    if [ ${#model_files[@]} -eq 0 ]; then
        echo "  No model files found in $dir"
        cd - > /dev/null
        return
    fi
    
    echo "  Found ${#model_files[@]} model files"
    
    # Find the highest iteration model (last in sorted array)
    highest_model="${model_files[-1]}"
    echo "  Highest iteration model: $highest_model"
    
    # If there's only one model, nothing to archive
    if [ ${#model_files[@]} -eq 1 ]; then
        echo "  Only one model file, nothing to archive"
        cd - > /dev/null
        return
    fi
    
    # Create archive name based on directory name and timestamp
    dir_basename=$(basename "$dir")
    archive_name="archived_models_${dir_basename}_$(date +%Y%m%d_%H%M%S).zip"
    
    # Archive ALL models (including the highest)
    models_to_archive=("${model_files[@]}")
    
    echo "  Archiving all ${#models_to_archive[@]} models to: $archive_name"
    
    # Create zip archive with all models
    zip -q "$archive_name" "${models_to_archive[@]}"
    
    if [ $? -eq 0 ]; then
        echo "  Archive created successfully"
        
        # Remove all models except the highest iteration
        models_to_remove=("${model_files[@]:0:${#model_files[@]}-1}")
        echo "  Removing ${#models_to_remove[@]} older model files..."
        for model in "${models_to_remove[@]}"; do
            rm "$model"
            echo "    Removed: $model"
        done
        
        echo "  Kept: $highest_model"
        echo "  Archive: $archive_name"
    else
        echo "  Error: Failed to create archive"
    fi
    
    cd - > /dev/null
}

# Find all directories containing model files
echo "Searching for directories with model files..."
model_dirs=($(find . -name "model_*.pt" -exec dirname {} \; | sort -u))

echo "Found ${#model_dirs[@]} directories with model files:"
for dir in "${model_dirs[@]}"; do
    echo "  $dir"
done
echo ""

# Process each directory containing models
for dir in "${model_dirs[@]}"; do
    process_directory "$dir"
    echo ""
done

echo "Model archival process completed!"