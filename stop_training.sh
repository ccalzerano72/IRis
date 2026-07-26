#!/bin/bash

# Stop training script

if [ -f training.pid ]; then
    PID=$(cat training.pid)
    echo "Stopping training process (PID: $PID)..."
    kill $PID 2>/dev/null
    
    # Wait a moment
    sleep 2
    
    # Force kill if still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "Force killing process..."
        kill -9 $PID 2>/dev/null
    fi
    
    rm training.pid
    echo "✓ Training stopped"
    
    # Clear GPU memory
    echo "Clearing GPU memory..."
    python clear_gpu_memory.py
else
    echo "No training.pid file found"
    echo "Searching for training processes..."
    
    # Find and kill any training processes
    pkill -f "script/restoration/train.py"
    
    echo "✓ All training processes stopped"
    
    # Clear GPU memory
    python clear_gpu_memory.py
fi
