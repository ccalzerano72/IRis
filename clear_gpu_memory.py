#!/usr/bin/env python3
"""
Clear GPU memory before starting training.
Run this if you get CUDA out of memory errors.
"""

import torch
import gc

def clear_gpu_memory():
    """Clear all GPU memory"""
    print("Clearing GPU memory...")
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Print memory stats
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated(i) / 1024**3:.2f} GB")
            
            # Reset peak stats
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.reset_accumulated_memory_stats(i)
        
        print("\n✓ GPU memory cleared!")
    else:
        print("No CUDA devices available")

if __name__ == "__main__":
    clear_gpu_memory()
