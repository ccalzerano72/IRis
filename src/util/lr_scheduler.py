# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import numpy as np


class CosineAnnealingWarmRestarts:
    def __init__(self, T_0, T_mult=1, eta_min_ratio=0.0, warmup_steps=0, total_iter_length=None) -> None:
        """
        Cosine Annealing with Warm Restarts scheduler with initial warmup.
        
        This scheduler combines:
        1. Linear warmup from 0 to 1.0 over warmup_steps
        2. Cosine annealing cycles with warm restarts
        
        Args:
            T_0 (int): Number of iterations for the first restart cycle
            T_mult (int): Factor to increase T_i after each restart (default: 1)
            eta_min_ratio (float): Minimum learning rate as ratio of base LR (default: 0.0)
            warmup_steps (int): Number of warmup steps with linear increase (default: 0)
            total_iter_length (int): Total expected iterations (for bounds checking, optional)
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min_ratio = eta_min_ratio
        self.warmup_steps = warmup_steps
        self.total_iter_length = total_iter_length
        
        # Pre-calculate restart points for efficiency
        self._calculate_restart_points()
    
    def _calculate_restart_points(self):
        """Pre-calculate the iteration points where restarts occur."""
        self.restart_points = [self.warmup_steps]  # First restart after warmup
        
        T_i = self.T_0
        current_iter = self.warmup_steps
        
        # Calculate restart points until we exceed total_iter_length (if provided)
        max_iter = self.total_iter_length if self.total_iter_length else 1000000
        while current_iter < max_iter:
            current_iter += T_i
            self.restart_points.append(current_iter)
            T_i *= self.T_mult
            
            # Safety check to prevent infinite loops
            if len(self.restart_points) > 100:
                break
    
    def __call__(self, n_iter) -> float:
        """
        Calculate learning rate multiplier for given iteration.
        
        Args:
            n_iter (int): Current iteration number (0-based)
            
        Returns:
            float: Learning rate multiplier (0.0 to 1.0)
        """
        # Warmup phase: linear increase from 0 to 1
        if n_iter < self.warmup_steps:
            if self.warmup_steps == 0:
                return 1.0
            return n_iter / self.warmup_steps
        
        # Find which restart cycle we're in
        cycle_start = self.warmup_steps
        T_i = self.T_0
        
        for i in range(len(self.restart_points) - 1):
            cycle_end = self.restart_points[i + 1]
            
            if n_iter < cycle_end:
                # We're in this cycle
                cycle_progress = (n_iter - cycle_start) / T_i
                
                # Cosine annealing formula
                # Goes from 1.0 to eta_min_ratio following cosine curve
                cosine_factor = 0.5 * (1 + np.cos(np.pi * cycle_progress))
                lr_multiplier = self.eta_min_ratio + (1.0 - self.eta_min_ratio) * cosine_factor
                
                return lr_multiplier
            
            # Move to next cycle
            cycle_start = cycle_end
            T_i *= self.T_mult
        
        # If we're beyond all calculated cycles, return minimum LR
        return self.eta_min_ratio


class IterExponential:
    def __init__(self, total_iter_length, final_ratio, warmup_steps=0) -> None:
        """
        Customized iteration-wise exponential scheduler.
        Re-calculate for every step, to reduce error accumulation

        Args:
            total_iter_length (int): Expected total iteration number
            final_ratio (float): Expected LR ratio at n_iter = total_iter_length
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            alpha = 1.0 * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            alpha = self.final_ratio
        else:
            actual_iter = n_iter - self.warmup_steps
            alpha = np.exp(
                actual_iter / self.effective_length * np.log(self.final_ratio)
            )
        return alpha


if "__main__" == __name__:
    # Test IterExponential scheduler
    lr_scheduler_exp = IterExponential(
        total_iter_length=50000, final_ratio=0.01, warmup_steps=200
    )
    
    # Test CosineAnnealingWarmRestarts scheduler
    lr_scheduler_cosine = CosineAnnealingWarmRestarts(
        T_0=3000,           # First cycle: 5000 iterations
        T_mult=1.5,           # Each cycle doubles in length  
        eta_min_ratio=0.001, # Minimum LR = 1% of base LR
        warmup_steps=300,   # Linear warmup for first 200 steps
        total_iter_length=250000
    )

    x = np.arange(50000)
    alphas_exp = [lr_scheduler_exp(i) for i in x]
    alphas_cosine = [lr_scheduler_cosine(i) for i in x]
    
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    
    # Plot both schedulers
    plt.subplot(2, 1, 1)
    plt.plot(x, alphas_exp, label='IterExponential', linewidth=2)
    plt.title('IterExponential Scheduler')
    plt.xlabel('Iteration')
    plt.ylabel('LR Multiplier')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(x, alphas_cosine, label='CosineAnnealingWarmRestarts', linewidth=2, color='orange')
    plt.title('CosineAnnealingWarmRestarts Scheduler')
    plt.xlabel('Iteration')
    plt.ylabel('LR Multiplier')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("lr_schedulers_comparison.png", dpi=150, bbox_inches='tight')
    print("Scheduler comparison saved to lr_schedulers_comparison.png")
    
    # Print some key values for verification
    print("\nCosineAnnealingWarmRestarts key values:")
    print(f"Warmup end (iter 200): {lr_scheduler_cosine(200):.4f}")
    print(f"First cycle mid (iter 2700): {lr_scheduler_cosine(2700):.4f}")
    print(f"First cycle end (iter 5200): {lr_scheduler_cosine(5200):.4f}")
    print(f"Second cycle start (iter 5201): {lr_scheduler_cosine(5201):.4f}")
    print(f"Second cycle mid (iter 10200): {lr_scheduler_cosine(10200):.4f}")
    print(f"Final iteration (iter 49999): {lr_scheduler_cosine(49999):.4f}")
