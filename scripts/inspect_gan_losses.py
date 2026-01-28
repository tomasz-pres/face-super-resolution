#!/usr/bin/env python
"""Check what's in the GAN checkpoint"""

import torch

checkpoint = torch.load('checkpoints/final_custom_model.pth', map_location='cpu')

history = checkpoint['training_history']

print("Available keys:", list(history.keys()))
print()

print("Lengths:")
for key in history.keys():
    print(f"  {key}: {len(history[key])}")
print()

# Check d_loss
print("d_loss samples (first 5):")
if len(history['d_loss']) > 0:
    for i, val in enumerate(history['d_loss'][:5]):
        print(f"  Epoch {i}: {val}")
else:
    print("  Empty list!")

print()

# Check g_loss
print("g_loss samples (first 5):")
for i, val in enumerate(history['g_loss'][:5]):
    print(f"  Epoch {i}: {val}")

print()

# Check if there are d_real and d_fake
if 'd_real' in history:
    print("d_real samples (first 5):")
    for i, val in enumerate(history['d_real'][:5]):
        print(f"  Epoch {i}: {val}")

print()

if 'd_fake' in history:
    print("d_fake samples (first 5):")
    for i, val in enumerate(history['d_fake'][:5]):
        print(f"  Epoch {i}: {val}")
