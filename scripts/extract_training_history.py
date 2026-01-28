#!/usr/bin/env python
"""
Extract Training History from Checkpoint
"""

import torch
from pathlib import Path

checkpoint_path = "checkpoints/final_custom_model.pth"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\nCheckpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")
    if isinstance(checkpoint[key], (list, dict)) and key != 'model_state_dict':
        print(f"      Type: {type(checkpoint[key])}")
        if isinstance(checkpoint[key], list):
            print(f"      Length: {len(checkpoint[key])}")
        elif isinstance(checkpoint[key], dict):
            print(f"      Keys: {list(checkpoint[key].keys())[:10]}")

# Try to find training history
if 'history' in checkpoint:
    print("\n✓ Found 'history' key")
    print(f"History type: {type(checkpoint['history'])}")
    if isinstance(checkpoint['history'], dict):
        print(f"History keys: {checkpoint['history'].keys()}")
        for key in checkpoint['history'].keys():
            if isinstance(checkpoint['history'][key], list):
                print(f"  {key}: {len(checkpoint['history'][key])} epochs")
