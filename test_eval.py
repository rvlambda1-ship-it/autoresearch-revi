#!/usr/bin/env python3
"""Test if evaluation phase works independently."""
import sys
import torch
from prepare import Tokenizer, make_dataloader, evaluate_bpb, MAX_SEQ_LEN

print("Loading tokenizer...", flush=True)
tokenizer = Tokenizer.from_directory()

print("Creating val dataloader...", flush=True)
val_loader = make_dataloader(tokenizer, 16, MAX_SEQ_LEN, "val")

print("Fetching first val batch...", flush=True)
x, y, _ = next(val_loader)
print(f"Got batch: x.shape={x.shape}, y.shape={y.shape}", flush=True)

print("Fetching second val batch...", flush=True)
x, y, _ = next(val_loader)
print(f"Got batch: x.shape={x.shape}, y.shape={y.shape}", flush=True)

print("Test complete - val_loader works!", flush=True)
