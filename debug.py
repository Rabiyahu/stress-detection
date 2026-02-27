# debug_full.py
import sys, os, json
import numpy as np
import torch
import traceback
from main import StressDetector, extract_audio_from_video, extract_audio_features, extract_visual_features, MAX_LEN, N_MFCC, VIDEO_FEATURE_DIM

def inspect_model(sd: StressDetector):
    print("=== MODEL INFO ===")
    print("device:", sd.device)
    # param counts
    total = sum(p.numel() for p in sd.model.parameters())
    trainable = sum(p.numel() for p in sd.model.parameters() if p.requires_grad)
    print(f"total params: {total:,}, trainable: {trainable:,}")
    # print first layer weight stats if present
    for name, p in sd.model.named_parameters():
        if 'weight' in name:
            print(f"{name} shape={tuple(p.shape)} mean={p.data.mean().item():.6f} std={p.data.std().item():.6f}")
            break

def inspect_features(video_path):
    print("\n=== AUDIO EXTRACTION ===")
    try:
        y = extract_audio_from_video(video_path, cleanup=True)
        print("waveform len:", len(y))
        mfcc = extract_audio_features(y, max_len=MAX_LEN, n_mfcc=N_MFCC)
        print("mfcc shape:", mfcc.shape, "mfcc mean/std:", float(mfcc.mean()), float(mfcc.std()))
        print("mfcc sample[0]:", mfcc[0][:6].tolist())
    except Exception as e:
        print("Audio error:", e); traceback.print_exc()

    print("\n=== VISUAL EXTRACTION ===")
    try:
        vis = extract_visual_features(video_path, max_len=MAX_LEN, target_dim=VIDEO_FEATURE_DIM)
        print("visual shape:", vis.shape, "visual mean/std:", float(vis.mean()), float(vis.std()))
        zeros = (np.isclose(vis, 0.0).all(axis=1)).sum()
        print(f"zero frames: {zeros}/{vis.shape[0]} ({zeros/vis.shape[0]:.3f})")
        print("visual sample[0,:10]:", vis[0,:10].tolist())
    except Exception as e:
        print("Visual error:", e); traceback.print_exc()

def run_inference(sd: StressDetector, video_path):
    print("\n=== MODEL INFERENCE ===")
    out = sd.predict(video_path)
    print("predict() returned:")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_full.py /path/to/video.mp4 [model_path]")
        sys.exit(1)
    video = sys.argv[1]
    mp = sys.argv[2] if len(sys.argv) > 2 else "transformer_model.pt"
    print("Using model:", mp)
    sd = StressDetector(model_path=mp)
    inspect_model(sd)
    inspect_features(video)
    run_inference(sd, video)
