import numpy as np
import os

OUT = r"C:\Users\PESU-RF\capstone 211\capstone_project\Human Activity Recognition - Video Dataset__DCT"

print("=" * 65)
print("DEEP DATA VALIDATION")
print("=" * 65)

# --- 1. Pick one video with 2 I-frames and compare them ---
sample_video = None
sample_files = []
for root, dirs, files in os.walk(OUT):
    npys = sorted([f for f in files if f.endswith('.npy')])
    if len(npys) >= 2:
        sample_video = root
        sample_files = [os.path.join(root, f) for f in npys[:2]]
        break

if sample_files:
    a = np.load(sample_files[0])
    b = np.load(sample_files[1])
    print(f"\n📂 Sample video: {os.path.relpath(sample_video, OUT)}")
    print(f"\n[Frame 1 → {os.path.basename(sample_files[0])}]")
    print(f"  Shape : {a.shape}")
    print(f"  Low   ch → min:{a[:,:,0].min():.3f}  max:{a[:,:,0].max():.3f}  mean:{a[:,:,0].mean():.3f}")
    print(f"  Mid   ch → min:{a[:,:,1].min():.3f}  max:{a[:,:,1].max():.3f}  mean:{a[:,:,1].mean():.3f}")
    print(f"  High  ch → min:{a[:,:,2].min():.3f}  max:{a[:,:,2].max():.3f}  mean:{a[:,:,2].mean():.3f}")

    print(f"\n[Frame 2 → {os.path.basename(sample_files[1])}]")
    print(f"  Shape : {b.shape}")
    print(f"  Low   ch → min:{b[:,:,0].min():.3f}  max:{b[:,:,0].max():.3f}  mean:{b[:,:,0].mean():.3f}")
    print(f"  Mid   ch → min:{b[:,:,1].min():.3f}  max:{b[:,:,1].max():.3f}  mean:{b[:,:,1].mean():.3f}")
    print(f"  High  ch → min:{b[:,:,2].min():.3f}  max:{b[:,:,2].max():.3f}  mean:{b[:,:,2].mean():.3f}")

    diff = np.mean(np.abs(a - b))
    are_identical = np.array_equal(a, b)
    print(f"\n  Mean absolute difference between I-frames: {diff:.4f}")
    print(f"  Are they identical? {'❌ YES (BAD - same data!)' if are_identical else '✅ NO (GOOD - different frames!)'}")

    # Check channels are distinct from each other within same frame
    low_vs_mid = np.mean(np.abs(a[:,:,0] - a[:,:,1]))
    mid_vs_high = np.mean(np.abs(a[:,:,1] - a[:,:,2]))
    print(f"\n  Channel distinctness (Frame 1):")
    print(f"  Low vs Mid  diff: {low_vs_mid:.4f}  {'✅' if low_vs_mid > 0.01 else '❌'}")
    print(f"  Mid vs High diff: {mid_vs_high:.4f}  {'✅' if mid_vs_high > 0.01 else '❌'}")

# --- 2. Full dataset stats ---
print("\n" + "=" * 65)
print("FULL DATASET SUMMARY")
print("=" * 65)
all_maxes = []
all_means = []
bad = 0
total = 0
for root, dirs, files in os.walk(OUT):
    for f in sorted(files):
        if f.endswith('.npy'):
            arr = np.load(os.path.join(root, f))
            if arr.max() == 0:
                bad += 1
            all_maxes.append(arr.max())
            all_means.append(arr.mean())
            total += 1

print(f"  Total files   : {total}")
print(f"  Empty (max=0) : {bad}  {'✅' if bad == 0 else '❌'}")
print(f"  Max value     : min={min(all_maxes):.3f}  avg={np.mean(all_maxes):.3f}  max={max(all_maxes):.3f}")
print(f"  Mean value    : min={min(all_means):.4f}  avg={np.mean(all_means):.4f}  max={max(all_means):.4f}")
print(f"\n{'✅ DATA LOOKS CORRECT!' if bad == 0 and np.mean(all_maxes) > 1.0 else '❌ ISSUES FOUND!'}")
