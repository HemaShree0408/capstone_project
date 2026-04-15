import numpy as np
import os

out = r'C:\Users\PESU-RF\capstone 211\capstone_project\Human Activity Recognition - Video Dataset__DCT'
checked = 0
bad = 0

for root, dirs, files in os.walk(out):
    for f in files:
        if f.endswith('.npy'):
            path = os.path.join(root, f)
            arr = np.load(path)
            is_empty = arr.max() == 0
            size_mb = os.path.getsize(path) / (1024 * 1024)
            status = "❌ EMPTY" if is_empty else "✅ OK"
            if checked < 6 or is_empty:
                rel = os.path.relpath(path, out)
                print(f"{status}  Shape:{arr.shape}  Max:{arr.max():.4f}  Size:{size_mb:.2f}MB  {rel}")
            if is_empty:
                bad += 1
            checked += 1

print(f"\n--- Total: {checked} files | Bad (empty): {bad} | Good: {checked - bad} ---")
