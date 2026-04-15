import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def build_frequency_attention_maps(filename, frame_target=1):
    """
    Parses the extracted H.264 quantized transform coefficients and builds
    a Spatial Feature Map (Frequency-Band Attention Map) for Object Detection.
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    # Header and Data definitions
    HDR_FORMAT = '<Iiiii'
    HDR_SIZE = struct.calcsize(HDR_FORMAT)
    
    # 16 * 48 * 2 elements of int16 (2 bytes each) = 3072 bytes
    COEFF_BYTES = 16 * 48 * 2 * 2
    LUMA_DC_BYTES = 3 * 16 * 2 * 2
    REC_SIZE = HDR_SIZE + COEFF_BYTES + LUMA_DC_BYTES
    
    file_size = os.path.getsize(filename)
    num_mbs = file_size // REC_SIZE
    
    # First, find the maximum width and height (mb_x, mb_y) for our tensor allocation
    max_mb_x = 0
    max_mb_y = 0
    with open(filename, 'rb') as f:
        for _ in range(num_mbs):
            chunk = f.read(REC_SIZE)
            frame_num, mb_x, mb_y, mb_type, cbp = struct.unpack(HDR_FORMAT, chunk[:HDR_SIZE])
            if frame_num == frame_target:
                if mb_x > max_mb_x: max_mb_x = mb_x
                if mb_y > max_mb_y: max_mb_y = mb_y
                
    # A single Macroblock (MB) is 16x16 pixels, which is made of sixteen 4x4 blocks.
    # Therefore, the grid of 4x4 blocks is (max_mb_y + 1) * 4 x (max_mb_x + 1) * 4
    grid_rows = (max_mb_y + 1) * 4
    grid_cols = (max_mb_x + 1) * 4
    
    # We create a 3-channel tensor: (H, W, 3) 
    # Channel 0: Low Freq, Channel 1: Mid Freq, Channel 2: High Freq
    feature_tensor = np.zeros((grid_rows, grid_cols, 3), dtype=np.float32)
    
    # Define our frequency bands mapped to a linear 1D index (4x4 = 16 indices)
    # 0  1  2  3
    # 4  5  6  7
    # 8  9 10 11
    # 12 13 14 15
    low_indices  = [0, 1, 4]
    mid_indices  = [2, 3, 5, 6, 8]
    high_indices = [7, 9, 10, 11, 12, 13, 14, 15]

    print(f"Building Spatial Tensor of size: {grid_rows} x {grid_cols} x 3")
    
    # Read the data and populate the tensor
    mbs_processed = 0
    with open(filename, 'rb') as f:
        for _ in range(num_mbs):
            chunk = f.read(REC_SIZE)
            frame_num, mb_x, mb_y, mb_type, cbp = struct.unpack(HDR_FORMAT, chunk[:HDR_SIZE])
            
            # Skip if it is not the target frame
            if frame_num != frame_target:
                continue
                
            mbs_processed += 1
            
            # The next bytes are the raw 1D array of quantized transform values
            # Array holds 48 blocks total (16 Luma Y, 16 Chroma Cb, 16 Chroma Cr)
            coeffs_raw = np.frombuffer(chunk[HDR_SIZE : HDR_SIZE + COEFF_BYTES], dtype=np.int16)
            
            # Step 1: Approximate Dequantization
            # (Because we want relative spatial feature maps, mapping raw integers 
            #  scales linearly to energy. E.g., multiplying everything by 'step' 
            #  doesn't change the CNN feature contrast much. We use a flat scale factor.)
            dequantized = coeffs_raw.astype(np.float32) * 1.0 
            
            # We ONLY process the first 16 blocks (The Luma/Brightness channel)
            # 16 blocks * 16 coefficients = 256 values
            luma_blocks = dequantized[:256]
            
            # Step 2: Loop over the sixteen 4x4 blocks in this Macroblock
            for block_idx in range(16):
                # Extract the 16 coefficients for this particular 4x4 block
                start_idx = block_idx * 16
                block_coeffs = luma_blocks[start_idx : start_idx + 16]
                
                # Step 3: Compute Energies (Σ |coeff|²)
                e_low = np.sum(np.square(block_coeffs[low_indices]))
                e_mid = np.sum(np.square(block_coeffs[mid_indices]))
                e_high = np.sum(np.square(block_coeffs[high_indices]))
                
                # Step 4: Map the linear block_idx (0-15 H.264 Z-scan) into 4x4 spatial grid positions
                # Z-scan order algorithm for H.264 blocks:
                # 0  1   4  5
                # 2  3   6  7
                # 8  9  12 13
                # 10 11 14 15
                block_col = (block_idx % 2) + ((block_idx // 4) % 2) * 2
                block_row = ((block_idx // 2) % 2) + ((block_idx // 8) % 2) * 2
                
                # Global coordinates on the whole video frame
                global_row = mb_y * 4 + block_row
                global_col = mb_x * 4 + block_col
                
                # Step 5: Assign to our 3-Channel Tensor (applying log(1 + E) as requested)
                # This compresses massive energy spikes into a very neural-network-friendly range
                feature_tensor[global_row, global_col, 0] = np.log1p(e_low)
                feature_tensor[global_row, global_col, 1] = np.log1p(e_mid)
                feature_tensor[global_row, global_col, 2] = np.log1p(e_high)

    print(f"Processed {mbs_processed} macroblocks for Frame {frame_target}.")
    
    # -------------------------------------------------------------
    # Visualization: Plotting the 3 channels separately
    # The tensor is already log(1 + E) scaled, so it is ready to plot!
    # -------------------------------------------------------------
    vis_tensor = feature_tensor
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Show Low Freq Map
    ax = axes[0]
    im = ax.imshow(vis_tensor[:, :, 0], cmap='viridis', interpolation='nearest')
    ax.set_title("Low Frequency (Smooth Regions)")
    fig.colorbar(im, ax=ax)
    
    # Show Mid Freq Map
    ax = axes[1]
    im = ax.imshow(vis_tensor[:, :, 1], cmap='plasma', interpolation='nearest')
    ax.set_title("Mid Frequency (Edges)")
    fig.colorbar(im, ax=ax)
    
    # Show High Freq Map
    ax = axes[2]
    im = ax.imshow(vis_tensor[:, :, 2], cmap='inferno', interpolation='nearest')
    ax.set_title("High Frequency (Fine Textures)")
    fig.colorbar(im, ax=ax)
    
    # Show Multi-Channel Composite (RGB)
    # We normalize each channel to 0.0 - 1.0 for RGB plotting to prevent visual clipping
    composite = np.zeros_like(vis_tensor)
    max_l = np.max(vis_tensor[:, :, 0])
    max_m = np.max(vis_tensor[:, :, 1])
    max_h = np.max(vis_tensor[:, :, 2])
    
    if max_l > 0: composite[:, :, 0] = vis_tensor[:, :, 0] / max_l
    if max_m > 0: composite[:, :, 1] = vis_tensor[:, :, 1] / max_m
    if max_h > 0: composite[:, :, 2] = vis_tensor[:, :, 2] / max_h
    
    ax = axes[3]
    ax.imshow(composite)
    ax.set_title("Composite Feature Map (RGB)")

    plt.tight_layout()
    output_png = "frequency_attention_maps.png"
    plt.savefig(output_png, dpi=150)
    print(f"Saved visualization to '{output_png}'! Check your Capstone folder!")
    plt.show()

if __name__ == "__main__":
    build_frequency_attention_maps('coeffs.bin', frame_target=1)
