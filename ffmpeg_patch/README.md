# Custom FFmpeg: H.264 DCT-coefficient and motion-vector extraction

The Python code in this project needs a **modified FFmpeg** whose H.264 decoder dumps
I-frame transform coefficients and P-frame motion vectors while decoding. The full build
(source + `ffmpeg.exe`, several GB) is not stored in git; this folder contains everything
needed to rebuild it.

| File | Content |
|---|---|
| `custom_h264_extract.patch` | `git` patch against upstream FFmpeg commit **`7516bf24db`** (357 lines added, 19 removed) |
| `files/libavcodec/h264_coeff_extract.c/.h` | **new**: the extraction code (I-frame coefficient dump + P-frame MV dump) |
| `files/libavcodec/h264_slice.c` | hooks after `ff_h264_decode_mb_cabac/cavlc`: dump MVs, dump coefficients, skip pixel reconstruction of I-frame macroblocks |
| `files/libavcodec/h264dec.c` | opens the dump files at decoder init, closes them at decoder end |
| `files/libavcodec/Makefile` | adds `h264_coeff_extract.o` to the H.264 decoder |
| `files/libavdevice/dshow.c`, `files/fftools/fftools.manifest` | small MinGW/Windows build fixes |

`files/` holds the complete modified files (same result as applying the patch).

## What the modified decoder writes

| File | When | Record |
|---|---|---|
| `<prefix>_I.bin` | every macroblock of every **I-frame** | 24-byte header (`int64 pts, int32 mb_x, mb_y, qp, slice_type`) + 3072 bytes quantised coefficients (`int16[16*48*2]`) + 192 bytes luma DC |
| `<prefix>_P.bin` | every macroblock of every **P-frame** | 80 bytes: `int64 frame_idx, int32 mb_x, int32 mb_y, int16 mv[16][2]` (list-0, quarter-pixel) |

`<prefix>` = environment variable `H264_COEFF_EXTRACT_FILE`, default `STRICT_FINAL` in the
current directory. The files are opened every time an H.264 decoder starts (also by `ffprobe`).
For I-frame macroblocks the pixel reconstruction (`ff_h264_hl_decode_mb`) is skipped;
P/B-frames are still reconstructed normally.

> **Use `-threads 1`.** The dump uses global file handles without locking; with FFmpeg's default
> frame-threaded decoding several threads write the MV file at once and the records interleave
> (corrupted, different on every run). Single-threaded decoding is deterministic.

## Build on Windows (MSYS2 / MinGW-w64)

1. Install [MSYS2](https://www.msys2.org/), open **MSYS2 MinGW x64** and install the tools:
   ```bash
   pacman -S make diffutils pkgconf nasm mingw-w64-x86_64-gcc git
   ```
2. Get FFmpeg at the exact commit the patch was made for:
   ```bash
   git clone https://github.com/FFmpeg/FFmpeg.git
   cd FFmpeg
   git checkout 7516bf24db
   ```
3. Apply the patch (or copy the files from `files/` over the FFmpeg tree):
   ```bash
   git apply /path/to/capstone_project/ffmpeg_patch/custom_h264_extract.patch
   ```
4. Build (the `make` step takes 5–15 minutes):
   ```bash
   export PATH=/mingw64/bin:$PATH
   ./configure
   make -j$(nproc)
   ```
5. Place the result where the Python code expects it: copy the FFmpeg folder (with
   `ffmpeg.exe`, `ffprobe.exe` and the MinGW DLLs they need, e.g. from `/mingw64/bin`) to
   `capstone_project/FFmpeg/FFmpeg/`, so that `FFmpeg/FFmpeg/ffmpeg.exe` exists.

## Quick test

```bash
export H264_COEFF_EXTRACT_FILE=STRICT_FINAL
./ffmpeg.exe -threads 1 -i MyVideo.mp4 -an -f null -
ls -l STRICT_FINAL_I.bin STRICT_FINAL_P.bin
```

Then, from the project root, `python fast_compressed_ops.py MyVideo.mp4` parses both files and
checks the Python parsers against each other.
