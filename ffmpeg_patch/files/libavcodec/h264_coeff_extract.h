/*
 * H.264 Transform Coefficient Extractor
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * H.264 DCT/transform coefficient and motion-vector extraction interface.
 *
 * This header exposes two extractors that intercept internal decoder data:
 *
 * 1. I-frame DCT coefficients (ff_h264_extract_coeffs)
 *    Intercepts raw quantized transform coefficients from the entropy-decoded
 *    H.264 bitstream (CAVLC or CABAC), before any inverse quantization or
 *    inverse transform is applied.
 *    Output: STRICT_FINAL_I.bin (or ${H264_COEFF_EXTRACT_FILE}_I.bin)
 *
 *    Per-macroblock record (little-endian):
 *      int64_t  pts              presentation timestamp
 *      int32_t  mb_x             macroblock column index
 *      int32_t  mb_y             macroblock row index
 *      int32_t  qp               quantization parameter
 *      int32_t  slice_type       AV_PICTURE_TYPE_* value
 *      int16_t  coeff[16*48]     raw quantized DCT coefficients
 *      int16_t  luma_dc[16*3*2]  luma DC coefficients
 *
 * 2. P-frame motion vectors (ff_h264_extract_mvs)
 *    Intercepts the decoded list-0 motion vectors for every macroblock of
 *    every P-frame.  B-frames are silently skipped.
 *    Output: STRICT_FINAL_P.bin (or ${H264_COEFF_EXTRACT_FILE}_P.bin)
 *
 *    Per-macroblock record (little-endian, fixed 80 bytes):
 *      int64_t  frame_idx        sequential P-frame index (0, 1, 2, ...)
 *      int32_t  mb_x             macroblock column
 *      int32_t  mb_y             macroblock row
 *      int16_t  mv[16][2]        16 sub-block {dx, dy} pairs (quarter-pixel)
 *
 * Usage:
 *   Set the environment variable H264_COEFF_EXTRACT_FILE to a path prefix
 *   before running any FFmpeg tool, e.g.
 *       set H264_COEFF_EXTRACT_FILE=coeffs
 *   Both files are always created.  If the variable is not set the prefix
 *   defaults to "STRICT_FINAL".
 */

#ifndef AVCODEC_H264_COEFF_EXTRACT_H
#define AVCODEC_H264_COEFF_EXTRACT_H

#include <stdint.h>
#include "h264dec.h"

/**
 * Initialise the coefficient extractor for the given codec context.
 *
 * Opens the output file specified by the H264_COEFF_EXTRACT_FILE environment
 * variable.  If the variable is not set, extraction is disabled and all
 * subsequent calls to ff_h264_extract_coeffs() are no-ops.
 *
 * @param avctx  codec context (used for logging only)
 * @return 0 on success, or a negative AVERROR code on failure
 */
int ff_h264_coeff_extract_init(AVCodecContext *avctx);

/**
 * Close the coefficient extractor and flush/close the output file.
 *
 * Safe to call even when extraction was never initialised.
 */
void ff_h264_coeff_extract_close(void);

/**
 * Dump the quantized transform coefficients for one I-frame macroblock.
 *
 * Must be called *after* entropy decoding (CAVLC or CABAC) has filled
 * sl->mb[] and *before* ff_h264_hl_decode_mb() is called so that the
 * coefficients have not yet been inverse-quantized or inverse-transformed.
 *
 * @param h   H.264 context
 * @param sl  slice context for the current macroblock
 * @return 1 if I-frame extraction is active (caller should skip pixel reconstruction),
 *         0 otherwise (normal decoding continues)
 */
int ff_h264_extract_coeffs(const H264Context *h, const H264SliceContext *sl);

/**
 * Dump the list-0 motion vectors for one P-frame macroblock.
 *
 * Must be called at the same hook point as ff_h264_extract_coeffs(), i.e.
 * after ff_h264_decode_mb_cabac/cavlc() fills the slice context.
 * B-frames and I-frames are silently ignored.
 *
 * @param h   H.264 context
 * @param sl  slice context for the current macroblock
 */
void ff_h264_extract_mvs(const H264Context *h, const H264SliceContext *sl);

#endif /* AVCODEC_H264_COEFF_EXTRACT_H */