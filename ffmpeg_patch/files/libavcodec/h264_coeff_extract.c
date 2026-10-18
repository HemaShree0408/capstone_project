#include "h264_coeff_extract.h"
#include "h264dec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Global file handles
 * ----------------------------------------------------------------------- */
static FILE *g_file_i = NULL;  /* I-frame DCT coefficients  → STRICT_FINAL_I.bin */
static FILE *g_file_p = NULL;  /* P-frame motion vectors    → STRICT_FINAL_P.bin */

/* Running frame counter so each decoded frame gets a sequential index.
 * The counter is shared: it increments whenever we see mb_x==0 && mb_y==0
 * in ff_h264_extract_mvs(), i.e. once per P-frame. */
static int64_t g_p_frame_idx = 0;

/* -----------------------------------------------------------------------
 * Init: open both output files
 * ----------------------------------------------------------------------- */
int ff_h264_coeff_extract_init(AVCodecContext *avctx)
{
    const char *filename = getenv("H264_COEFF_EXTRACT_FILE");
    if (!filename)
        filename = "STRICT_FINAL";

    /* --- I-frame file --- */
    char name_i[1024];
    snprintf(name_i, sizeof(name_i), "%s_I.bin", filename);
    g_file_i = fopen(name_i, "wb");
    if (!g_file_i) {
        av_log(avctx, AV_LOG_ERROR,
               "H264 Extractor: Failed to open I-Frame bin: %s\n", name_i);
        return -1;
    }
    av_log(avctx, AV_LOG_INFO,
           "H264 Extractor: I-Frame extraction active. Output: %s\n", name_i);

    /* --- P-frame MV file --- */
    char name_p[1024];
    snprintf(name_p, sizeof(name_p), "%s_P.bin", filename);
    g_file_p = fopen(name_p, "wb");
    if (!g_file_p) {
        av_log(avctx, AV_LOG_ERROR,
               "H264 Extractor: Failed to open P-Frame MV bin: %s\n", name_p);
        fclose(g_file_i);
        g_file_i = NULL;
        return -1;
    }
    av_log(avctx, AV_LOG_INFO,
           "H264 Extractor: P-Frame MV extraction active. Output: %s\n", name_p);

    g_p_frame_idx = 0;
    return 0;
}

/* -----------------------------------------------------------------------
 * Close: flush and close both files
 * ----------------------------------------------------------------------- */
void ff_h264_coeff_extract_close(void)
{
    if (g_file_i) { fclose(g_file_i); g_file_i = NULL; }
    if (g_file_p) { fclose(g_file_p); g_file_p = NULL; }
}

/* -----------------------------------------------------------------------
 * ff_h264_extract_coeffs – I-frame DCT coefficient dump (unchanged logic)
 * ----------------------------------------------------------------------- */
int ff_h264_extract_coeffs(const H264Context *h, const H264SliceContext *sl)
{
    /* ONLY process I-Frames */
    if (sl->slice_type != AV_PICTURE_TYPE_I || !g_file_i)
        return 0;

    if (sl->mb_x == 0 && sl->mb_y == 0) {
        if (h && h->avctx)
            av_log(h->avctx, AV_LOG_INFO,
                   "[EXTRACTION] Processing I-Frame (PTS %lld)\n",
                   h->cur_pic_ptr ? h->cur_pic_ptr->f->pts : -1);
    }

    /* Per-macroblock binary header */
    struct {
        int64_t pts;
        int32_t mb_x;
        int32_t mb_y;
        int32_t qp;
        int32_t slice_type;
    } header = {
        h->cur_pic_ptr ? h->cur_pic_ptr->f->pts : -1,
        sl->mb_x,
        sl->mb_y,
        sl->qscale,
        (int32_t)sl->slice_type
    };
    fwrite(&header, sizeof(header), 1, g_file_i);

    /* AC coefficients (3072 bytes) */
    fwrite(sl->mb, 1, 16 * 48 * 2 * 2, g_file_i);

    /* Luma DC coefficients (192 bytes) */
    fwrite(sl->mb_luma_dc, 1, 16 * 3 * 4, g_file_i);

    return 1; /* skip pixel reconstruction for I-frames */
}

/* -----------------------------------------------------------------------
 * ff_h264_extract_mvs – P-frame motion vector dump
 *
 * Called from the same hook points as ff_h264_extract_coeffs() in
 * h264_slice.c, immediately after ff_h264_decode_mb_cabac/cavlc().
 *
 * Rules:
 *   - P-frame (slice_type_nos == AV_PICTURE_TYPE_P):
 *       Write one record per macroblock containing the 16 list-0
 *       sub-block MVs (quarter-pixel, signed int16_t pairs dx/dy).
 *   - B-frame: silently ignored (return 0, normal decoding continues).
 *   - I-frame: silently ignored here (handled by ff_h264_extract_coeffs).
 *
 * On-disk record format (little-endian, fixed size = 80 bytes):
 *
 *   int64_t  frame_idx          sequential P-frame index (0, 1, 2, …)
 *   int32_t  mb_x               macroblock column
 *   int32_t  mb_y               macroblock row
 *   int16_t  mv[16][2]          16 × (dx, dy) in quarter-pixel units
 *                               from list-0 motion_val; (0,0) for skipped
 *
 * Total: 8 + 4 + 4 + 16*2*2 = 80 bytes per macroblock record.
 *
 * The Python reader reconstructs the dominant MV for each macroblock by
 * averaging (or taking the median of) the 16 sub-block vectors.
 * ----------------------------------------------------------------------- */
void ff_h264_extract_mvs(const H264Context *h, const H264SliceContext *sl)
{
    /* Only care about P-frames; skip I and B */
    if (!g_file_p)
        return;
    if (sl->slice_type_nos != AV_PICTURE_TYPE_P)
        return;

    /* Detect frame boundary: when we're at the first macroblock (0,0)
     * of a new P-frame, bump the sequential frame counter. */
    if (sl->mb_x == 0 && sl->mb_y == 0) {
        if (h && h->avctx)
            av_log(h->avctx, AV_LOG_INFO,
                   "[MV EXTRACTION] P-Frame idx=%lld (PTS %lld)\n",
                   (long long)g_p_frame_idx,
                   h->cur_pic_ptr ? (long long)h->cur_pic_ptr->f->pts : -1LL);
        g_p_frame_idx++;
    }

    /* ------------------------------------------------------------------
     * Read the 16 list-0 motion vectors for this macroblock.
     *
     * h->cur_pic.motion_val[list][b_xy] is int16_t[2] = {dx, dy}
     * where b_xy = 4*mb_x + 4*mb_y * h->b_stride  (top-left sub-block).
     * The macroblock spans a 4×4 grid of sub-blocks in b-space.
     * ------------------------------------------------------------------ */
    int16_t mv_out[16][2];   /* 16 sub-block (dx,dy) pairs */
    memset(mv_out, 0, sizeof(mv_out));

    if (h->cur_pic.motion_val[0]) {
        const int b_stride = h->b_stride;
        const int b_xy     = 4 * sl->mb_x + 4 * sl->mb_y * b_stride;
        const int16_t (*mv_src)[2] = h->cur_pic.motion_val[0];

        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                int idx = row * 4 + col;
                int b   = b_xy + col + row * b_stride;
                mv_out[idx][0] = mv_src[b][0];   /* dx  (quarter-pixel) */
                mv_out[idx][1] = mv_src[b][1];   /* dy  (quarter-pixel) */
            }
        }
    }

    /* ------------------------------------------------------------------
     * Write the fixed-size record to STRICT_FINAL_P.bin
     * ------------------------------------------------------------------ */
    int64_t frame_idx_val = g_p_frame_idx - 1;   /* current frame (just incremented above) */
    int32_t mb_x = (int32_t)sl->mb_x;
    int32_t mb_y = (int32_t)sl->mb_y;

    fwrite(&frame_idx_val, sizeof(int64_t), 1, g_file_p);
    fwrite(&mb_x,          sizeof(int32_t), 1, g_file_p);
    fwrite(&mb_y,          sizeof(int32_t), 1, g_file_p);
    fwrite(mv_out,         sizeof(mv_out),  1, g_file_p);
}