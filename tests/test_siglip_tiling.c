/**
 * @file test_siglip_tiling.c
 * @brief Regression test for the Idefics3 image preprocessing (tile grid + Lanczos resampling).
 *
 * The expected grids and pixel values come from transformers' Idefics3ImageProcessor
 * (granite-docling preprocessor_config: longest_edge 2048, max_image_size 512, LANCZOS).
 * The old runtime always used a fixed 4 x 3 + 1 grid, which broke every image that is not
 * page-shaped (e.g. wide table crops) -- those need 5..9 tiles.
 */

#include "eif_siglip.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

static int write_ppm(const char *path, int w, int h)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned char px[3] = {(unsigned char)(x * 255 / (w - 1)), (unsigned char)(y * 255 / (h - 1)),
                                   (unsigned char)((x * 7 + y * 13) % 256)};
            fwrite(px, 1, 3, f);
        }
    fclose(f);
    return 0;
}

static void check_golden(const char *path, int w, int h, int exp_rows, int exp_cols,
                         const int pts[][4], const float *vals, int n_pts)
{
    float *tiles = NULL;
    int rows = 0, cols = 0;
    CHECK(write_ppm(path, w, h) == 0, "cannot write %s", path);
    CHECK(eif_siglip_preprocess_image(path, &tiles, &rows, &cols) == 0, "preprocess %dx%d", w, h);
    if (!tiles) return;
    CHECK(rows == exp_rows && cols == exp_cols, "%dx%d grid %dx%d, expected %dx%d", w, h, rows, cols, exp_rows, exp_cols);
    const size_t tile = 3u * 512u * 512u;
    for (int i = 0; i < n_pts; i++) {
        int t = pts[i][0] < 0 ? rows * cols : pts[i][0]; /* -1 = global tile */
        float v = tiles[(size_t)t * tile + (size_t)pts[i][1] * 512 * 512 + (size_t)pts[i][2] * 512 + pts[i][3]];
        CHECK(fabsf(v - vals[i]) < 0.02f, "%dx%d tile %d c%d (%d,%d) = %.5f, expected %.5f", w, h, t, pts[i][1],
              pts[i][2], pts[i][3], v, vals[i]);
    }
    free(tiles);
    remove(path);
}

int main(void)
{
    /* grid selection, from Idefics3ImageProcessor: {w, h, rows, cols} */
    static const int plan[][4] = {
        {449, 92, 1, 4},   {186, 58, 2, 4},    {768, 1024, 4, 3}, {1275, 1650, 4, 4},
        {1241, 1754, 4, 3}, {2480, 3508, 4, 3}, {300, 300, 4, 4},  {600, 200, 2, 4},
        {100, 1000, 4, 1}, {512, 512, 4, 4},   {2048, 2048, 4, 4}, {1, 1, 4, 4},
    };
    for (size_t i = 0; i < sizeof(plan) / sizeof(plan[0]); i++) {
        int r = 0, c = 0;
        CHECK(eif_siglip_plan_tiles(plan[i][0], plan[i][1], &r, &c) == 0, "plan %dx%d", plan[i][0], plan[i][1]);
        CHECK(r == plan[i][2] && c == plan[i][3], "plan %dx%d -> %dx%d, expected %dx%d", plan[i][0], plan[i][1], r, c,
              plan[i][2], plan[i][3]);
        CHECK(r * c + 1 <= EIF_SIGLIP_MAX_TILES, "tile count above EIF_SIGLIP_MAX_TILES");
    }
    CHECK(eif_siglip_plan_tiles(0, 10, NULL, NULL) != 0, "zero width must be rejected");

    /* pixel values of a synthetic gradient image, from the same processor: {tile, channel, y, x} */
    static const int wide_pts[][4] = {{0, 0, 0, 0}, {0, 1, 100, 200}, {1, 2, 300, 50}, {-1, 0, 256, 256},
                                      {-1, 1, 17, 490}, {-1, 2, 500, 3}};
    static const float wide_vals[] = {-1.0f, -0.61569f, -0.83529f, 0.00392f, -0.9451f, 0.22353f};
    check_golden("/tmp/eif_tiling_wide.ppm", 449, 92, 1, 4, wide_pts, wide_vals, 6);

    static const int tall_pts[][4] = {{0, 0, 0, 0}, {0, 1, 100, 200}, {1, 2, 300, 50}, {-1, 0, 256, 256},
                                      {-1, 1, 17, 490}, {-1, 2, 500, 3}};
    static const float tall_vals[] = {-1.0f, -0.90588f, 0.68627f, -0.00392f, -0.93725f, 0.6549f};
    check_golden("/tmp/eif_tiling_tall.ppm", 300, 520, 4, 3, tall_pts, tall_vals, 6);

    if (failures == 0) printf("test_siglip_tiling: all checks passed\n");
    return failures ? 1 : 0;
}
