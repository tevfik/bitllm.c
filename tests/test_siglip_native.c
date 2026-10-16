/**
 * @file test_siglip_native.c
 * @brief Unit test verifying native C99 SigLIP2 Vision Encoder & Connector.
 */

#include "eif_siglip.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *find_existing(const char *paths[], int n)
{
    for (int i = 0; i < n; i++) {
        FILE *f = fopen(paths[i], "rb");
        if (f) { fclose(f); return paths[i]; }
    }
    return paths[0];
}

int main(int argc, char **argv)
{
    const char *model_candidates[] = {
        "../../../siglip_test/docling_siglip_test.eifm",
        "../../siglip_test/docling_siglip_test.eifm",
        "artifacts/siglip_test/docling_siglip_test.eifm",
        "../artifacts/siglip_test/docling_siglip_test.eifm"
    };
    const char *pix_candidates[] = {
        "../../../siglip_test/tile0_pixels.bin",
        "../../siglip_test/tile0_pixels.bin",
        "artifacts/siglip_test/tile0_pixels.bin",
        "../artifacts/siglip_test/tile0_pixels.bin"
    };
    const char *ref_candidates[] = {
        "../../../siglip_test/tile0_conn_ref.bin",
        "../../siglip_test/tile0_conn_ref.bin",
        "artifacts/siglip_test/tile0_conn_ref.bin",
        "../artifacts/siglip_test/tile0_conn_ref.bin"
    };

    const char *model_path = (argc > 1) ? argv[1] : find_existing(model_candidates, 4);
    const char *pixels_path = (argc > 2) ? argv[2] : find_existing(pix_candidates, 4);
    const char *ref_conn_path = (argc > 3) ? argv[3] : find_existing(ref_candidates, 4);

    printf("====================================================\n");
    printf("  Test: Native C99 SigLIP2 Vision Encoder Validation\n");
    printf("  Model   : %s\n", model_path);
    printf("  Pixels  : %s\n", pixels_path);
    printf("  Ref Conn: %s\n", ref_conn_path);
    printf("====================================================\n");

    /* 1. Read SVLM header from model file */
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open model: %s\n", model_path);
        return 1;
    }
    uint32_t svlm_hdr[12];
    if (fread(svlm_hdr, sizeof(uint32_t), 12, f) != 12) {
        fprintf(stderr, "Failed to read SVLM header\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    uint32_t v_off = svlm_hdr[6];
    uint32_t v_sz = svlm_hdr[10];
    printf("  SVLM Vision Offset: %u (0x%08X), Size: %u bytes (%.2f MB)\n",
           v_off, v_off, v_sz, (double)v_sz / (1024.0 * 1024.0));

    /* 2. Load SigLIP model */
    eif_siglip_t siglip;
    int rc = eif_siglip_load_from_file(&siglip, model_path, v_off, v_sz);
    if (rc != 0) {
        fprintf(stderr, "Failed to load SigLIP model (rc=%d)\n", rc);
        return 1;
    }
    printf("  ✓ Loaded SigLIP2 Vision Encoder (%d layers, dim=%d, heads=%d, interm=%d)\n",
           siglip.header.num_layers, siglip.header.hidden_size,
           siglip.header.num_heads, siglip.header.intermediate_dim);

    /* 3. Read tile0 input pixels [3, 512, 512] */
    FILE *fp = fopen(pixels_path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open pixels file: %s\n", pixels_path);
        eif_siglip_free(&siglip);
        return 1;
    }
    size_t pix_floats = 3 * 512 * 512;
    float *tile_pixels = (float *)malloc(pix_floats * sizeof(float));
    if (fread(tile_pixels, sizeof(float), pix_floats, fp) != pix_floats) {
        fprintf(stderr, "Failed to read tile pixels\n");
        fclose(fp);
        free(tile_pixels);
        eif_siglip_free(&siglip);
        return 1;
    }
    fclose(fp);

    /* 4. Read reference connector output [64, 576] */
    FILE *fr = fopen(ref_conn_path, "rb");
    if (!fr) {
        fprintf(stderr, "Failed to open reference connector file: %s\n", ref_conn_path);
        free(tile_pixels);
        eif_siglip_free(&siglip);
        return 1;
    }
    size_t ref_floats = 64 * 576;
    float *ref_tokens = (float *)malloc(ref_floats * sizeof(float));
    if (fread(ref_tokens, sizeof(float), ref_floats, fr) != ref_floats) {
        fprintf(stderr, "Failed to read reference tokens\n");
        fclose(fr);
        free(tile_pixels);
        free(ref_tokens);
        eif_siglip_free(&siglip);
        return 1;
    }
    fclose(fr);

    /* 5. Run native C99 tile encoding */
    float *out_tokens = (float *)malloc(ref_floats * sizeof(float));
    printf("  • Running native C99 ViT forward + Connector on tile 0...\n");
    rc = eif_siglip_encode_tile(&siglip, tile_pixels, out_tokens);
    if (rc != 0) {
        fprintf(stderr, "eif_siglip_encode_tile failed (rc=%d)\n", rc);
        free(tile_pixels);
        free(ref_tokens);
        free(out_tokens);
        eif_siglip_free(&siglip);
        return 1;
    }

    /* 6. Validate mathematical precision against PyTorch reference */
    double dot = 0.0, norm_out = 0.0, norm_ref = 0.0;
    float max_diff = 0.0f;
    double sum_diff = 0.0;

    for (size_t i = 0; i < ref_floats; i++) {
        float diff = fabsf(out_tokens[i] - ref_tokens[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
        dot += (double)out_tokens[i] * (double)ref_tokens[i];
        norm_out += (double)out_tokens[i] * (double)out_tokens[i];
        norm_ref += (double)ref_tokens[i] * (double)ref_tokens[i];
    }

    double cos_sim = dot / (sqrt(norm_out) * sqrt(norm_ref));
    double mean_diff = sum_diff / (double)ref_floats;

    printf("\n--- Precision Evaluation ---\n");
    printf("  Output Shape    : [64, 576] (36,864 float values)\n");
    printf("  Max Diff        : %.6e\n", max_diff);
    printf("  Mean Diff       : %.6e\n", mean_diff);
    printf("  Cosine Similarity: %.8f\n", cos_sim);

    free(tile_pixels);
    free(ref_tokens);
    free(out_tokens);
    eif_siglip_free(&siglip);

    if (cos_sim >= 0.999) {
        printf("\n✅ PASS: Native C99 SigLIP2 matches PyTorch reference (Cosine Similarity = %.6f >= 0.999)!\n", cos_sim);
        return 0;
    } else {
        printf("\n❌ FAIL: Cosine similarity %.6f is below threshold 0.999!\n", cos_sim);
        return 1;
    }
}
