/**
 * @file test_loop_guard.c
 * @brief eif_llm_repetition_span: detects degenerate generation loops without firing on normal DocTags.
 */
#include "eif_llm.h"
#include <stdio.h>

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } } while (0)

int main(void)
{
    int32_t t[512];

    /* 1. distinct tokens: no loop */
    for (int i = 0; i < 200; i++) t[i] = 1000 + i;
    CHECK(eif_llm_repetition_span(t, 200, 64) == 0, "distinct tokens");

    /* 2. a 7-token phrase repeated 20 times after normal text (the Turkish body-text failure) */
    int n = 0;
    for (int i = 0; i < 50; i++) t[n++] = 2000 + i;
    for (int r = 0; r < 20; r++) for (int k = 0; k < 7; k++) t[n++] = 3000 + k;
    int s = eif_llm_repetition_span(t, n, 64);
    CHECK(s >= 140 && s <= 141, "7-token loop x20: span %d", s);

    /* 3. the same phrase only 3 times is not a loop */
    n = 0;
    for (int i = 0; i < 50; i++) t[n++] = 2000 + i;
    for (int r = 0; r < 3; r++) for (int k = 0; k < 30; k++) t[n++] = 3000 + k;
    CHECK(eif_llm_repetition_span(t, n, 64) < 64, "30-token phrase x3 must stay below the 64-token guard");

    /* 4. OTSL-like table: repeating structure but changing cell text / loc tokens */
    n = 0;
    for (int row = 0; row < 20; row++) {
        t[n++] = 100339;               /* fixed tag */
        t[n++] = 5000 + row;           /* row-specific text */
        t[n++] = 100339;
        t[n++] = 6000 + row * 3;
        t[n++] = 198;                  /* <nl> */
    }
    CHECK(eif_llm_repetition_span(t, n, 64) < 64, "table rows with changing content");

    /* 5. one token repeated 70 times */
    for (int i = 0; i < 70; i++) t[i] = 42;
    CHECK(eif_llm_repetition_span(t, 70, 64) == 70, "single-token run");

    /* 6. period longer than max_period is ignored */
    n = 0;
    for (int r = 0; r < 5; r++) for (int k = 0; k < 80; k++) t[n++] = 7000 + k;
    CHECK(eif_llm_repetition_span(t, n, 64) == 0, "period 80 > max_period");

    CHECK(eif_llm_repetition_span(NULL, 10, 64) == 0, "NULL input");
    if (!failures) printf("test_loop_guard: all checks passed\n");
    return failures ? 1 : 0;
}
