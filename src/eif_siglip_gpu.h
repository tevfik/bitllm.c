/** @file eif_siglip_gpu.h  @brief SigLIP2 encoder on the Vulkan backend (internal, see eif_siglip_use_gpu). */
#ifndef EIF_SIGLIP_GPU_H
#define EIF_SIGLIP_GPU_H

#include "eif_siglip.h"

/* Upload the weights and allocate activations for up to max_batch tiles per submit (halved until it fits).
 * Returns 0 on success; on failure nothing stays allocated. */
int eif_siglip_gpu_attach(eif_siglip_t *model, eif_gpu_context_t *ctx, int max_batch);
void eif_siglip_gpu_release(eif_siglip_t *model);
int eif_siglip_gpu_batch(const eif_siglip_t *model);
/* Encode n tiles ([3, 512, 512] each) into outs[i] ([64, 576] each). Returns 0 on success. */
int eif_siglip_gpu_encode(const eif_siglip_t *model, const float *const *tiles, int n_tiles, float *const *outs);

#endif
