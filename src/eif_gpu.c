/**
 * @file eif_gpu.c
 * @brief Optional Vulkan compute backend. The loader is opened at runtime (no link dependency); every
 *        failure (no loader, no device, no memory) makes the callers fall back to the CPU kernels.
 */

#include "eif_gpu.h"
#include "eif_gpu_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(EIF_USE_VULKAN)
#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"
#include "eif_vk_shaders.h"
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define EIF_VK_INSTANCE_FUNCS(X)                                                                     \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)              \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties)               \
    X(vkCreateDevice) X(vkGetDeviceProcAddr)

#define EIF_VK_DEVICE_FUNCS(X)                                                                       \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer)                      \
    X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory)       \
    X(vkMapMemory) X(vkUnmapMemory) X(vkCreateCommandPool) X(vkDestroyCommandPool)                   \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer)                        \
    X(vkResetCommandBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants)    \
    X(vkCmdDispatch) X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkQueueSubmit) X(vkDeviceWaitIdle) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout)                  \
    X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)             \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool)                       \
    X(vkDestroyDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets)                  \
    X(vkUpdateDescriptorSets) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences)

#define EIF_VK_NBIND 5
#define EIF_VK_MAX_SETS 2048
#define EIF_VK_STAGING_BYTES ((size_t)16 << 20)

struct eif_gpu_buf {
    VkBuffer buf;
    VkDeviceMemory mem;
    size_t size;
    void *mapped;
};

struct eif_gpu_weight {
    eif_gpu_buf_t *buf;
    int N, K;
    bool q8;
    uint32_t s_off; /* scales offset in floats (INT8) */
};

typedef struct {
    void *lib;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
#define X(n) PFN_##n n;
    EIF_VK_INSTANCE_FUNCS(X)
    EIF_VK_DEVICE_FUNCS(X)
#undef X
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    VkQueue queue;
    VkPhysicalDeviceMemoryProperties mem_props;
    VkDeviceSize max_range;
    uint32_t max_groups[3];
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pl;
    VkPipeline pipes[EIF_GPU_K_COUNT];
    VkDescriptorPool dpool;
    bool recording;
    bool failed;
    eif_gpu_buf_t *dummy;
    eif_gpu_buf_t *staging;
    eif_gpu_buf_t *scr[5]; /* generic GEMM scratch: A, W, scales, bias, C */
} vk_state_t;

static void *vk_open_loader(void)
{
    const char *env = getenv("BITLLM_VULKAN_LIB");
#if defined(_WIN32)
    if (env && *env) return (void *)LoadLibraryA(env);
    return (void *)LoadLibraryA("vulkan-1.dll");
#else
    static const char *names[] = {
#if defined(__APPLE__)
        "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib",
#else
        "libvulkan.so.1", "libvulkan.so",
#endif
    };
    if (env && *env) return dlopen(env, RTLD_NOW | RTLD_LOCAL);
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        void *h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (h) return h;
    }
    return NULL;
#endif
}

static void *vk_sym(void *lib, const char *name)
{
#if defined(_WIN32)
    return (void *)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static void vk_close_loader(void *lib)
{
    if (!lib) return;
#if defined(_WIN32)
    FreeLibrary((HMODULE)lib);
#else
    dlclose(lib);
#endif
}

static bool env_on(const char *name)
{
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0") != 0;
}

static uint32_t pick_memory_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
                                 VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred)
{
    for (int pass = 0; pass < 2; pass++) {
        VkMemoryPropertyFlags want = pass == 0 ? (required | preferred) : required;
        for (uint32_t i = 0; i < mp->memoryTypeCount; i++)
            if ((bits & (1u << i)) && (mp->memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return UINT32_MAX;
}

eif_gpu_buf_t *eif_gpu_buf_create(eif_gpu_context_t *ctx, size_t bytes, bool host_visible)
{
    if (!ctx || !ctx->internal_handle) return NULL;
    vk_state_t *s = (vk_state_t *)ctx->internal_handle;
    if (bytes < 16) bytes = 16;
    eif_gpu_buf_t *b = (eif_gpu_buf_t *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->size = bytes;
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (s->vkCreateBuffer(s->device, &bi, NULL, &b->buf) != VK_SUCCESS) { free(b); return NULL; }
    VkMemoryRequirements req;
    s->vkGetBufferMemoryRequirements(s->device, b->buf, &req);

    VkMemoryPropertyFlags required = host_visible
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags preferred = host_visible ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0;
    /* first the preferred type, then any type meeting the requirement (e.g. small BAR heaps) */
    uint32_t types[2] = {
        pick_memory_type(&s->mem_props, req.memoryTypeBits, required, preferred),
        pick_memory_type(&s->mem_props, req.memoryTypeBits, required, 0),
    };
    VkResult r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    for (int i = 0; i < 2 && r != VK_SUCCESS; i++) {
        if (types[i] == UINT32_MAX || (i == 1 && types[1] == types[0])) continue;
        VkMemoryAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req.size,
            .memoryTypeIndex = types[i],
        };
        r = s->vkAllocateMemory(s->device, &ai, NULL, &b->mem);
    }
    if (r != VK_SUCCESS || s->vkBindBufferMemory(s->device, b->buf, b->mem, 0) != VK_SUCCESS ||
        (host_visible && s->vkMapMemory(s->device, b->mem, 0, VK_WHOLE_SIZE, 0, &b->mapped) != VK_SUCCESS)) {
        if (r == VK_SUCCESS) s->vkFreeMemory(s->device, b->mem, NULL);
        s->vkDestroyBuffer(s->device, b->buf, NULL);
        free(b);
        return NULL;
    }
    return b;
}

void eif_gpu_buf_destroy(eif_gpu_context_t *ctx, eif_gpu_buf_t *b)
{
    if (!b) return;
    if (ctx && ctx->internal_handle) {
        vk_state_t *s = (vk_state_t *)ctx->internal_handle;
        if (b->mapped) s->vkUnmapMemory(s->device, b->mem);
        s->vkDestroyBuffer(s->device, b->buf, NULL);
        s->vkFreeMemory(s->device, b->mem, NULL);
    }
    free(b);
}

void *eif_gpu_buf_map(eif_gpu_buf_t *b) { return b ? b->mapped : NULL; }

size_t eif_gpu_max_binding_bytes(const eif_gpu_context_t *ctx)
{
    if (!ctx || !ctx->internal_handle) return 0;
    return (size_t)((const vk_state_t *)ctx->internal_handle)->max_range;
}

static bool vk_begin_cmd(vk_state_t *s)
{
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (s->vkResetCommandBuffer(s->cmd, 0) != VK_SUCCESS) return false;
    return s->vkBeginCommandBuffer(s->cmd, &bi) == VK_SUCCESS;
}

static bool vk_submit_wait(vk_state_t *s)
{
    if (s->vkEndCommandBuffer(s->cmd) != VK_SUCCESS) return false;
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &s->cmd,
    };
    if (s->vkQueueSubmit(s->queue, 1, &si, s->fence) != VK_SUCCESS) return false;
    VkResult r = s->vkWaitForFences(s->device, 1, &s->fence, VK_TRUE, UINT64_MAX);
    s->vkResetFences(s->device, 1, &s->fence);
    return r == VK_SUCCESS;
}

bool eif_gpu_buf_upload(eif_gpu_context_t *ctx, eif_gpu_buf_t *b, size_t offset, const void *src, size_t bytes)
{
    if (!ctx || !ctx->internal_handle || !b || !src || offset + bytes > b->size) return false;
    vk_state_t *s = (vk_state_t *)ctx->internal_handle;
    if (b->mapped) {
        memcpy((uint8_t *)b->mapped + offset, src, bytes);
        return true;
    }
    if (s->recording) return false;
    if (!s->staging) {
        s->staging = eif_gpu_buf_create(ctx, EIF_VK_STAGING_BYTES, true);
        if (!s->staging) return false;
    }
    for (size_t done = 0; done < bytes;) {
        size_t n = bytes - done < EIF_VK_STAGING_BYTES ? bytes - done : EIF_VK_STAGING_BYTES;
        memcpy(s->staging->mapped, (const uint8_t *)src + done, n);
        VkBufferCopy region = { .srcOffset = 0, .dstOffset = offset + done, .size = n };
        if (!vk_begin_cmd(s)) return false;
        s->vkCmdCopyBuffer(s->cmd, s->staging->buf, b->buf, 1, &region);
        if (!vk_submit_wait(s)) return false;
        done += n;
    }
    return true;
}

bool eif_gpu_cmd_begin(eif_gpu_context_t *ctx)
{
    if (!ctx || !ctx->is_available || !ctx->internal_handle) return false;
    vk_state_t *s = (vk_state_t *)ctx->internal_handle;
    if (s->recording) return false;
    if (s->vkResetDescriptorPool(s->device, s->dpool, 0) != VK_SUCCESS || !vk_begin_cmd(s)) return false;
    s->recording = true;
    s->failed = false;
    return true;
}

bool eif_gpu_cmd_dispatch(eif_gpu_context_t *ctx, int kernel, eif_gpu_buf_t *const bufs[5],
                          const void *push, uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz)
{
    vk_state_t *s = (vk_state_t *)ctx->internal_handle;
    if (!s->recording || s->failed) return false;
    if (kernel < 0 || kernel >= EIF_GPU_K_COUNT || push_bytes > 128 || gx == 0 || gy == 0 || gz == 0 ||
        gx > s->max_groups[0] || gy > s->max_groups[1] || gz > s->max_groups[2]) {
        s->failed = true;
        return false;
    }
    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = s->dpool,
        .descriptorSetCount = 1,
        .pSetLayouts = &s->dsl,
    };
    if (s->vkAllocateDescriptorSets(s->device, &ai, &set) != VK_SUCCESS) {
        s->failed = true;
        return false;
    }
    VkDescriptorBufferInfo bi[EIF_VK_NBIND];
    VkWriteDescriptorSet w[EIF_VK_NBIND];
    for (int i = 0; i < EIF_VK_NBIND; i++) {
        const eif_gpu_buf_t *b = (bufs && bufs[i]) ? bufs[i] : s->dummy;
        if (b->size > s->max_range) { s->failed = true; return false; }
        bi[i].buffer = b->buf;
        bi[i].offset = 0;
        bi[i].range = VK_WHOLE_SIZE;
        memset(&w[i], 0, sizeof(w[i]));
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set;
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    s->vkUpdateDescriptorSets(s->device, EIF_VK_NBIND, w, 0, NULL);
    s->vkCmdBindPipeline(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipes[kernel]);
    s->vkCmdBindDescriptorSets(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->pl, 0, 1, &set, 0, NULL);
    if (push_bytes) s->vkCmdPushConstants(s->cmd, s->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    s->vkCmdDispatch(s->cmd, gx, gy, gz);
    return true;
}

bool eif_gpu_cmd_gemm(eif_gpu_context_t *ctx, int kernel, eif_gpu_buf_t *A, eif_gpu_buf_t *B, eif_gpu_buf_t *S,
                      eif_gpu_buf_t *bias, eif_gpu_buf_t *C, const eif_gpu_gemm_pc_t *pc, uint32_t gz)
{
    /* 128x128 tiles (8x8 outputs per thread) when the grid still fills the GPU, 64x64 otherwise */
    bool big = (kernel == EIF_GPU_K_GEMM_F32 || kernel == EIF_GPU_K_GEMM_Q8) &&
               (uint64_t)((pc->M + 127) / 128) * ((pc->N + 127) / 128) * gz >= 96;
    if (big) kernel = (kernel == EIF_GPU_K_GEMM_F32) ? EIF_GPU_K_GEMM_F32_BIG : EIF_GPU_K_GEMM_Q8_BIG;
    uint32_t tile = big ? 128 : 64;
    eif_gpu_buf_t *bufs[5] = {A, B, S, bias, C};
    return eif_gpu_cmd_dispatch(ctx, kernel, bufs, pc, sizeof(*pc), (pc->N + tile - 1) / tile,
                                (pc->M + tile - 1) / tile, gz);
}

void eif_gpu_cmd_barrier(eif_gpu_context_t *ctx)
{
    vk_state_t *s = (vk_state_t *)ctx->internal_handle;
    if (!s->recording) return;
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    s->vkCmdPipelineBarrier(s->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &mb, 0, NULL, 0, NULL);
}

bool eif_gpu_cmd_submit(eif_gpu_context_t *ctx)
{
    vk_state_t *s = (vk_state_t *)ctx->internal_handle;
    if (!s->recording) return false;
    s->recording = false;
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    s->vkCmdPipelineBarrier(s->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                            0, 1, &mb, 0, NULL, 0, NULL);
    bool ok = vk_submit_wait(s);
    return ok && !s->failed;
}

static bool vk_load_device_funcs(vk_state_t *s)
{
#define X(n) if (!(s->n = (PFN_##n)s->vkGetDeviceProcAddr(s->device, #n))) return false;
    EIF_VK_DEVICE_FUNCS(X)
#undef X
    return true;
}

static bool vk_create_pipelines(vk_state_t *s)
{
    static const struct { const uint32_t *code; size_t size; } spv[EIF_GPU_K_COUNT] = {
        [EIF_GPU_K_GEMM_F32] = {eif_spv_gemm_f32, sizeof(eif_spv_gemm_f32)},
        [EIF_GPU_K_GEMM_Q8] = {eif_spv_gemm_q8, sizeof(eif_spv_gemm_q8)},
        [EIF_GPU_K_GEMM_KN] = {eif_spv_gemm_kn, sizeof(eif_spv_gemm_kn)},
        [EIF_GPU_K_GEMM_F32_BIG] = {eif_spv_gemm_f32_big, sizeof(eif_spv_gemm_f32_big)},
        [EIF_GPU_K_GEMM_Q8_BIG] = {eif_spv_gemm_q8_big, sizeof(eif_spv_gemm_q8_big)},
        [EIF_GPU_K_LAYERNORM] = {eif_spv_layernorm, sizeof(eif_spv_layernorm)},
        [EIF_GPU_K_SOFTMAX] = {eif_spv_softmax, sizeof(eif_spv_softmax)},
        [EIF_GPU_K_BCAST] = {eif_spv_bcast, sizeof(eif_spv_bcast)},
        [EIF_GPU_K_MERGE] = {eif_spv_merge, sizeof(eif_spv_merge)},
    };
    VkDescriptorSetLayoutBinding b[EIF_VK_NBIND];
    for (int i = 0; i < EIF_VK_NBIND; i++) {
        b[i].binding = (uint32_t)i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo dli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = EIF_VK_NBIND,
        .pBindings = b,
    };
    if (s->vkCreateDescriptorSetLayout(s->device, &dli, NULL, &s->dsl) != VK_SUCCESS) return false;
    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 128 };
    VkPipelineLayoutCreateInfo pli = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &s->dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    if (s->vkCreatePipelineLayout(s->device, &pli, NULL, &s->pl) != VK_SUCCESS) return false;
    for (int k = 0; k < EIF_GPU_K_COUNT; k++) {
        VkShaderModuleCreateInfo smi = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spv[k].size,
            .pCode = spv[k].code,
        };
        VkShaderModule mod;
        if (s->vkCreateShaderModule(s->device, &smi, NULL, &mod) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = mod,
                .pName = "main",
            },
            .layout = s->pl,
        };
        VkResult r = s->vkCreateComputePipelines(s->device, VK_NULL_HANDLE, 1, &ci, NULL, &s->pipes[k]);
        s->vkDestroyShaderModule(s->device, mod, NULL);
        if (r != VK_SUCCESS) return false;
    }
    VkDescriptorPoolSize ps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = EIF_VK_MAX_SETS * EIF_VK_NBIND,
    };
    VkDescriptorPoolCreateInfo dpi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = EIF_VK_MAX_SETS,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    return s->vkCreateDescriptorPool(s->device, &dpi, NULL, &s->dpool) == VK_SUCCESS;
}

static void vk_destroy_state(vk_state_t *s)
{
    if (!s) return;
    if (s->device && s->vkDestroyDevice && s->vkResetFences) { /* all device functions were loaded */
        s->vkDeviceWaitIdle(s->device);
        eif_gpu_context_t tmp;
        memset(&tmp, 0, sizeof(tmp));
        tmp.internal_handle = s;
        for (int i = 0; i < 5; i++) eif_gpu_buf_destroy(&tmp, s->scr[i]);
        eif_gpu_buf_destroy(&tmp, s->staging);
        eif_gpu_buf_destroy(&tmp, s->dummy);
        if (s->dpool) s->vkDestroyDescriptorPool(s->device, s->dpool, NULL);
        for (int k = 0; k < EIF_GPU_K_COUNT; k++)
            if (s->pipes[k]) s->vkDestroyPipeline(s->device, s->pipes[k], NULL);
        if (s->pl) s->vkDestroyPipelineLayout(s->device, s->pl, NULL);
        if (s->dsl) s->vkDestroyDescriptorSetLayout(s->device, s->dsl, NULL);
        if (s->fence) s->vkDestroyFence(s->device, s->fence, NULL);
        if (s->cmd_pool) s->vkDestroyCommandPool(s->device, s->cmd_pool, NULL);
        s->vkDestroyDevice(s->device, NULL);
    } else if (s->device && s->vkDestroyDevice) {
        s->vkDestroyDevice(s->device, NULL);
    }
    if (s->instance && s->vkDestroyInstance) s->vkDestroyInstance(s->instance, NULL);
    vk_close_loader(s->lib);
    free(s);
}

static int device_rank(VkPhysicalDeviceType t, bool allow_cpu)
{
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return allow_cpu ? 1 : 0;
    default: return 1;
    }
}

static bool vk_init_state(eif_gpu_context_t *ctx, vk_state_t *s)
{
    s->lib = vk_open_loader();
    if (!s->lib) {
        snprintf(ctx->device_name, sizeof(ctx->device_name), "Unavailable (Vulkan loader not found)");
        return false;
    }
    s->vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)vk_sym(s->lib, "vkGetInstanceProcAddr");
    if (!s->vkGetInstanceProcAddr) return false;
    PFN_vkCreateInstance create_instance =
        (PFN_vkCreateInstance)s->vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create_instance) return false;

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "bitllm.c",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "bitllm.c",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    if (create_instance(&ici, NULL, &s->instance) != VK_SUCCESS) {
        s->instance = VK_NULL_HANDLE;
        snprintf(ctx->device_name, sizeof(ctx->device_name), "Unavailable (vkCreateInstance failed)");
        return false;
    }
#define X(n) if (!(s->n = (PFN_##n)s->vkGetInstanceProcAddr(s->instance, #n))) return false;
    EIF_VK_INSTANCE_FUNCS(X)
#undef X

    uint32_t ndev = 0;
    if (s->vkEnumeratePhysicalDevices(s->instance, &ndev, NULL) != VK_SUCCESS || ndev == 0) {
        snprintf(ctx->device_name, sizeof(ctx->device_name), "No Vulkan physical devices found");
        return false;
    }
    VkPhysicalDevice devs[16];
    if (ndev > 16) ndev = 16;
    VkResult er = s->vkEnumeratePhysicalDevices(s->instance, &ndev, devs);
    if (er != VK_SUCCESS && er != VK_INCOMPLETE) return false;

    const bool allow_cpu = env_on("BITLLM_GPU_ALLOW_CPU");
    int best_rank = 0;
    uint32_t qfam = 0;
    VkPhysicalDeviceProperties props;
    memset(&props, 0, sizeof(props));
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties p;
        s->vkGetPhysicalDeviceProperties(devs[i], &p);
        int rank = device_rank(p.deviceType, allow_cpu);
        if (rank <= best_rank) continue;
        uint32_t nq = 0;
        s->vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, NULL);
        VkQueueFamilyProperties qp[16];
        if (nq > 16) nq = 16;
        s->vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, qp);
        for (uint32_t q = 0; q < nq; q++) {
            if (qp[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                best_rank = rank;
                s->phys = devs[i];
                qfam = q;
                props = p;
                break;
            }
        }
    }
    if (!s->phys) {
        snprintf(ctx->device_name, sizeof(ctx->device_name), "No suitable Vulkan compute device");
        return false;
    }
    snprintf(ctx->device_name, sizeof(ctx->device_name), "%s", props.deviceName);
    s->max_range = props.limits.maxStorageBufferRange;
    for (int i = 0; i < 3; i++) s->max_groups[i] = props.limits.maxComputeWorkGroupCount[i];
    if (props.limits.maxPushConstantsSize < 128 || props.limits.maxComputeWorkGroupInvocations < 256 ||
        props.limits.maxComputeSharedMemorySize < 16384)
        return false;
    s->vkGetPhysicalDeviceMemoryProperties(s->phys, &s->mem_props);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfam,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo di = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qi,
    };
    if (s->vkCreateDevice(s->phys, &di, NULL, &s->device) != VK_SUCCESS) {
        s->device = VK_NULL_HANDLE;
        return false;
    }
    if (!vk_load_device_funcs(s)) return false;
    s->vkGetDeviceQueue(s->device, qfam, 0, &s->queue);

    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = qfam,
    };
    if (s->vkCreateCommandPool(s->device, &cpi, NULL, &s->cmd_pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = s->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (s->vkAllocateCommandBuffers(s->device, &cai, &s->cmd) != VK_SUCCESS) return false;
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (s->vkCreateFence(s->device, &fi, NULL, &s->fence) != VK_SUCCESS) return false;
    if (!vk_create_pipelines(s)) return false;

    ctx->internal_handle = s;
    s->dummy = eif_gpu_buf_create(ctx, 256, false);
    return s->dummy != NULL;
}

bool eif_gpu_init(eif_gpu_context_t *ctx)
{
    if (!ctx) return false;
    memset(ctx, 0, sizeof(*ctx));
    snprintf(ctx->backend_name, sizeof(ctx->backend_name), "Vulkan Compute");
    if (env_on("BITLLM_NO_GPU")) {
        snprintf(ctx->device_name, sizeof(ctx->device_name), "Disabled (BITLLM_NO_GPU)");
        return false;
    }
    vk_state_t *s = (vk_state_t *)calloc(1, sizeof(vk_state_t));
    if (!s) return false;
    if (!vk_init_state(ctx, s)) {
        vk_destroy_state(s);
        ctx->internal_handle = NULL;
        ctx->is_available = false;
        return false;
    }
    ctx->is_available = true;
    return true;
}

void eif_gpu_cleanup(eif_gpu_context_t *ctx)
{
    if (!ctx || !ctx->internal_handle) return;
    vk_destroy_state((vk_state_t *)ctx->internal_handle);
    ctx->internal_handle = NULL;
    ctx->is_available = false;
}

/* ---- generic GEMM entry points ------------------------------------------------------------- */

static bool ensure_scratch(eif_gpu_context_t *ctx, int slot, size_t bytes)
{
    vk_state_t *s = (vk_state_t *)ctx->internal_handle;
    if (s->scr[slot] && s->scr[slot]->size >= bytes) return true;
    eif_gpu_buf_destroy(ctx, s->scr[slot]);
    s->scr[slot] = eif_gpu_buf_create(ctx, bytes, true);
    return s->scr[slot] != NULL;
}

static bool gemm_valid(int M, int N, int K)
{
    return M > 0 && N > 0 && K > 0 && (uint64_t)M * K < (1ull << 32) && (uint64_t)N * K < (1ull << 32) &&
           (uint64_t)M * N < (1ull << 32);
}

/* wbuf is a resident weight buffer or the FP32 W scratch slot. */
static bool run_gemm(eif_gpu_context_t *ctx, const float *A, eif_gpu_buf_t *wbuf, bool q8, uint32_t s_off,
                     const float *bias, float *C, int M, int N, int K, int act, bool residual)
{
    vk_state_t *s = (vk_state_t *)ctx->internal_handle;
    const size_t sa = (size_t)M * K * sizeof(float), sc = (size_t)M * N * sizeof(float);
    if (!ensure_scratch(ctx, 0, sa) || !ensure_scratch(ctx, 4, sc) ||
        (bias && !ensure_scratch(ctx, 3, (size_t)N * sizeof(float))))
        return false;
    memcpy(s->scr[0]->mapped, A, sa);
    if (bias) memcpy(s->scr[3]->mapped, bias, (size_t)N * sizeof(float));
    if (residual) memcpy(s->scr[4]->mapped, C, sc);
    eif_gpu_gemm_pc_t pc;
    memset(&pc, 0, sizeof(pc));
    pc.M = (uint32_t)M; pc.N = (uint32_t)N; pc.K = (uint32_t)K;
    pc.lda = (uint32_t)K; pc.ldb = (uint32_t)K; pc.ldc = (uint32_t)N;
    pc.s_off = s_off;
    pc.flags = (bias ? EIF_GPU_GEMM_BIAS : 0) | (act == 1 ? EIF_GPU_GEMM_GELU : 0) |
               (residual ? EIF_GPU_GEMM_RESIDUAL : 0) | (q8 ? EIF_GPU_GEMM_SCALES : 0);
    pc.zdiv = 1;
    pc.alpha = 1.0f;
    if (!eif_gpu_cmd_begin(ctx)) return false;
    eif_gpu_cmd_gemm(ctx, q8 ? EIF_GPU_K_GEMM_Q8 : EIF_GPU_K_GEMM_F32, s->scr[0], wbuf, q8 ? wbuf : NULL,
                     bias ? s->scr[3] : NULL, s->scr[4], &pc, 1);
    if (!eif_gpu_cmd_submit(ctx)) return false;
    memcpy(C, s->scr[4]->mapped, sc);
    return true;
}

bool eif_gpu_gemm_f32(eif_gpu_context_t *ctx, const float *A, const float *W, const float *bias, float *C,
                      int M, int N, int K, int activation_mode, bool residual_add)
{
    if (!ctx || !ctx->is_available || !ctx->internal_handle || !A || !W || !C || !gemm_valid(M, N, K))
        return false;
    vk_state_t *s = (vk_state_t *)ctx->internal_handle;
    const size_t sw = (size_t)N * K * sizeof(float);
    if (!ensure_scratch(ctx, 1, sw)) return false;
    memcpy(s->scr[1]->mapped, W, sw);
    return run_gemm(ctx, A, s->scr[1], false, 0, bias, C, M, N, K, activation_mode, residual_add);
}

eif_gpu_weight_t *eif_gpu_weight_upload_f32(eif_gpu_context_t *ctx, const float *W, int N, int K)
{
    if (!ctx || !ctx->is_available || !W || !gemm_valid(1, N, K)) return NULL;
    eif_gpu_weight_t *w = (eif_gpu_weight_t *)calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->N = N; w->K = K;
    const size_t bytes = (size_t)N * K * sizeof(float);
    w->buf = eif_gpu_buf_create(ctx, bytes, false);
    if (!w->buf || !eif_gpu_buf_upload(ctx, w->buf, 0, W, bytes)) {
        eif_gpu_weight_free(ctx, w);
        return NULL;
    }
    return w;
}

eif_gpu_weight_t *eif_gpu_weight_upload_q8(eif_gpu_context_t *ctx, const int8_t *codes, const float *scales,
                                           int N, int K)
{
    if (!ctx || !ctx->is_available || !codes || !scales || !gemm_valid(1, N, K)) return NULL;
    eif_gpu_weight_t *w = (eif_gpu_weight_t *)calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->N = N; w->K = K; w->q8 = true;
    const size_t cbytes = ((size_t)N * K + 15) & ~(size_t)15;
    w->s_off = (uint32_t)(cbytes / sizeof(float));
    w->buf = eif_gpu_buf_create(ctx, cbytes + (size_t)N * sizeof(float), false);
    if (!w->buf || !eif_gpu_buf_upload(ctx, w->buf, 0, codes, (size_t)N * K) ||
        !eif_gpu_buf_upload(ctx, w->buf, cbytes, scales, (size_t)N * sizeof(float))) {
        eif_gpu_weight_free(ctx, w);
        return NULL;
    }
    return w;
}

void eif_gpu_weight_free(eif_gpu_context_t *ctx, eif_gpu_weight_t *w)
{
    if (!w) return;
    eif_gpu_buf_destroy(ctx, w->buf);
    free(w);
}

bool eif_gpu_gemm(eif_gpu_context_t *ctx, const float *A, const eif_gpu_weight_t *W, const float *bias,
                  float *C, int M, int activation_mode, bool residual_add)
{
    if (!ctx || !ctx->is_available || !ctx->internal_handle || !A || !W || !C || !gemm_valid(M, W->N, W->K))
        return false;
    return run_gemm(ctx, A, W->buf, W->q8, W->s_off, bias, C, M, W->N, W->K, activation_mode, residual_add);
}

#else /* !EIF_USE_VULKAN: CPU only */

bool eif_gpu_init(eif_gpu_context_t *ctx)
{
    if (!ctx) return false;
    memset(ctx, 0, sizeof(*ctx));
    snprintf(ctx->backend_name, sizeof(ctx->backend_name), "Disabled (CPU Native)");
    snprintf(ctx->device_name, sizeof(ctx->device_name), "Host CPU (built without Vulkan)");
    return false;
}

void eif_gpu_cleanup(eif_gpu_context_t *ctx)
{
    if (!ctx) return;
    ctx->is_available = false;
    ctx->internal_handle = NULL;
}

bool eif_gpu_gemm_f32(eif_gpu_context_t *ctx, const float *A, const float *W, const float *bias, float *C,
                      int M, int N, int K, int activation_mode, bool residual_add)
{
    (void)ctx; (void)A; (void)W; (void)bias; (void)C; (void)M; (void)N; (void)K;
    (void)activation_mode; (void)residual_add;
    return false;
}

eif_gpu_weight_t *eif_gpu_weight_upload_f32(eif_gpu_context_t *ctx, const float *W, int N, int K)
{
    (void)ctx; (void)W; (void)N; (void)K;
    return NULL;
}

eif_gpu_weight_t *eif_gpu_weight_upload_q8(eif_gpu_context_t *ctx, const int8_t *codes, const float *scales,
                                           int N, int K)
{
    (void)ctx; (void)codes; (void)scales; (void)N; (void)K;
    return NULL;
}

void eif_gpu_weight_free(eif_gpu_context_t *ctx, eif_gpu_weight_t *w) { (void)ctx; (void)w; }

bool eif_gpu_gemm(eif_gpu_context_t *ctx, const float *A, const eif_gpu_weight_t *W, const float *bias,
                  float *C, int M, int activation_mode, bool residual_add)
{
    (void)ctx; (void)A; (void)W; (void)bias; (void)C; (void)M; (void)activation_mode; (void)residual_add;
    return false;
}

eif_gpu_buf_t *eif_gpu_buf_create(eif_gpu_context_t *ctx, size_t bytes, bool host_visible)
{
    (void)ctx; (void)bytes; (void)host_visible;
    return NULL;
}
void eif_gpu_buf_destroy(eif_gpu_context_t *ctx, eif_gpu_buf_t *b) { (void)ctx; (void)b; }
void *eif_gpu_buf_map(eif_gpu_buf_t *b) { (void)b; return NULL; }
bool eif_gpu_buf_upload(eif_gpu_context_t *ctx, eif_gpu_buf_t *b, size_t offset, const void *src, size_t bytes)
{
    (void)ctx; (void)b; (void)offset; (void)src; (void)bytes;
    return false;
}
size_t eif_gpu_max_binding_bytes(const eif_gpu_context_t *ctx) { (void)ctx; return 0; }
bool eif_gpu_cmd_begin(eif_gpu_context_t *ctx) { (void)ctx; return false; }
bool eif_gpu_cmd_dispatch(eif_gpu_context_t *ctx, int kernel, eif_gpu_buf_t *const bufs[5],
                          const void *push, uint32_t push_bytes, uint32_t gx, uint32_t gy, uint32_t gz)
{
    (void)ctx; (void)kernel; (void)bufs; (void)push; (void)push_bytes; (void)gx; (void)gy; (void)gz;
    return false;
}
bool eif_gpu_cmd_gemm(eif_gpu_context_t *ctx, int kernel, eif_gpu_buf_t *A, eif_gpu_buf_t *B, eif_gpu_buf_t *S,
                      eif_gpu_buf_t *bias, eif_gpu_buf_t *C, const eif_gpu_gemm_pc_t *pc, uint32_t gz)
{
    (void)ctx; (void)kernel; (void)A; (void)B; (void)S; (void)bias; (void)C; (void)pc; (void)gz;
    return false;
}
void eif_gpu_cmd_barrier(eif_gpu_context_t *ctx) { (void)ctx; }
bool eif_gpu_cmd_submit(eif_gpu_context_t *ctx) { (void)ctx; return false; }

#endif /* EIF_USE_VULKAN */

bool eif_gpu_matmul_bitnet_f32(eif_gpu_context_t *ctx, const uint8_t *weights, const float *scales,
                               const float *input, const float *bias, float *output, int rows, int cols)
{
    (void)ctx; (void)weights; (void)scales; (void)input; (void)bias; (void)output; (void)rows; (void)cols;
    return false; /* ternary LM matmuls stay on the CPU kernels */
}

bool eif_gpu_is_available(const eif_gpu_context_t *ctx)
{
    return (ctx != NULL) && ctx->is_available;
}
