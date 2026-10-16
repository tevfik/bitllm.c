/**
 * @file eif_gpu.c
 * @brief EIF Optional GPU Compute Backend Implementation (Vulkan Compute)
 */

#include "eif_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(EIF_USE_VULKAN)
#include "vulkan/vulkan.h"
#include "vulkan_gemm_spv.h"

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t queue_family_index;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    /* GEMM Pipeline */
    VkShaderModule gemm_module;
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipe_layout;
    VkPipeline pipeline;
    VkDescriptorPool desc_pool;
    VkDescriptorSet desc_set;

    /* GPU Buffers */
    VkPhysicalDeviceMemoryProperties mem_props;
    VkBuffer buf_a;
    VkDeviceMemory mem_a;
    size_t cap_a;
    float *mapped_a;

    VkBuffer buf_w;
    VkDeviceMemory mem_w;
    size_t cap_w;
    float *mapped_w;

    VkBuffer buf_b;
    VkDeviceMemory mem_b;
    size_t cap_b;
    float *mapped_b;

    VkBuffer buf_c;
    VkDeviceMemory mem_c;
    size_t cap_c;
    float *mapped_c;
} eif_vulkan_state_t;

static uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties *mem_props,
                                 uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props->memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0;
}

static bool create_gpu_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties *mem_props,
                              size_t size, VkBuffer *out_buf, VkDeviceMemory *out_mem, float **out_mapped)
{
    VkBufferCreateInfo b_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(device, &b_info, NULL, out_buf) != VK_SUCCESS) return false;

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, *out_buf, &reqs);

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = reqs.size,
        .memoryTypeIndex = find_memory_type(mem_props, reqs.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    if (vkAllocateMemory(device, &alloc_info, NULL, out_mem) != VK_SUCCESS) {
        vkDestroyBuffer(device, *out_buf, NULL);
        return false;
    }

    if (vkBindBufferMemory(device, *out_buf, *out_mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device, *out_buf, NULL);
        vkFreeMemory(device, *out_mem, NULL);
        return false;
    }

    if (vkMapMemory(device, *out_mem, 0, size, 0, (void **)out_mapped) != VK_SUCCESS) {
        vkDestroyBuffer(device, *out_buf, NULL);
        vkFreeMemory(device, *out_mem, NULL);
        return false;
    }

    return true;
}

bool eif_gpu_init(eif_gpu_context_t *ctx)
{
    if (!ctx) return false;
    memset(ctx, 0, sizeof(*ctx));
    snprintf(ctx->backend_name, sizeof(ctx->backend_name), "Vulkan Compute");

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "EIF Edge Runtime",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "EIF Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_1,
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = vkCreateInstance(&create_info, NULL, &instance);
    if (res != VK_SUCCESS) {
        snprintf(ctx->device_name, sizeof(ctx->device_name), "Unavailable (vkCreateInstance failed)");
        ctx->is_available = false;
        return false;
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    if (device_count == 0) {
        vkDestroyInstance(instance, NULL);
        snprintf(ctx->device_name, sizeof(ctx->device_name), "No Vulkan physical devices found");
        ctx->is_available = false;
        return false;
    }

    VkPhysicalDevice *devices = (VkPhysicalDevice *)malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &device_count, devices);

    /* Prefer dedicated/integrated discrete GPU over CPU llvmpipe fallback */
    VkPhysicalDevice physical_device = devices[0];
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties dev_props;
        vkGetPhysicalDeviceProperties(devices[i], &dev_props);
        if (dev_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
            dev_props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            physical_device = devices[i];
            break;
        }
    }
    free(devices);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device, &props);
    snprintf(ctx->device_name, sizeof(ctx->device_name), "%s", props.deviceName);

    /* Locate compute queue family */
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, NULL);
    VkQueueFamilyProperties *qf_props = (VkQueueFamilyProperties *)malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, qf_props);

    uint32_t compute_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }
    free(qf_props);

    if (compute_queue_family == UINT32_MAX) {
        vkDestroyInstance(instance, NULL);
        ctx->is_available = false;
        return false;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };

    VkDevice device = VK_NULL_HANDLE;
    res = vkCreateDevice(physical_device, &dev_info, NULL, &device);
    if (res != VK_SUCCESS) {
        vkDestroyInstance(instance, NULL);
        ctx->is_available = false;
        return false;
    }

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, compute_queue_family, 0, &queue);

    eif_vulkan_state_t *state = (eif_vulkan_state_t *)calloc(1, sizeof(eif_vulkan_state_t));
    if (!state) {
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
        return false;
    }
    state->instance = instance;
    state->physical_device = physical_device;
    state->device = device;
    state->compute_queue = queue;
    state->queue_family_index = compute_queue_family;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &state->mem_props);

    /* Command Pool & Buffer */
    VkCommandPoolCreateInfo cp_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    if (vkCreateCommandPool(device, &cp_info, NULL, &state->command_pool) != VK_SUCCESS) {
        eif_gpu_cleanup(ctx);
        return false;
    }

    VkCommandBufferAllocateInfo cb_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = state->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(device, &cb_alloc, &state->command_buffer) != VK_SUCCESS) {
        eif_gpu_cleanup(ctx);
        return false;
    }

    /* Compile Shader Module */
    VkShaderModuleCreateInfo sm_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_gemm_spv_size,
        .pCode = vulkan_gemm_spv,
    };
    if (vkCreateShaderModule(device, &sm_info, NULL, &state->gemm_module) != VK_SUCCESS) {
        eif_gpu_cleanup(ctx);
        return false;
    }

    /* Descriptor Set Layout */
    VkDescriptorSetLayoutBinding bindings[4];
    for (int i = 0; i < 4; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = NULL;
    }
    VkDescriptorSetLayoutCreateInfo dsl_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(device, &dsl_info, NULL, &state->desc_layout) != VK_SUCCESS) {
        eif_gpu_cleanup(ctx);
        return false;
    }

    /* Pipeline Layout with Push Constants */
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 6 * sizeof(int),
    };
    VkPipelineLayoutCreateInfo pl_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &state->desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    if (vkCreatePipelineLayout(device, &pl_info, NULL, &state->pipe_layout) != VK_SUCCESS) {
        eif_gpu_cleanup(ctx);
        return false;
    }

    /* Compute Pipeline */
    VkComputePipelineCreateInfo pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = state->gemm_module,
            .pName = "main",
        },
        .layout = state->pipe_layout,
    };
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &state->pipeline) != VK_SUCCESS) {
        eif_gpu_cleanup(ctx);
        return false;
    }

    /* Descriptor Pool & Set */
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4,
    };
    VkDescriptorPoolCreateInfo dp_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    if (vkCreateDescriptorPool(device, &dp_info, NULL, &state->desc_pool) != VK_SUCCESS) {
        eif_gpu_cleanup(ctx);
        return false;
    }

    VkDescriptorSetAllocateInfo ds_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = state->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &state->desc_layout,
    };
    if (vkAllocateDescriptorSets(device, &ds_alloc, &state->desc_set) != VK_SUCCESS) {
        eif_gpu_cleanup(ctx);
        return false;
    }

    /* Preallocate persistent storage buffers for ViT layers:
     * A: [1024, 3072] = ~12.5 MB
     * W: [3072, 3072] = ~37.7 MB
     * Bias: [3072]    = ~12 KB
     * C: [1024, 3072] = ~12.5 MB
     */
    state->cap_a = 1024 * 3072 * sizeof(float);
    state->cap_w = 3072 * 3072 * sizeof(float);
    state->cap_b = 3072 * sizeof(float);
    state->cap_c = 1024 * 3072 * sizeof(float);

    if (!create_gpu_buffer(device, &state->mem_props, state->cap_a, &state->buf_a, &state->mem_a, &state->mapped_a) ||
        !create_gpu_buffer(device, &state->mem_props, state->cap_w, &state->buf_w, &state->mem_w, &state->mapped_w) ||
        !create_gpu_buffer(device, &state->mem_props, state->cap_b, &state->buf_b, &state->mem_b, &state->mapped_b) ||
        !create_gpu_buffer(device, &state->mem_props, state->cap_c, &state->buf_c, &state->mem_c, &state->mapped_c)) {
        eif_gpu_cleanup(ctx);
        return false;
    }

    /* Bind buffers to Descriptor Set */
    VkDescriptorBufferInfo b_infos[4] = {
        { .buffer = state->buf_a, .offset = 0, .range = state->cap_a },
        { .buffer = state->buf_w, .offset = 0, .range = state->cap_w },
        { .buffer = state->buf_b, .offset = 0, .range = state->cap_b },
        { .buffer = state->buf_c, .offset = 0, .range = state->cap_c },
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = NULL;
        writes[i].dstSet = state->desc_set;
        writes[i].dstBinding = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pImageInfo = NULL;
        writes[i].pBufferInfo = &b_infos[i];
        writes[i].pTexelBufferView = NULL;
    }
    vkUpdateDescriptorSets(device, 4, writes, 0, NULL);

    ctx->internal_handle = state;
    ctx->is_available = true;
    return true;
}

void eif_gpu_cleanup(eif_gpu_context_t *ctx)
{
    if (!ctx || !ctx->internal_handle) return;
    eif_vulkan_state_t *s = (eif_vulkan_state_t *)ctx->internal_handle;
    if (s->device) {
        vkDeviceWaitIdle(s->device);

        if (s->mapped_a) vkUnmapMemory(s->device, s->mem_a);
        if (s->mapped_w) vkUnmapMemory(s->device, s->mem_w);
        if (s->mapped_b) vkUnmapMemory(s->device, s->mem_b);
        if (s->mapped_c) vkUnmapMemory(s->device, s->mem_c);

        if (s->buf_a) vkDestroyBuffer(s->device, s->buf_a, NULL);
        if (s->buf_w) vkDestroyBuffer(s->device, s->buf_w, NULL);
        if (s->buf_b) vkDestroyBuffer(s->device, s->buf_b, NULL);
        if (s->buf_c) vkDestroyBuffer(s->device, s->buf_c, NULL);

        if (s->mem_a) vkFreeMemory(s->device, s->mem_a, NULL);
        if (s->mem_w) vkFreeMemory(s->device, s->mem_w, NULL);
        if (s->mem_b) vkFreeMemory(s->device, s->mem_b, NULL);
        if (s->mem_c) vkFreeMemory(s->device, s->mem_c, NULL);

        if (s->desc_pool) vkDestroyDescriptorPool(s->device, s->desc_pool, NULL);
        if (s->pipeline) vkDestroyPipeline(s->device, s->pipeline, NULL);
        if (s->pipe_layout) vkDestroyPipelineLayout(s->device, s->pipe_layout, NULL);
        if (s->desc_layout) vkDestroyDescriptorSetLayout(s->device, s->desc_layout, NULL);
        if (s->gemm_module) vkDestroyShaderModule(s->device, s->gemm_module, NULL);

        if (s->command_pool) vkDestroyCommandPool(s->device, s->command_pool, NULL);
        vkDestroyDevice(s->device, NULL);
    }
    if (s->instance) vkDestroyInstance(s->instance, NULL);
    free(s);
    ctx->internal_handle = NULL;
    ctx->is_available = false;
}

bool eif_gpu_gemm_f32(eif_gpu_context_t *ctx,
                      const float *A,
                      const float *W,
                      const float *bias,
                      float *C,
                      int M, int N, int K,
                      int activation_mode,
                      bool residual_add)
{
    if (!ctx || !ctx->is_available || !ctx->internal_handle) return false;
    eif_vulkan_state_t *s = (eif_vulkan_state_t *)ctx->internal_handle;

    size_t sz_a = (size_t)M * K * sizeof(float);
    size_t sz_w = (size_t)N * K * sizeof(float);
    size_t sz_b = (size_t)N * sizeof(float);
    size_t sz_c = (size_t)M * N * sizeof(float);

    if (sz_a > s->cap_a || sz_w > s->cap_w || sz_b > s->cap_b || sz_c > s->cap_c) {
        return false; /* Falls back to CPU SIMD */
    }

    /* 1. Copy inputs to mapped GPU host-visible buffers */
    memcpy(s->mapped_a, A, sz_a);
    memcpy(s->mapped_w, W, sz_w);
    if (bias) {
        memcpy(s->mapped_b, bias, sz_b);
    } else {
        memset(s->mapped_b, 0, sz_b);
    }
    if (residual_add) {
        memcpy(s->mapped_c, C, sz_c);
    }

    /* 2. Dispatch compute shader */
    int push_constants[6] = {
        M, N, K,
        activation_mode,
        residual_add ? 1 : 0,
        bias ? 1 : 0
    };

    VkCommandBufferBeginInfo cb_begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(s->command_buffer, &cb_begin);
    vkCmdBindPipeline(s->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, s->pipeline);
    vkCmdBindDescriptorSets(s->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            s->pipe_layout, 0, 1, &s->desc_set, 0, NULL);
    vkCmdPushConstants(s->command_buffer, s->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_constants), push_constants);

    uint32_t group_x = (uint32_t)(N + 15) / 16;
    uint32_t group_y = (uint32_t)(M + 15) / 16;
    vkCmdDispatch(s->command_buffer, group_x, group_y, 1);
    vkEndCommandBuffer(s->command_buffer);

    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &s->command_buffer,
    };
    if (vkQueueSubmit(s->compute_queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) {
        return false;
    }
    vkQueueWaitIdle(s->compute_queue);

    /* 3. Read back output */
    memcpy(C, s->mapped_c, sz_c);
    return true;
}

bool eif_gpu_matmul_bitnet_f32(eif_gpu_context_t *ctx,
                               const uint8_t *weights,
                               const float *scales,
                               const float *input,
                               const float *bias,
                               float *output,
                               int rows,
                               int cols)
{
    (void)ctx; (void)weights; (void)scales; (void)input; (void)bias; (void)output; (void)rows; (void)cols;
    return false; /* Currently uses high-performance T-MAC / NEON kernel */
}

#else /* !defined(EIF_USE_VULKAN) -> Graceful CPU Fallback */

bool eif_gpu_init(eif_gpu_context_t *ctx)
{
    if (!ctx) return false;
    memset(ctx, 0, sizeof(*ctx));
    snprintf(ctx->backend_name, sizeof(ctx->backend_name), "Disabled (CPU Native)");
    snprintf(ctx->device_name, sizeof(ctx->device_name), "Host CPU (Zero GPU Dependency)");
    ctx->is_available = false;
    ctx->internal_handle = NULL;
    return false;
}

void eif_gpu_cleanup(eif_gpu_context_t *ctx)
{
    if (!ctx) return;
    ctx->is_available = false;
    ctx->internal_handle = NULL;
}

bool eif_gpu_gemm_f32(eif_gpu_context_t *ctx,
                      const float *A,
                      const float *W,
                      const float *bias,
                      float *C,
                      int M, int N, int K,
                      int activation_mode,
                      bool residual_add)
{
    (void)ctx; (void)A; (void)W; (void)bias; (void)C; (void)M; (void)N; (void)K;
    (void)activation_mode; (void)residual_add;
    return false;
}

bool eif_gpu_matmul_bitnet_f32(eif_gpu_context_t *ctx,
                               const uint8_t *weights,
                               const float *scales,
                               const float *input,
                               const float *bias,
                               float *output,
                               int rows,
                               int cols)
{
    (void)ctx; (void)weights; (void)scales; (void)input; (void)bias; (void)output; (void)rows; (void)cols;
    return false;
}

#endif /* EIF_USE_VULKAN */

bool eif_gpu_is_available(const eif_gpu_context_t *ctx)
{
    return (ctx != NULL) && ctx->is_available;
}
