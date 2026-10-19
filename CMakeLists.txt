cmake_minimum_required(VERSION 3.16)
project(bitllm VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        add_compile_options(-O3 -mavx2 -mfma -ffast-math -Wall -Wextra -Wno-unused-parameter)
    else()
        add_compile_options(-O3 -ffast-math -Wall -Wextra -Wno-unused-parameter)
    endif()
else()
    # Portable x86_64 release target: AVX2 + FMA prevents AVX-512 EVEX illegal instructions on non-AVX512 hosts
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        add_compile_options(-O3 -mavx2 -mfma -ffast-math -Wall -Wextra -Wno-unused-parameter)
    else()
        add_compile_options(-O3 -ffast-math -Wall -Wextra -Wno-unused-parameter)
    endif()
endif()

# OpenMP support for parallel inference
find_package(OpenMP)
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP found: Enabling multi-threaded inference")
    add_compile_options(${OpenMP_C_FLAGS})
endif()

# Optional Vulkan GPU Compute support
option(BITLLM_ENABLE_VULKAN "Enable optional Vulkan GPU acceleration" OFF)
if(BITLLM_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
    if(NOT Vulkan_LIBRARIES)
        find_library(VULKAN_LIB NAMES vulkan vulkan-1 vulkan.so.1 HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib)
        if(VULKAN_LIB)
            set(Vulkan_FOUND TRUE CACHE BOOL "" FORCE)
            set(Vulkan_LIBRARIES ${VULKAN_LIB} CACHE FILEPATH "" FORCE)
        endif()
    endif()
    if(Vulkan_FOUND OR Vulkan_LIBRARIES)
        message(STATUS "Vulkan found: Enabling optional BitLLM GPU compute acceleration")
        add_compile_definitions(EIF_USE_VULKAN=1)
    else()
        message(STATUS "Vulkan not found: Gracefully defaulting to CPU-only execution (zero dependency)")
    endif()
endif()

include_directories(include)

# Core sources
set(BITLLM_SOURCES
    src/eif_llm.c
    src/eif_qwen35.c
    src/eif_tinyllm.c
    src/eif_bpe_tokenizer.c
    src/eif_quantize_bitnet.c
    src/eif_bert.c
    src/eif_gpu.c
    src/eif_siglip.c
)

# 1. Static Library (libbitllm.a)
add_library(bitllm_static STATIC ${BITLLM_SOURCES})
set_target_properties(bitllm_static PROPERTIES OUTPUT_NAME "bitllm")
target_include_directories(bitllm_static PUBLIC include)
if(UNIX)
    target_link_libraries(bitllm_static PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(bitllm_static PRIVATE OpenMP::OpenMP_C)
endif()
if(BITLLM_ENABLE_VULKAN AND Vulkan_FOUND)
    target_link_libraries(bitllm_static PRIVATE ${Vulkan_LIBRARIES})
endif()

# Backward compatibility alias & symlinks
add_library(eif_runtime_static ALIAS bitllm_static)
add_custom_command(TARGET bitllm_static POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:bitllm_static> ${CMAKE_BINARY_DIR}/libeif_runtime.a
)

# 2. Shared Library (libbitllm.so / .dll)
add_library(bitllm_shared SHARED ${BITLLM_SOURCES})
set_target_properties(bitllm_shared PROPERTIES OUTPUT_NAME "bitllm")
target_include_directories(bitllm_shared PUBLIC include)
if(UNIX)
    target_link_libraries(bitllm_shared PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(bitllm_shared PRIVATE OpenMP::OpenMP_C)
endif()
if(BITLLM_ENABLE_VULKAN AND Vulkan_FOUND)
    target_link_libraries(bitllm_shared PRIVATE ${Vulkan_LIBRARIES})
endif()

# Backward compatibility alias & symlinks
add_library(eif_runtime_shared ALIAS bitllm_shared)
add_custom_command(TARGET bitllm_shared POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:bitllm_shared> ${CMAKE_BINARY_DIR}/libeif_runtime.so
)

# 3. CLI Runner Executable (bitllm-run & eif-run alias)
add_executable(bitllm-run cli/main.c)
target_link_libraries(bitllm-run PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(bitllm-run PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(bitllm-run PRIVATE OpenMP::OpenMP_C)
endif()

# Symlink or secondary target for eif-run compatibility
add_custom_target(eif-run ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:bitllm-run> ${CMAKE_BINARY_DIR}/eif-run
    DEPENDS bitllm-run
)

# 4. Feature Test Suite (test_all_features)
add_executable(test_all_features tests/test_all_features.c)
target_link_libraries(test_all_features PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_all_features PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_all_features PRIVATE OpenMP::OpenMP_C)
endif()

# 5. Live Model Test Suite (test_live_models)
add_executable(test_live_models tests/test_live_models.c)
target_link_libraries(test_live_models PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_live_models PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_live_models PRIVATE OpenMP::OpenMP_C)
endif()

# 6. ALiBi & Benchmark Test Suite (test_alibi_and_benchmark)
add_executable(test_alibi_and_benchmark tests/test_alibi_and_benchmark.c)
target_link_libraries(test_alibi_and_benchmark PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_alibi_and_benchmark PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_alibi_and_benchmark PRIVATE OpenMP::OpenMP_C)
endif()

# 7. Docling Native Multimodal Test (test_docling_native)
add_executable(test_docling_native tests/test_docling_native.c)
target_link_libraries(test_docling_native PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_docling_native PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_docling_native PRIVATE OpenMP::OpenMP_C)
endif()

# 8. SigLIP2 Vision Encoder Unit Test (test_siglip_native)
add_executable(test_siglip_native tests/test_siglip_native.c)
target_link_libraries(test_siglip_native PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_siglip_native PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_siglip_native PRIVATE OpenMP::OpenMP_C)
endif()

# 9. Hadamard rotation + group-scaled ternary matmul (test_bitnet_hadamard)
add_executable(test_bitnet_hadamard tests/test_bitnet_hadamard.c)
target_link_libraries(test_bitnet_hadamard PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_bitnet_hadamard PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_bitnet_hadamard PRIVATE OpenMP::OpenMP_C)
endif()

# 9b. Idefics3 image tiling / resampling (test_siglip_tiling)
add_executable(test_siglip_tiling tests/test_siglip_tiling.c)
target_link_libraries(test_siglip_tiling PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_siglip_tiling PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_siglip_tiling PRIVATE OpenMP::OpenMP_C)
endif()

# 9c. Generation loop guard (test_loop_guard)
add_executable(test_loop_guard tests/test_loop_guard.c)
target_link_libraries(test_loop_guard PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_loop_guard PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_loop_guard PRIVATE OpenMP::OpenMP_C)
endif()

# 9d. INT8 row-scaled GEMV kernels (test_int8_gemv)
add_executable(test_int8_gemv tests/test_int8_gemv.c)
target_link_libraries(test_int8_gemv PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_int8_gemv PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_int8_gemv PRIVATE OpenMP::OpenMP_C)
endif()

# 10. Quantised SigLIP2 vision weights (INT8 / INT4 g128 / + Hadamard) (test_siglip_quant)
add_executable(test_siglip_quant tests/test_siglip_quant.c)
target_link_libraries(test_siglip_quant PRIVATE bitllm_static)
if(UNIX)
    target_link_libraries(test_siglip_quant PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(test_siglip_quant PRIVATE OpenMP::OpenMP_C)
endif()

enable_testing()
add_test(NAME test_siglip_quant COMMAND test_siglip_quant)
add_test(NAME test_siglip_tiling COMMAND test_siglip_tiling)
add_test(NAME test_loop_guard COMMAND test_loop_guard)
add_test(NAME test_int8_gemv COMMAND test_int8_gemv)
add_test(NAME test_all_features COMMAND test_all_features)
add_test(NAME test_live_models COMMAND test_live_models)
add_test(NAME test_alibi_and_benchmark COMMAND test_alibi_and_benchmark)
add_test(NAME test_siglip_native COMMAND test_siglip_native)
add_test(NAME test_bitnet_hadamard COMMAND test_bitnet_hadamard)

# Install rules
install(TARGETS bitllm_static bitllm_shared bitllm-run
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
