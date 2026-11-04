# IRD C++ build.
#   Always:              ird_preproc (static lib), test_preproc_golden
#   With CUDA+TensorRT:  ird_runtime (shared lib, C API), ird_build_engine, ird_infer
#
# Jetson (JetPack 5.1+ / 6.x):
#   cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
# TensorRT is found in /usr/include/aarch64-linux-gnu and /usr/lib/aarch64-linux-gnu by default;
# set TENSORRT_ROOT for other layouts.
cmake_minimum_required(VERSION 3.18)
project(ird LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # -ffp-contract=off: no fused multiply-add, so results match the Python side bit for bit.
  add_compile_options(-Wall -Wextra -ffp-contract=off)
endif()

option(IRD_WITH_TENSORRT "Build the TensorRT runtime and tools" ON)
set(TENSORRT_ROOT "" CACHE PATH "TensorRT install root (optional)")

# ---- preprocessing (no CUDA) ------------------------------------------------------------
add_library(ird_preproc STATIC src/preproc.cpp)
target_include_directories(ird_preproc PUBLIC include)
set_target_properties(ird_preproc PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(test_preproc_golden tests/test_preproc_golden.cpp)
target_link_libraries(test_preproc_golden PRIVATE ird_preproc)

add_executable(decode_dump tests/decode_dump.cpp runtime/decoder.cpp)   # for tests/test_decode_parity.py
target_include_directories(decode_dump PRIVATE runtime)

# ---- TensorRT runtime ---------------------------------------------------------------------
if(IRD_WITH_TENSORRT)
  find_package(CUDAToolkit)
  find_path(TRT_INCLUDE NvInfer.h
            HINTS ${TENSORRT_ROOT} ENV TENSORRT_ROOT
            PATH_SUFFIXES include include/aarch64-linux-gnu include/x86_64-linux-gnu
            PATHS /usr /usr/local/tensorrt)
  find_library(TRT_LIB NAMES nvinfer nvinfer_10 HINTS ${TENSORRT_ROOT} ENV TENSORRT_ROOT
               PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu PATHS /usr /usr/local/tensorrt)
  find_library(TRT_ONNX_LIB NAMES nvonnxparser nvonnxparser_10 HINTS ${TENSORRT_ROOT} ENV TENSORRT_ROOT
               PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu PATHS /usr /usr/local/tensorrt)
  if(CUDAToolkit_FOUND AND TRT_INCLUDE AND TRT_LIB AND TRT_ONNX_LIB)
    message(STATUS "TensorRT: ${TRT_INCLUDE} ${TRT_LIB}")
    add_library(ird_runtime SHARED runtime/ird_api.cpp runtime/decoder.cpp runtime/runtime_config.cpp)
    target_include_directories(ird_runtime PUBLIC include PRIVATE runtime ${TRT_INCLUDE})
    target_link_libraries(ird_runtime PRIVATE ird_preproc ${TRT_LIB} CUDA::cudart)
    # MSVC writes the import library (ird_runtime.lib) only for exported symbols; export the C API.
    set_target_properties(ird_runtime PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

    add_executable(ird_build_engine tools/ird_build_engine.cpp runtime/runtime_config.cpp)
    target_include_directories(ird_build_engine PRIVATE ${TRT_INCLUDE} runtime ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/stb)
    target_link_libraries(ird_build_engine PRIVATE ird_preproc ${TRT_LIB} ${TRT_ONNX_LIB} CUDA::cudart)

    add_executable(ird_infer tools/ird_infer.cpp)
    target_include_directories(ird_infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/stb)
    target_link_libraries(ird_infer PRIVATE ird_runtime CUDA::cudart)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      set_source_files_properties(tools/ird_infer.cpp PROPERTIES COMPILE_OPTIONS "-Wno-unused-function;-Wno-missing-field-initializers")
    endif()
  else()
    message(WARNING "CUDA/TensorRT not found: building preprocessing only")
  endif()
endif()
