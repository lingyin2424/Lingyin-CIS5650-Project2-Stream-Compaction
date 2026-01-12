set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "thrust.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_library(stream_compaction ${sources} ${headers})

# CUDA 架构设置
# 当前系统 GPU: NVIDIA TITAN RTX (计算能力 7.5)
# 优先编译 7.5，针对 TITAN RTX 优化（单一架构以获得最佳性能）
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Release 模式：只编译 TITAN RTX 架构以获得最佳性能和最小体积
    set_target_properties(stream_compaction PROPERTIES 
        CUDA_ARCHITECTURES "75")
else()
    set_target_properties(stream_compaction PROPERTIES 
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Release 模式：完全优化，无调试信息
target_compile_options(stream_compaction PRIVATE 
    "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>"
    "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;--use_fast_math;-Xptxas=-v>"
)

# MSVC C++ 编译选项
if(MSVC)
  target_compile_options(stream_compaction PRIVATE
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2;/Ob2;/Oi;/Ot;/GL>"
    )
endif()
