if(INFER_CUDA_AVAILABLE)
    add_library(infer_cuda_monitor STATIC vram_monitor.cpp)

    # Mark the .cpp as CUDA so CMake compiles it with nvcc
    set_source_files_properties(vram_monitor.cpp PROPERTIES LANGUAGE CUDA)

    target_include_directories(infer_cuda_monitor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(infer_cuda_monitor PUBLIC INFER_CUDA_AVAILABLE=1)
    target_link_libraries(infer_cuda_monitor PRIVATE CUDA::cudart)

    # PIC required for linking into shared libinfer_api.so
    set_target_properties(infer_cuda_monitor PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # ─── vram_monitor_test ───────────────────────────────────────────────────
    add_executable(vram_monitor_test vram_monitor_test.cpp)
    set_source_files_properties(vram_monitor_test.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(vram_monitor_test PRIVATE infer_cuda_monitor GTest::gtest_main CUDA::cudart)
    target_compile_definitions(vram_monitor_test PRIVATE INFER_CUDA_AVAILABLE=1)
    include(GoogleTest)
    gtest_discover_tests(vram_monitor_test)
endif()
