cmake_minimum_required(VERSION 3.16)
project(infergo-sdk C CXX)

find_library(INFERGO_LIB infer_api
    PATHS ${INFERGO_LIB_DIR} /usr/local/lib /usr/lib
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../build/cpp/api)

find_path(INFERGO_INCLUDE infer_api.h
    PATHS ${INFERGO_INCLUDE_DIR} /usr/local/include
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include)

add_library(infergo::infergo SHARED IMPORTED)
set_target_properties(infergo::infergo PROPERTIES
    IMPORTED_LOCATION ${INFERGO_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${INFERGO_INCLUDE})

foreach(ex chat embed detect rag)
    add_executable(${ex} examples/${ex}.c)
    target_link_libraries(${ex} PRIVATE infergo::infergo)
    target_include_directories(${ex} PRIVATE ${INFERGO_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

install(FILES infergo.h DESTINATION include)
configure_file(infergo.pc.in ${CMAKE_CURRENT_BINARY_DIR}/infergo.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/infergo.pc DESTINATION lib/pkgconfig)
