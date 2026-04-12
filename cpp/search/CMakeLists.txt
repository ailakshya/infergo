# cpp/search/CMakeLists.txt — HNSW vector search index

add_library(infer_search STATIC hnsw.cpp vector_db.cpp)

target_include_directories(infer_search
    PUBLIC  ${CMAKE_SOURCE_DIR}/cpp/include
            ${CMAKE_CURRENT_SOURCE_DIR}
)
