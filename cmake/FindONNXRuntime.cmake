# cmake/FindONNXRuntime.cmake
#
# Locates a prebuilt ONNX Runtime installation.
#
# Search order:
#   1. ONNXRUNTIME_ROOT cmake variable (set via -DONNXRUNTIME_ROOT=...)
#   2. ONNXRUNTIME_ROOT environment variable
#   3. ~/onnxruntime  (default download location per project convention)
#
# Defines on success:
#   ONNXRuntime_FOUND          — TRUE
#   ONNXRuntime_INCLUDE_DIRS   — path to onnxruntime_c_api.h
#   ONNXRuntime_LIBRARIES      — full path to libonnxruntime.so
#   ONNXRuntime_VERSION        — version string from VERSION_NUMBER file (if present)
#
# Creates imported target:
#   ONNXRuntime::ONNXRuntime   — use with target_link_libraries()

# ─── Locate root ─────────────────────────────────────────────────────────────

if(NOT ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    else()
        set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime")
    endif()
endif()

# ─── Find header ─────────────────────────────────────────────────────────────

find_path(ONNXRuntime_INCLUDE_DIRS
    NAMES onnxruntime_c_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
)

# ─── Find library ────────────────────────────────────────────────────────────

find_library(ONNXRuntime_LIBRARIES
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
)

# ─── Version (optional) ──────────────────────────────────────────────────────

if(EXISTS "${ONNXRUNTIME_ROOT}/VERSION_NUMBER")
    file(READ "${ONNXRUNTIME_ROOT}/VERSION_NUMBER" ONNXRuntime_VERSION)
    string(STRIP "${ONNXRuntime_VERSION}" ONNXRuntime_VERSION)
endif()

# ─── Standard result handling ────────────────────────────────────────────────

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARIES ONNXRuntime_INCLUDE_DIRS
    VERSION_VAR   ONNXRuntime_VERSION
)

# ─── Imported target ─────────────────────────────────────────────────────────

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
    add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_LOCATION             "${ONNXRuntime_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIRS}"
    )
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIRS ONNXRuntime_LIBRARIES)
