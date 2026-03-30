# cmake/Toolchain.cmake
# Toolchain wrapper for infergo.
#
# Usage (explicit):
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain.cmake -B build
#
# Usage (implicit, recommended):
#   export VCPKG_ROOT=/path/to/vcpkg
#   cmake -B build          # root CMakeLists.txt picks it up automatically
#
# Install vcpkg:
#   git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
#   ~/vcpkg/bootstrap-vcpkg.sh
#   export VCPKG_ROOT=~/vcpkg

if(NOT DEFINED VCPKG_ROOT)
    if(DEFINED ENV{VCPKG_ROOT})
        set(VCPKG_ROOT "$ENV{VCPKG_ROOT}")
    else()
        message(FATAL_ERROR
            "VCPKG_ROOT is not set.\n"
            "Install vcpkg and set the environment variable:\n"
            "  git clone https://github.com/microsoft/vcpkg.git ~/vcpkg\n"
            "  ~/vcpkg/bootstrap-vcpkg.sh\n"
            "  export VCPKG_ROOT=~/vcpkg\n"
        )
    endif()
endif()

set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")
