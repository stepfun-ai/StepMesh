# You can build either from the source tree or from the released Python package.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(klx_backend LANGUAGES C CXX)

execute_process(COMMAND ${Python_EXECUTABLE}
        -c "import torch; print(int(torch.compiled_with_cxx11_abi()))"
        OUTPUT_VARIABLE TORCH_CXX11_ABI OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC -O3 -Wall -finline-functions -msse2 -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} ")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_VERBOSE_MAKEFILE OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig;print(sysconfig.get_config_var('SOABI'))"
  OUTPUT_VARIABLE PY_SOABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)


add_library(klx_backend SHARED klx_backend.cc)

target_include_directories(klx_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}../../include)


set(MODULE_NAME fserver_lib)
set(PY_EXT_SUFFIX ".so")

set(PYTHON_EXTENSION_NAME
    "${MODULE_NAME}.${PY_SOABI}${PY_EXT_SUFFIX}"
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(klx_backend
    PRIVATE
    CUDA::cudart
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

install(TARGETS klx_backend
    LIBRARY DESTINATION lib
)

# Note: If you are building your own Python package,
# please update this path to point to your own relative directory.
set_target_properties(klx_backend PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/../"
    INSTALL_RPATH "\$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)
