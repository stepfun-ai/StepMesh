cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(af LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC -O3 -Wall -finline-functions  -msse2 ")

# import pytorch library
find_package (Python COMPONENTS Interpreter Development)
execute_process(COMMAND ${Python_EXECUTABLE}
        -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
message("MY PYTHON_EXECUTABLE ${Python_EXECUTABLE}")
message("MY PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")

list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
find_package(Torch REQUIRED CONFIG)
message("MY TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS}")
message("MY CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}")
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_ZMQ -DDMLC_USE_CUDA -DSTEPMESH_USE_GDR -DDMLC_USE_RDMA")

link_directories("${PROJECT_SOURCE_DIR}/deps/lib")

list(APPEND STEPAF_LIBS pthread dl zmq ibverbs rdmacm)

include_directories("${PROJECT_SOURCE_DIR}/include/")
include_directories("${PROJECT_SOURCE_DIR}/deps/include/")
include_directories("${PROJECT_BINARY_DIR}/include/")
include_directories("${PROJECT_BINARY_DIR}/src/")

if("${CUDA_INCLUDE_DIRS}" STREQUAL "")
    message(WARNING "CUDA_INCLUDE_DIRS is empty")
else()
    include_directories("${CUDA_INCLUDE_DIRS}")
endif()

FILE(GLOB SOURCE "src/*.cc")
FILE(GLOB BACKEND_SOURCE "src/backend/*.cc")
message("MY SOURCES ${SOURCE}")
message("MY BACKEND_SOURCE ${BACKEND_SOURCE}")

add_library(af ${SOURCE} ${BACKEND_SOURCE})

target_link_libraries(af
        ${STEPAF_LIBS}
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY})

add_subdirectory(tests)
