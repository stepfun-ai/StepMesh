set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wall -ldl -lpthread -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++17 -ldl -lpthread -g")

add_executable(test_benchmark_af_tensor test_benchmark_af_tensor.cc test_common.h)
target_link_libraries(test_benchmark_af_tensor dl pthread m af)

add_executable(test_benchmark_af_verify test_benchmark_af_verify.cc test_common.h)
target_link_libraries(test_benchmark_af_verify dl pthread m af)

add_executable(test_register_recv_buffer test_register_recv_buffer.cc test_common.h)
target_link_libraries(test_register_recv_buffer dl pthread m af)

add_subdirectory(./utests)