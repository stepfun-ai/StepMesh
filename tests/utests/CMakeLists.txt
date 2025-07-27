set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wall -ldl -lpthread -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++17 -ldl -lpthread -g")

add_executable(ut_scheduler
        ut_scheduler.cc
        ut_common.h)
target_link_libraries(ut_scheduler
        af)

add_executable(ut_server
        ut_server.cc
        ut_common.h)
target_link_libraries(ut_server
        af)

add_executable(ut_tensor_worker
        ut_tensor_worker.cc
        ut_common.h)
target_link_libraries(ut_tensor_worker
        af)

add_executable(stepmesh_echo_test stepmesh_echo_test.cc test_common.h)
target_link_libraries(stepmesh_echo_test dl pthread m af)

add_executable(stepmesh_push_test stepmesh_push_test.cc test_common.h)
target_link_libraries(stepmesh_push_test dl pthread m af)

add_executable(stepmesh_pull_test stepmesh_pull_test.cc test_common.h)
target_link_libraries(stepmesh_pull_test dl pthread m af)

add_executable(stepmesh_register_test stepmesh_register_test.cc test_common.h)
target_link_libraries(stepmesh_register_test dl pthread m af)