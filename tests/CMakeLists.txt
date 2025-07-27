set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wall -ldl -lpthread -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++17 -ldl -lpthread -g")

add_subdirectory(./utests)