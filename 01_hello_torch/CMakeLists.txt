cmake_minimum_required(VERSION 3.0)
project(torch_test)

set(CMAKE_PREFIX_PATH "/home/eren/anaconda3/envs/teknofest_yarisma/lib/python3.7/site-packages/torch")

set(CMAKE_CXX_STANDARD 14)
find_package(Torch REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example main.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}")