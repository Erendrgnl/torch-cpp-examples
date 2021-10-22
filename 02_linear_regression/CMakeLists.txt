cmake_minimum_required(VERSION 3.0)
project(torch_test)

set(CMAKE_PREFIX_PATH "/home/eren/anaconda3/envs/teknofest_yarisma/lib/python3.7/site-packages/torch")

set(CMAKE_CXX_STANDARD 14)
find_package(Torch REQUIRED)

add_executable(example linear_regression.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}")