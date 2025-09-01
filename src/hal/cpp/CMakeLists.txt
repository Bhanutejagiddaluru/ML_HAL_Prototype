
cmake_minimum_required(VERSION 3.20)
project(hal_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Threads REQUIRED)

pybind11_add_module(hal_ext hal_ext.cpp)
target_link_libraries(hal_ext PRIVATE Threads::Threads)
