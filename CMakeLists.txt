cmake_minimum_required(VERSION 3.10.2)
project(molecular_dynamics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp -O3")

add_executable(molecular_dynamics main.cpp)