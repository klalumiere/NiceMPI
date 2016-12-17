cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
message("-- Build type is ${CMAKE_BUILD_TYPE}")

project(NiceMPI LANGUAGES CXX)

find_package(MPI REQUIRED)

if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    find_package(GTest)
    if(GTEST_FOUND)
        enable_testing()
    endif()
endif()

include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(include)
add_subdirectory(src)
