cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
    set(msg "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used)")
    set(msg "${msg} Debug Release RelWithDebInfo MinSizeRel.")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING ${msg} FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

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
