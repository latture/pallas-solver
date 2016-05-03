CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)
PROJECT(Pallas C CXX)

option(BUILD_PALLAS_TESTS "Build unit tests" ON)
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(EIGEN_SUBDIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/eigen/")
set(GTEST_SUBDIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/googletest/googletest")
set(FIND_GLOG_SUBDIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH "${FIND_GLOG_SUBDIR};${CMAKE_MODULE_PATH}")

macro(check_dir_exists NAME)
    if(NOT EXISTS ${NAME})
        message(FATAL_ERROR "Directory specified by " NAME " (" ${NAME} ") was not found.")
    endif()
endmacro()

check_dir_exists(${EIGEN_SUBDIR})
check_dir_exists(${GTEST_SUBDIR})
check_dir_exists(${FIND_GLOG_SUBDIR})

find_package(Ceres REQUIRED)
find_package(Glog REQUIRED)

include_directories(APPEND
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/
                    ${CERES_INCLUDE_DIRS}
                    ${EIGEN_SUBDIR}
                    ${GTEST_SUBDIR}
                    ${GLOG_INCLUDE_DIR})

add_subdirectory(${GTEST_SUBDIR})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/pallas)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
