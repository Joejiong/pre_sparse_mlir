#===============================================================================
# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

### General policy inherit from DNNL ###

cmake_minimum_required(VERSION 2.8.11)

if(POLICY CMP0022)
    cmake_policy(SET CMP0022 NEW)
endif()

if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# Enable RPATH on MacOS/OSX
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()

# Do not export symbols from executables
if(POLICY CMP0065)
    cmake_policy(SET CMP0065 NEW)
endif()

# Pass all flags to try_compile
if(POLICY CMP0056)
    cmake_policy(SET CMP0056 NEW)
endif()
if(POLICY CMP0066)
    cmake_policy(SET CMP0066 NEW)
endif()

# Use <PackageName>_ROOT env. variable as a prefix
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ...")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
include("cmake/utils.cmake")

### Semi-compiler specical config ###

set(PROJECT_NAME "kernel_exp")
set(PROJECT_FULL_NAME "kernel_exp")
set(PROJECT_VERSION "0.0.1")

if (CMAKE_VERSION VERSION_LESS 3.0)
    project(${PROJECT_NAME} C CXX)
else()
    cmake_policy(SET CMP0048 NEW)
    project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)
endif()

set(INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/install)

link_directories(${INSTALL_DIR}/lib)

option(ENABLE_LIBXSMM "enable libxsmm" OFF)
if(NOT ENABLE_LIBXSMM STREQUAL ON)
    message(STATUS "libxsmm is OFF")
endif()
option(ENABLE_LIBXSMM_AVX "libxsmm avx option" ON)
if(ENABLE_LIBXSMM_AVX)
set(LIBXSMM_AVX 3)
else()
set(LIBXSMM_AVX 0)
endif()

option(CMEMU "CM Emulator root" OFF)
option(IGC_PATH "IGC root (real GPU/simulator mode)" OFF)
option(KERNEL_PROFILE "Enable kernel profile" OFF)


set(XSMM_LIB ${INSTALL_DIR}/lib/libxsmm.so)

include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

if(${ENABLE_LIBXSMM} STREQUAL ON)
    if (CMAKE_VERSION VERSION_LESS 3.0)
    ExternalProject_Add(ext_xsmm
        PREFIX ${INSTALL_DIR}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libxsmm
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libxsmm PREFIX="${INSTALL_DIR}" BLAS=0 AVX=${LIBXSMM_AVX} FORTRAN=0 SHARED=1 STATIC=0 -j install
    )
    else()
    # Use BUILD_BYPRODUCTS to add ninja dependency
    ExternalProject_Add(ext_xsmm
        BUILD_BYPRODUCTS ${XSMM_LIB}
        PREFIX ${INSTALL_DIR}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libxsmm
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libxsmm PREFIX="${INSTALL_DIR}" BLAS=0 AVX=${LIBXSMM_AVX} FORTRAN=0 SHARED=1 STATIC=0 -j install
    )
    endif()

    add_library(xsmm SHARED IMPORTED GLOBAL)
    add_dependencies(xsmm ext_xsmm)

    # specify where the library is and where to find the headers
    set_target_properties(xsmm
        PROPERTIES
        IMPORTED_LOCATION ${XSMM_LIB}
    )
endif()


find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)

set(DNNL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/mkl-dnn/")

add_subdirectory(dnnl_brgemm)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/include
    ${PROJECT_BINARY_DIR}/3rdparty/mkl-dnn/install/include
    ${DNNL_PATH}/src
    ${DNNL_PATH}/include
    ${MKL_INCLUDE_DIR}
    dnnl_brgemm
    )

if(NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-variable -Wno-unused-function -std=c99 -fopenmp -fPIC -Werror -DSC_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-variable -Wno-unused-function -std=c++17 -fopenmp -fPIC -Werror -DSC_HOME=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")
endif()

if(NOT ${KERNEL_PROFILE} STREQUAL "OFF")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSC_KERNEL_PROFILE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSC_KERNEL_PROFILE")
endif()

add_subdirectory(matmul)