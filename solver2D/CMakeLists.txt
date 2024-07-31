#
# ---------------------------------------
# Define project and CMAKE requirements
# ---------------------------------------

message("")
string(TIMESTAMP NOW "%H:%M on %a, %d %b")
message("-- Running CMake v${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION} at ${NOW}")
message("")

cmake_minimum_required(VERSION 3.10)

set(project "solver2D")
project(${project})

# ---------------------------------------
# Check compiler
# ---------------------------------------

message(STATUS "Using GCC version ${CMAKE_CXX_COMPILER_VERSION}")
message("")

# ---------------------------------------
# Select build type and platform
# ---------------------------------------

# Check build arguments
# ---------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
else()
	message(FATAL_ERROR "Build type not recognized, available options are only: Debug or Release!" )
endif()

if(CMAKE_BUILD_ENV STREQUAL "OMP")
elseif(CMAKE_BUILD_ENV STREQUAL "CUDA")
else()
	message(FATAL_ERROR "Build environment not recognized, available options are only: OMP or CUDA!" )
endif()

# Set default: sequential debug
# ---------------------------------------

set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "Setting build type to '${default_build_type}' (default) since none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}")
else()
	message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}'.")
endif()

set(default_build_env "OMP")
if(NOT CMAKE_BUILD_ENV)
	message(STATUS "Setting build environment to '${default_build_env}' (default) since none was specified.")
	set(CMAKE_BUILD_ENV "${default_build_env}")
else()
	message(STATUS "Setting build environment to '${CMAKE_BUILD_ENV}'.")
endif()

set(default_build_dist "OFF")
if(NOT CMAKE_BUILD_DIST)
	message(STATUS "Setting build environment to '${default_build_env}' (default) since none was specified.")
	set(CMAKE_BUILD_DIST "${default_build_env}")
else()
	message(STATUS "Setting distributed environment to '${CMAKE_BUILD_DIST}'.")
endif()

# ---------------------------------------
# Add compile options
# ---------------------------------------

# C++ standard
# ---------------------------------------

add_compile_options(-std=c++14)

# g++
# ---------------------------------------

if(CMAKE_BUILD_ENV STREQUAL "OMP")
	add_compile_options(-pedantic)
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		add_compile_options(-O0
				-g3
				-Wall
				-Wextra
				-Wconversion)
	elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
		add_compile_options(-O0)
	endif()
endif()

# nvcc
# ---------------------------------------

if(CMAKE_BUILD_ENV STREQUAL "CUDA")
	add_definitions(-D__STAV_CUDA__)
	add_compile_options(-ccbin g++
						-arch=compute_50
						-rdc=true
						-use_fast_math
						-expt-extended-lambda
						-Xptxas -v)
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		add_compile_options(--debug --device-debug)
	elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
		add_compile_options(-Xcompiler -fopenmp)
	endif()
endif()

# ---------------------------------------
# Add sources
# ---------------------------------------

# Common OMP + CUDA
# ---------------------------------------

include_directories(code/headers)
list(APPEND srcList
		code/sources/geometry.cpp
		code/sources/initial.cpp
		code/sources/common.cpp
		code/sources/control.cpp
		code/sources/numerics.cpp
		code/sources/forcing.cpp
		code/sources/friction.cpp
		code/sources/sediment.cpp
		code/sources/output.cpp
		code/sources/mesh.cpp
		code/sources/boundaries.cpp
		code/sources/simulation.cpp)

if(CMAKE_BUILD_DIST STREQUAL "MPI")

	# MPI + OMP
	# ---------------------------------------
	if(CMAKE_BUILD_ENV STREQUAL "OMP")
		list(APPEND srcList
				code/sources/mpiIO.cpp
				code/sources/mpiRun.cpp
				code/Solver2D_vMPI.cpp)
	endif()

	# MPI + CUDA
	# ---------------------------------------
	if(CMAKE_BUILD_ENV STREQUAL "CUDA")
		list(APPEND srcList
				code/sources/gpuIO.cpp
				code/sources/gpuRun.cpp
				code/sources/mpiIO.cpp
				code/sources/mpiRun.cpp
				code/Solver2D_vALL.cpp)

		list(APPEND cudaList ${srcList})
		set_source_files_properties(${cudaList} PROPERTIES LANGUAGE CUDA)
	endif()

else()

	# Only OMP
	# ---------------------------------------
	if(CMAKE_BUILD_ENV STREQUAL "OMP")
		list(APPEND srcList
				code/Solver2D_vOMP.cpp)
	endif()

	# Only CUDA
	# ---------------------------------------
	if(CMAKE_BUILD_ENV STREQUAL "CUDA")
		list(APPEND srcList
				code/sources/gpuIO.cpp
				code/sources/gpuRun.cpp
				code/Solver2D_vCUDA.cpp)

		list(APPEND cudaList ${srcList})
		set_source_files_properties(${cudaList} PROPERTIES LANGUAGE CUDA)
	endif()
endif()

# ---------------------------------------
# Add target
# ---------------------------------------

add_executable(${project} ${srcList})

# ---------------------------------------
# Add required libs
# ---------------------------------------

message("")

# OpenMP 4.5+ for CPU
# ---------------------------------------

set(OMP_MIN_VERSION "4.5")
find_package(OpenMP ${OMP_MIN_VERSION} QUIET REQUIRED)
if(OpenMP_CXX_FOUND)
	message(STATUS "Found OpenMP: v${OpenMP_CXX_VERSION}")
	target_include_directories(${project} PRIVATE ${OpenMP_INCLUDE_DIRS})
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		target_link_libraries(${project} PRIVATE gomp)
	elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
		target_link_libraries(${project} PRIVATE OpenMP::OpenMP_CXX)
	endif()
endif()

# CUDA 9.1+ for GPU
# ---------------------------------------

if(CMAKE_BUILD_ENV STREQUAL "CUDA")
	message("")
	enable_language("CUDA")
	set(CUDA_MIN_VERSION "9.1")
	find_package(CUDA ${CUDA_MIN_VERSION} QUIET REQUIRED)
	if(CUDA_FOUND)
		message("")
		message(STATUS "Found CUDA: v${CUDA_VERSION}")
		add_definitions(-D__STAV_CUDA__)
	endif()
endif()

# MPI 2.1+ for GPU
# ---------------------------------------

if(CMAKE_BUILD_DIST STREQUAL "MPI")
	message("")
	set(MPI_MIN_VERSION "2.1")
	find_package(MPI ${MPI_MIN_VERSION} QUIET REQUIRED)
	if(MPI_CXX_FOUND)
		message("")
		message(STATUS "Found MPI: v${MPI_CXX_VERSION}")
		add_definitions(-DOMPI_SKIP_MPICXX)
		add_definitions(-D__STAV_MPI__)
		target_include_directories(${project} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
		target_link_libraries(${project} PRIVATE ${MPI_CXX_LIBRARIES})
	endif()
endif()

# Done
# ---------------------------------------

message("")


#[[# HDF5 2.1+
# ---------------------------------------

message("")
set(HDF5_MIN_VERSION "1.10")
find_package(HDF5 ${HDF5_MIN_VERSION} QUIET REQUIRED)
if(HDF5_FOUND)
	message(STATUS "Found HDF5: v${HDF5_MIN_VERSION} required!")
	list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
	target_include_directories(${project} PRIVATE ${GDAL_INCLUDE_DIRS})
	target_link_libraries(${project} PRIVATE gdal)
endif()

message("")]]

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
