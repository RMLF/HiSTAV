
# ---------------------------------------
# Define project and CMAKE version
# ---------------------------------------

cmake_minimum_required(VERSION 3.10)

set(project "mesh")
project(${project})


# ---------------------------------------
# Check GCC
# ---------------------------------------

message("")
message("  Using GCC version ${CMAKE_CXX_COMPILER_VERSION}")
message("")


# ---------------------------------------
# Add Sources
# ---------------------------------------

include_directories(code/headers)
add_executable(${project}
    code/sources/common.cpp
    code/sources/meshBoundary.cpp
    code/sources/meshControl.cpp
    code/sources/meshEntities.cpp
    code/sources/meshGen.cpp
    code/sources/meshGlobals.cpp
    code/sources/meshInterpolation.cpp
    code/sources/meshOrder.cpp
    code/sources/meshOutput.cpp
    code/sources/meshRead.cpp
    code/mesh.cpp)


# ---------------------------------------
# Add required libs
# ---------------------------------------

# CGAL 4.8+
# ---------------------------------------
set(CGAL_MIN_VERSION "4.8")
find_package(CGAL ${CGAL_MIN_VERSION} REQUIRED)
if(CGAL_FOUND)
    message("   CGAL: v${CGAL_MIN_VERSION} is required!")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
    target_include_directories(${project} PRIVATE ${CGAL_INCLUDE_DIRS})
    target_link_libraries(${project} PRIVATE CGAL::CGAL)
endif()

# GDAL 2.1+
# ---------------------------------------
set(GDAL_MIN_VERSION "2.1")
find_package(GDAL ${GDAL_MIN_VERSION} REQUIRED)
if(GDAL_FOUND)
    message("   GDAL: v${GDAL_MIN_VERSION} is required!")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
    target_include_directories(${project} PRIVATE ${GDAL_INCLUDE_DIRS})
    target_link_libraries(${project} PRIVATE gdal)
endif()

# Boost 1.55+
# ---------------------------------------
set(Boost_MIN_VERSION "1.55")
find_package(Boost ${Boost_MIN_VERSION} REQUIRED)
if(Boost_FOUND)
    message("   Boost: v${Boost_MIN_VERSION} is required!")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
    target_include_directories(${project} PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(${project} PRIVATE ${Boost_LIBRARIES})
endif()


# ---------------------------------------
# Add compile options
# ---------------------------------------

target_compile_options(${project} PRIVATE -g3 -Wall -Wfatal-errors -Wpedantic)
target_compile_features(${project} PRIVATE cxx_std_14)

#message( FATAL_ERROR "  STOP!")
