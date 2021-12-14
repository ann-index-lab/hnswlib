cmake_minimum_required (VERSION 2.6)
project(hnsw_lib
    LANGUAGES CXX C)

find_package(HDF5 COMPONENTS CXX HL REQUIRED)

message(${HDF5_VERSION})

add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE .) 

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
    endif()

    add_executable(test_updates examples/updates_test.cpp)
    target_link_libraries(test_updates hnswlib)

    add_executable(searchKnnCloserFirst_test examples/searchKnnCloserFirst_test.cpp)
    target_link_libraries(searchKnnCloserFirst_test hnswlib)

    add_executable(main main.cpp sift.cpp)
    target_link_libraries(main hnswlib)
endif()

include_directories(
    ${HDF5_INCLUDE_DIRS}
    ${HDF5_CXX_INCLUDE_DIRS}
    ${HDF5_C_INCLUDE_DIRS}
    )

add_executable(hdf5_test hdf5_test.cpp)
target_link_libraries(hdf5_test ${HDF5_LIBRARIES})

add_executable(hdf5_create hdf5_create.cpp)
target_link_libraries(hdf5_create ${HDF5_LIBRARIES})
