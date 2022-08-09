cmake_minimum_required(VERSION 3.16)
project(LOSiamMask)

set(CMAKE_CXX_STANDARD 14)
# It prevents the decay to C++98 when the compiler does not support C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# It disables the use of compiler-specific extensions
# e.g. -std=c++14 rather than -std=gnu++14
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
#add_definitions(-D _GLIBCXX_USE_CXX11_ABI=0)
#set(CMAKE_PREFIX_PATH "/home/xmcchv/libtorch")
# Try to find OpenCV
# set(OpenCV_DIR ....)
set(CUDA_LAUNCH_BLOCKING 1)
find_package(PythonInterp REQUIRED)


find_package(OpenCV REQUIRED)
#find_package(OpenCV 4 REQUIRED
#        COMPONENTS
#        opencv_core
#        opencv_imgproc
#        opencv_highgui
#)
if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()


set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)
set(Torch_DIR libtorch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

include_directories(
        "."
        "./third_party/argparse"
        "./third_party/json/include"
        ${OpenCV_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/SiamMask
)

file(GLOB SOURCE_FILES src/*.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
#add_executable(${CMAKE_PROJECT_NAME} demo.cpp)

target_link_libraries (
        ${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)
