cmake_minimum_required(VERSION 3.17)
project(bnn)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorFlow Lite Integration
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/include/)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../tflite-dist/libs/linux_x64/)
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} tensorflowlite ${OpenCV_LIBS})




