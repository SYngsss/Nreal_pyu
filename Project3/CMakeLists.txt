cmake_minimum_required(VERSION 2.8)
project(OpenCV_)
find_package(OpenCV 3.0 QUIET)
find_package(Eigen3 3.1.0 REQUIRED)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
include_directories(src)
add_library(${PROJECT_NAME} SHARED src/init.cc)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
/usr/local/lib/libopencv_xfeatures2d.so # 手动添加特征点提取动态库
)

add_executable(main main.cc)
target_link_libraries(main ${PROJECT_NAME})