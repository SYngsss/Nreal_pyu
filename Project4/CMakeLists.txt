cmake_minimum_required(VERSION 2.8)
project(OpenCV_)
find_package(OpenCV 3.0 QUIET)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Ceres REQUIRED)
find_package(FMT REQUIRED)
find_package(Sophus REQUIRED )


include_directories(${CERES_INCLUDE_DIRS})
# include_directories("/usr/local/include/matplotlibcpp.h")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE PYTHON2.7_LIB "/usr/lib/python2.7/config-x86_64-linux-gnu/*.so")
#指定头文件路径
set(PYTHON2.7_INLCUDE_DIRS "/usr/include/python2.7")
#添加头文件到工程
include_directories(include
        ${PYTHON2.7_INLCUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        # ${NUMPY_INLCUDE_DIRS}
        )
        # /usr/lib/python2.7/dist-packages/numpy/core/include/numpy/__multiarray_api.h

include_directories(src)

add_library(${PROJECT_NAME} SHARED src/init.cc)


target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${CERES_LIBRARIES}
${Sophus_LIBRARIES}
fmt::fmt
/usr/local/lib/libopencv_xfeatures2d.so # 手动添加特征点提取动态库
)

add_executable(main main.cc)

target_link_libraries(main ${PROJECT_NAME})
target_link_libraries(main ${PYTHON2.7_LIB})
