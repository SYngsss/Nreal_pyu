# 静态库的cmake文件
include_directories(./lib_so)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(hello main.cc)
target_link_libraries(hello /home/yup/桌面/model/Nreal_Project/Project1/lib_so/libhello.a)

# 动态库的cmake文件
# include_directories(./lib_so)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# add_executable(hello main.cc)
# target_link_libraries(hello /home/yup/桌面/model/Nreal_Project/Project1/lib_so/libhello.so)
