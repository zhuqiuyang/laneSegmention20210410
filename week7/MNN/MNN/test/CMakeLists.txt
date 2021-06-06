cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnn_infer)

set(CMAKE_BUILD_TYPE Release)

include_directories("..\\MNN-1.0.0\\include")
link_directories("..\\Windows\\MNN_Windows_lib\\Dynamic_Library")

include_directories("D:\\Software\\opencv\\build\\include")
link_directories("D:\\Software\\opencv\\build\\x64\\vc12\\lib")

# include_directories("3rdparty\\FreeImage3180Win32Win64\\FreeImage\\Dist\\x64")
# link_directories("3rdparty\\FreeImage3180Win32Win64\\FreeImage\\Dist\\x64")

add_executable(main main.cpp)
target_link_libraries(main MNN)
target_link_libraries(main opencv_world420)
# target_link_libraries(main FreeImage)
set_property(TARGET main PROPERTY CXX_STANDARD 14)