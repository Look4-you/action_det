cmake_minimum_required(VERSION 2.8)
project(vapd_yolov8_det)

set(CMAKE_CXX_STANDARD 11)

# 使CMAKE支持C++11特性
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu++0x")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")


# opencv
SET(OpenCV_DIR /opt/opencv-3.4.0/build)
set(OpenCV_DIR "/data/xuwenyao/opencv-cuda114/share/OpenCV")
find_package(OpenCV  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(/data/xuwenyao/opencv-cuda114/include/)
link_directories(/data/xuwenyao/opencv-cuda114/lib/)

# CUDA
include_directories(/data/xuwenyao/cuda-11.4/include)
link_directories(/data/xuwenyao/cuda-11.4/lib64)
set(CMAKE_CUDA_COMPILER /data/xuwenyao/cuda-11.4/bin/nvcc)
enable_language(CUDA)

# tensorrt
include_directories(/data/xuwenyao/TensorRT-8.4.0.6/targets/x86_64-linux-gnu/include/)
# link_directories(/data/xuwenyao/TensorRT-8.4.0.6/bin)
link_directories(/data/xuwenyao/TensorRT-8.4.0.6/targets/x86_64-linux-gnu/lib/)


# include_directories(/data/xuwenyao/TensorRT-8.4.0.6/include/)
# # link_directories(/data/xuwenyao/TensorRT-8.4.0.6/bin)
# link_directories(/data/xuwenyao/TensorRT-8.4.0.6/lib)


add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/yolov8/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

include_directories(${PROJECT_SOURCE_DIR}/include
                        ${PROJECT_SOURCE_DIR}/src/yolov8/include
                        ${PROJECT_SOURCE_DIR}/src/vit
                        ${PROJECT_SOURCE_DIR}/src/yolov8/plugin)
# include_directories(${PROJECT_SOURCE_DIR}/src/yolov8/include)
# include_directories(${PROJECT_SOURCE_DIR}/src/yolov8/plugin)

FILE(GLOB SOURCES 
        src/*.cpp 
        src/vit/*.cpp
        src/vit/*.cu
        src/yolov8/*.cu
        src/yolov8/*.cpp
        )
 
ADD_EXECUTABLE(vapd_yolov8_det
		${SOURCES}
		# /home/TensorRT-8.2.4.2/samples/common/logger.cpp
        )

TARGET_LINK_LIBRARIES(vapd_yolov8_det
        ${OpenCV_LIBS}
        cudart
        /data/xuwenyao/TensorRT-8.4.0.6/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so.8.4.0
        # /app/xuwenyao/cuda/lib64/libcudnn.so
        myplugins
        nvparsers nvonnxparser nvinfer 
        ) 