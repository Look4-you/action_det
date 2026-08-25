// fused_merge.hpp
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

namespace vapd {
namespace cuda {

// 声明调用 CUDA Kernel 的 C++ 包装接口
void launchFusedMerge(const std::vector<cv::cuda::GpuMat>& inputs, 
                      cv::cuda::GpuMat& output, 
                      cv::cuda::Stream& stream = cv::cuda::Stream::Null());

} // namespace cuda
} // namespace vapd
