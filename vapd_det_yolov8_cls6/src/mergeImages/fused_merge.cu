// fused_merge.cu
#include "fused_merge.hpp"
#include <opencv2/core/cuda/common.hpp> // 包含 PtrStepSz 等 CUDA 辅助工具

// --- GPU 端设备函数：计算两张图的平均值并转为灰度 ---
// __device__ 表示该函数只能被 GPU 调用
__device__ __forceinline__ uchar computeGray(uchar3 p1, uchar3 p2) {
    // OpenCV GpuMat 默认是 BGR 格式: x=B, y=G, z=R
    // 计算两张图的平均值 (0.5 * A + 0.5 * B)
    float b = (p1.x + p2.x) * 0.5f;
    float g = (p1.y + p2.y) * 0.5f;
    float r = (p1.z + p2.z) * 0.5f;
    
    // RGB 转灰度公式：Y = 0.299*R + 0.587*G + 0.114*B 
    // 加 0.5f 是为了四舍五入
    return (uchar)(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
}

// --- GPU Kernel 函数 ---
// __global__ 表示这是一个 CUDA 核心，由 CPU 调用，在 GPU 上执行
__global__ void fusedMergeKernel(
    cv::cuda::PtrStepSz<uchar3> in0, cv::cuda::PtrStepSz<uchar3> in1,
    cv::cuda::PtrStepSz<uchar3> in2, cv::cuda::PtrStepSz<uchar3> in3,
    cv::cuda::PtrStepSz<uchar3> in4, cv::cuda::PtrStepSz<uchar3> in5,
    cv::cuda::PtrStepSz<uchar3> out) 
{
    // 计算当前线程对应的图像像素坐标 (x, y)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查：防止越界访问
    if (x >= in0.cols || y >= in0.rows) return;

    // 分别计算三个目标通道的灰度值
    // imgB <- in0 & in1
    uchar grayB = computeGray(in0(y, x), in1(y, x));
    
    // imgG <- in2 & in3
    uchar grayG = computeGray(in2(y, x), in3(y, x));
    
    // imgR <- in4 & in5
    uchar grayR = computeGray(in4(y, x), in5(y, x));

    // 合并为一张 BGR 图像并写入显存（一次性写入，极大地节省带宽）
    out(y, x) = make_uchar3(grayB, grayG, grayR);
}

namespace vapd {
namespace cuda {

// --- CPU 端包装函数：配置并启动 Kernel ---
void launchFusedMerge(const std::vector<cv::cuda::GpuMat>& inputs, 
                      cv::cuda::GpuMat& output, 
                      cv::cuda::Stream& stream) 
{
    if (inputs.size() < 6) return;

    int cols = inputs[0].cols;
    int rows = inputs[0].rows;

    // 分配输出图像的显存 (如果大小不变，OpenCV 内部不会重复分配)
    output.create(rows, cols, CV_8UC3);

    // 配置 CUDA 线程块大小 (Block) 和网格大小 (Grid)
    // 通常使用 16x16 或 32x8 的 Block 大小
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // 获取 OpenCV 的底层 CUDA 流
    cudaStream_t cuStream = cv::cuda::StreamAccessor::getStream(stream);

    // 启动 Kernel 函数
    fusedMergeKernel<<<grid, block, 0, cuStream>>>(
        inputs[0], inputs[1], 
        inputs[2], inputs[3], 
        inputs[4], inputs[5], 
        output
    );

    // 检查是否有启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
}

} // namespace cuda
} // namespace vapd
