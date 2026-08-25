#include <opencv2/core/cuda.hpp>

__global__ void split3Kernel(cv::cuda::PtrStepSz<float3> src, int rows, int cols, float *dst0,
                             float *dst1, float *dst2, int src_stride, int dst_stride)
{
  int element_x = (blockIdx.x << 5) + threadIdx.x;
  int element_y = (blockIdx.y << 3) + threadIdx.y;
  /*int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int element_x = idx%cols;
  int element_y = idx/cols;*/
  if (element_x >= cols || element_y >= rows)
  {
    return;
  }

  // int index = element_x * 3;
  // float* input = src + element_y * src_stride;
  // float value0, value1, value2;
  float3 input = src(element_y, element_x);
  float value0 = input.x; // input[index];
  float value1 = input.y; // input[index + 1];
  float value2 = input.z; // input[index + 2];

  int offset = element_y * dst_stride;
  float *output0 = dst0 + offset;
  float *output1 = dst1 + offset;
  float *output2 = dst2 + offset;
  //  output2[element_x] = fmaf(value0,1/57.375,-103.53/57.375);//(value0 - 103.53)/57.375;
  //  output1[element_x] = fmaf(value1,1/57.12,-116.28/57.12);//(value1 - 116.28)/57.12;
  //  output0[element_x] = fmaf(value2,1/58.395,-123.675/58.395);//(value2 - 123.675)/58.395;

// //vapd2.0
//   output2[element_x] = fmaf(value0, 1 / 58.395, -123.675 / 58.395);
//   output1[element_x] = fmaf(value1, 1 / 57.12, -116.28 / 57.12);
//   output0[element_x] = fmaf(value2, 1 / 57.375, -103.53 / 57.375);

//uniformerv2
  output2[element_x] = fmaf(value0,1.f / 57.375,-114.75/57.375);
  output1[element_x] = fmaf(value1,1.f / 57.375,-114.75/57.375);
  output0[element_x] = fmaf(value2,1.f / 57.375,-114.75/57.375);

  // output2[element_x] = fmaf(value0,1.f ,0.f);
  // output1[element_x] = fmaf(value1,1.f ,0.f);
  // output0[element_x] = fmaf(value2,1.f ,0.f);

  // std::vector<float> mean{123.675, 116.28, 103.53};//rgb
  // std::vector<float> std{1.f/58.395, 1.f/57.12, 1.f/57.375};

  // std::vector<float> mean{114.75,114.75,114.75,}; // rgb
  // std::vector<float> std{1.f / 57.375, 1.f / 57.375, 1.f / 57.375};
}

inline int divideUp(int total, int grain, int shift)
{
  return (total + grain - 1) >> shift;
}

void split3Channels(cv::cuda::PtrStepSz<float3> src, int rows, int cols,
                    float *dst0, float *dst1, float *dst2, int src_stride, int dst_stride)
{
  dim3 block, grid;
  block.x = 32;
  block.y = 8;
  grid.x = divideUp(cols, 32, 5);
  grid.y = divideUp(rows, 8, 3);

  /*block.x = 256;
  grid.x = int(std::ceil( (rows*cols)/256.f));*/

  split3Kernel<<<grid, block>>>(src, rows, cols, dst0, dst1, dst2, src_stride, dst_stride);
}