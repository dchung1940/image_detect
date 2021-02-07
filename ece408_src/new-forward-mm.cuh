
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH_TWO 24

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void forward_kernel_two(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)

{
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  __shared__ float Mat_X[TILE_WIDTH_TWO][TILE_WIDTH_TWO];
  __shared__ float Mat_K[TILE_WIDTH_TWO][TILE_WIDTH_TWO];
  int temp_h,temp_w,temp_c,temp_p,temp_q;

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE_WIDTH_TWO + ty;
  int col = blockIdx.x * TILE_WIDTH_TWO + tx;
  int img = blockIdx.z;

  int width = C*K*K; // The number of elements need to get each output element
  int height = H_out * W_out; // number of elements in each output feature map

  float result = 0;

  #pragma unroll
  for(int i = 0; i< (width + TILE_WIDTH_TWO -1)/TILE_WIDTH_TWO; i++)
  {
    int temp_row = TILE_WIDTH_TWO * i + ty;
    int temp_col = TILE_WIDTH_TWO * i + tx;

    if(temp_col < width && row < M)
    {
      Mat_K[ty][tx] = k[row * width + temp_col];
    }
    else{
      Mat_K[ty][tx]=0;
    }

    if(temp_row < width && col < height)
    {
      temp_h = col/W_out;
      temp_w = col%W_out;
      temp_c = temp_row / (K*K);
      temp_row = temp_row % (K*K);
      temp_p = temp_row / K;
      temp_q = temp_row % K;
      Mat_X[ty][tx] = x4d(img, temp_c, temp_h+temp_p, temp_w+temp_q);
    }
    else{
      Mat_X[ty][tx] = 0;
    }

    __syncthreads();
	#pragma unroll
    for(int j =0; j< TILE_WIDTH_TWO; j++)
    {
      result += Mat_K[ty][j]*Mat_X[j][tx];
    }
    __syncthreads();
  }

  if(row<M && col < height)
  {
    int index_off = img*M*height;
    y[index_off+row*height+col] = result;
  }

  #undef y4d
  #undef x4d
  #undef k4d
}


/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    //int W_grid,H_grid,Z;
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;


    dim3 blockDim(TILE_WIDTH_TWO,TILE_WIDTH_TWO,1);
    dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH_TWO),ceil(1.0*M/TILE_WIDTH_TWO),B);
    forward_kernel_two<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}
#endif
