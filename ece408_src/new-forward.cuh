
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH_ONE 24
#define TILE_WIDTH_TWO 24
#define CACHE_SIZE  16000

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float weight_cache[CACHE_SIZE];
__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    int n,m,h,w,c,p,q;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) weight_cache[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    int W_grid = ceil((float)W_out/TILE_WIDTH_ONE);

    n = blockIdx.x;
    m = blockIdx.y;
    h = blockIdx.z / W_grid*TILE_WIDTH_ONE + threadIdx.y;
    w = blockIdx.z % W_grid*TILE_WIDTH_ONE + threadIdx.x;
    float acc=0;
    if(h<H_out && w<W_out)
    {
    for(c=0; c<C; c++){
        for(p=0; p<K; p++){
          #pragma unroll 5
            for(q=0; q<K; q++)
            {
                acc += x4d(n,c,h+p,w+q)*k4d(m,c,p,q);
            }
        }
    }
     y4d(n,m,h,w) = acc;
}

#undef y4d
#undef x4d
#undef k4d
}


__global__ void forward_kernel_two(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)

{
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

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

  #pragma unroll 4
  for(int i = 0; i < ceil(width/(1.0*TILE_WIDTH_TWO)); i++)
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
	#pragma unroll 24
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

  #undef x4d
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
    int W_grid,H_grid,Z;
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

	//printf("%d, %d", C, K);

    if(M % TILE_WIDTH_ONE != 0)
    {
      //printf("in the first pass");
      W_grid = ceil((float)W_out/TILE_WIDTH_ONE);
      H_grid = ceil((float)H_out/TILE_WIDTH_ONE);
      Z = H_grid * W_grid;
      dim3 blockDim(TILE_WIDTH_ONE,TILE_WIDTH_ONE,1);
      dim3 gridDim(B,M,Z);
      int kernelSize = w.shape_[0] * w.shape_[1] * w.shape_[2] * w.shape_[3];
      cudaMemcpyToSymbol(weight_cache, w.dptr_, sizeof(float)*kernelSize, 0, cudaMemcpyDefault);
      forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);
    }
    else{
      //printf("in the second pass");
      dim3 blockDim(TILE_WIDTH_TWO,TILE_WIDTH_TWO,1);
      dim3 gridDim(ceil(1.0*H_out*W_out/TILE_WIDTH_TWO),ceil(1.0*M/TILE_WIDTH_TWO),B);
      forward_kernel_two<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    }

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
