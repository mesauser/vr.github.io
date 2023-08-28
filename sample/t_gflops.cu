#include <stdio.h>
#include <cuda.h>
#include <mma.h>

void 
print_cuda_error(){
  cudaError_t error = cudaGetLastError();
  if (error){
    printf("Cuda Error: %d %s\n", error, cudaGetErrorString(error));
  }
}

void cudainfo(void *kernel, int blockSize, size_t dynamicSMem) {
  int device;
  cudaDeviceProp prop;

  int numBlocks;
  int activeWarps;
  int maxWarps;

  double occupancy;


  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &numBlocks, kernel, blockSize, dynamicSMem);

  activeWarps = numBlocks * blockSize / prop.warpSize;
  maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

  occupancy = (double)activeWarps / maxWarps;
  printf("prop name = %s\n", prop.name);
  printf("prop clockRate = %d\n", prop.clockRate);
  printf("prop computeMode = %d\n", prop.computeMode);
  printf("prop warpSize = %d\n", prop.warpSize);
  printf("prop maxThreadsPerMultiProcessor= %d\n", prop.maxThreadsPerMultiProcessor);
  printf("prop maxwarps_per_SM = %d\n", prop.maxThreadsPerMultiProcessor/prop.warpSize);

  printf("prop regsPerBlock = %d\n", prop.regsPerBlock);
  printf("prop regsPerMultiprocessor = %d\n", prop.regsPerMultiprocessor);
  printf("prop numSms = %d\n", prop.multiProcessorCount);
  printf("prop sharedMemPerBlock = %lx\n", prop.sharedMemPerBlock);
  printf("prop sharedMemPerMultiprocessor = %lx\n", prop.sharedMemPerMultiprocessor);
  printf("prop sharedMemPerBlockOptin = %lx\n", prop.sharedMemPerBlockOptin);
  printf("prop reservedSharedMemPerBlock = %d\n", prop.reservedSharedMemPerBlock);

  printf("prop CTA = %d\n", numBlocks * blockSize);
  printf("prop occupancy= %f\n", occupancy);
  print_cuda_error();
}


using T=float;

__global__ void kernel_hello(){
  printf(" kernel hello: %d\n", threadIdx.x);
}
__global__ void kernel_vecadd( T * dst, T *src){
  int tid = blockDim.x*blockIdx.x +  threadIdx.x;
  T lm1[4];
  T lm2[4];
  lm1[0] = src[tid];
  lm2[0] = dst[tid];
  lm1[0] += lm2[0];

  dst[tid] = tid;
}

__global__ void kernel_atom(T *dst, int *indices){
  int tid = blockDim.x*blockIdx.x +  threadIdx.x;
  indices[2*tid+1]=clock();
  int idx = tid;
  dst[idx] = tid;
  indices[2*tid+2]=clock();
}

#define C_LAYOUT wmma::mem_row_major
#define M 16
#define N M
#define K M
using namespace nvcuda;
using namespace wmma;
using wmma::fragment;
using wmma::matrix_a;
using wmma::col_major;
__shared__ T shmem[4][8*8];

__global__ void kernel_wmma(T*dst, T *src){
/*
  int tid = blockDim.x*blockIdx.x +  threadIdx.x;
  int warpid = tid/32;
  int laneid = tid%32;

  __nv_bfloat16 bf1;
  half bf2;
*/
  half bf2 = 1.0f;

//  using WT=precision::tf32;
  using WT=half;
  fragment<matrix_a, M, N, K, WT, row_major> a;
  fragment<matrix_b, M, N, K, WT, row_major> b;
  wmma::fragment<wmma::accumulator, M, N, K, float> c;
  wmma::fragment<wmma::accumulator, M, N, K, float> d;

  int ldc = 16;
  half *hsrc = (half*) src;
  wmma::load_matrix_sync(a, hsrc, ldc);
  wmma::load_matrix_sync(b, hsrc, ldc);
  wmma::fill_fragment(c, 2.0f);
  wmma::fill_fragment(a, 1.0f);
  wmma::fill_fragment(b, 1.0f);
  wmma::mma_sync(d,a,b,c);
  wmma::store_matrix_sync(dst, d, ldc, wmma::mem_row_major);
}

#define MSIZE 4096
int main(){
  cudainfo((void*)kernel_body, MSIZE, 0);

  float *dst, *src, *h_dst, *h_src;

  cudaMalloc(&dst, MSIZE);
  cudaMalloc(&src, MSIZE);
  h_dst = (T*)malloc(MSIZE);
  h_src = (T*)malloc(MSIZE);
  for(int i =0;i< MSIZE/sizeof(T); i++){
    h_dst[i] = i;
    h_dst[i] = 0;
    h_src[i] = 0;
  }
  cudaMemcpy(dst, h_dst, MSIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(src, h_src, MSIZE, cudaMemcpyHostToDevice);
  /*
     kernel_vecadd<<<2,2>>>(dst,src);
     kernel_atom<<<8,2>>>(dst,(int*)src);
   */
  kernel_wmma<<<1, 32>>>(dst,src);
  memset(h_dst, 0, MSIZE);
  cudaMemcpy(h_dst, dst, MSIZE, cudaMemcpyDeviceToHost);
  for(int i =0;i < 16*32; i++){
    printf("%3.1f ", h_dst[i]);
    if ((i+1)%16 == 0) printf("\n%d : ", i/16+1);
  }
  printf("\n");
  printf("intbuf:\n");
  for(int i =0;i < 16; i++){
    int *i_src = (int*) malloc(MSIZE);
    cudaMemcpy(i_src, src, MSIZE, cudaMemcpyDeviceToHost);
    printf("%x ", i_src[i]);
    free(i_src);
  }
  printf("\n");

  kernel_hello<<<1,1>>>(); //stop by runtime
  print_cuda_error();
  return 0;
}
