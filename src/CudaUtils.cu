/******************************************************************************
MIT License

Copyright (c) 2016 Antti-Pekka Hynninen
Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

#include <stdio.h>
#ifdef ENABLE_NVTOOLS
#include <nvToolsExtCuda.h>
#endif
#include "CudaUtils.h"

//----------------------------------------------------------------------------------------

void set_device_array_async_T(void *data, int value, const size_t ndata, cudaStream_t stream, const size_t sizeofT) {
  cudaCheck(cudaMemsetAsync(data, value, sizeofT*ndata, stream));
}

void set_device_array_T(void *data, int value, const size_t ndata, const size_t sizeofT) {
  cudaCheck(cudaMemset(data, value, sizeofT*ndata));
}

//----------------------------------------------------------------------------------------
//
// Jittor malloc & free
//
void cutt_malloc(void** p, size_t len, size_t& allocation) {
  cudaCheck(cudaMalloc(p, len));
}

void cutt_free(void* p, size_t len, size_t& allocation) {
  cudaCheck(cudaFree(p));
}

void (*custom_cuda_malloc)(void** p, size_t len, size_t& allocation) = NULL;

void (*custom_cuda_free)(void* p, size_t len, size_t& allocation) = NULL;

//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
void allocate_device_T(void **pp, const size_t len, const size_t sizeofT) {
  cudaCheck(cudaMalloc(pp, sizeofT*len));
}

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
void deallocate_device_T(void **pp) {
  if (*pp != NULL) {
    cudaCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }

}

//----------------------------------------------------------------------------------------
//
// Jittor allocate gpu memory
// pp = memory pointer
// len = length of the array
//
void jit_allocate_device_T(void **pp, const size_t len, const size_t sizeofT, size_t& allocation) {
  if (custom_cuda_malloc==NULL){
    cutt_malloc(pp, sizeofT*len, allocation);
  }else custom_cuda_malloc(pp, sizeofT*len, allocation);
}

//----------------------------------------------------------------------------------------
//
// Jittor deallocate gpu memory
// pp = memory pointer
//
void jit_deallocate_device_T(void **pp, const size_t len, const size_t sizeofT, size_t& allocation) {
  if (*pp != NULL) {
    if (custom_cuda_free==NULL){
      cutt_free((void *)(*pp), sizeofT*len, allocation);
    }else custom_cuda_free((void *)(*pp), sizeofT*len, allocation);
    *pp = NULL;
  }

}

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
void copy_HtoD_async_T(const void *h_array, void *d_array, size_t array_len, cudaStream_t stream,
           const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(d_array, h_array, sizeofT*array_len, cudaMemcpyHostToDevice, stream));
}

void copy_HtoD_T(const void *h_array, void *d_array, size_t array_len,
     const size_t sizeofT) {
  cudaCheck(cudaMemcpy(d_array, h_array, sizeofT*array_len, cudaMemcpyHostToDevice));
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
void copy_DtoH_async_T(const void *d_array, void *h_array, const size_t array_len, cudaStream_t stream,
           const size_t sizeofT) {
  cudaCheck(cudaMemcpyAsync(h_array, d_array, sizeofT*array_len, cudaMemcpyDeviceToHost, stream));
}

void copy_DtoH_T(const void *d_array, void *h_array, const size_t array_len, const size_t sizeofT) {
  cudaCheck(cudaMemcpy(h_array, d_array, sizeofT*array_len, cudaMemcpyDeviceToHost));
}

//----------------------------------------------------------------------------------------
#ifdef ENABLE_NVTOOLS
void gpuRangeStart(const char *range_name) {
  static int color_id=0;
  nvtxEventAttributes_t att;
  att.version = NVTX_VERSION;
  att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  att.colorType = NVTX_COLOR_ARGB;
  if (color_id == 0) {
    att.color = 0xFFFF0000;
  } else if (color_id == 1) {
    att.color = 0xFF00FF00;
  } else if (color_id == 2) {
    att.color = 0xFF0000FF;
  } else if (color_id == 3) {
    att.color = 0xFFFF00FF;
  }
  color_id++;
  if (color_id > 3) color_id = 0;
  att.messageType = NVTX_MESSAGE_TYPE_ASCII;
  att.message.ascii = range_name;
  nvtxRangePushEx(&att);
}

void gpuRangeStop() {
  nvtxRangePop();
}
#endif
