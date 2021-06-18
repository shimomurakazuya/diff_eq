#ifndef INDEX_H_
#define INDEX_H_

#include "defines.h"

namespace Index {
    #ifdef __CUDA_ARCH__
    __host__ __device__ 
    #endif
    inline int index_xy(int i, int j) {
        return  ( i + defines::nx* j );
    };
};
#endif


