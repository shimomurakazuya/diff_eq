#ifndef INDEX_H_
#define INDEX_H_

#include "defines.h"

namespace Index {
<<<<<<< HEAD
#ifdef __CUDA_ARCH__
        __host__ __device__ 
#endif
=======
    #ifdef __CUDA_ARCH__
    __host__ __device__ 
    #endif
>>>>>>> 66fabc9622896da52bb50f4673cf7db3edee0e16
    inline int index_xy(int i, int j) {
        return  ( i + defines::nx* j );
    };
};
#endif


