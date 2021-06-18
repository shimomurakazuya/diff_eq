#ifndef KENREL_H_
#define KERNEL_H_
 
#include "defines.h"
#include "Index.h"


namespace  kernel{
//#ifdef __CUDA_ARCH__
//    __host__ __device__
//#endif
        __global__  void DiffusionEq(real* fn, const real* f){
     //        for(int j=0; j<defines::ny; j++) {
     const int j = blockIdx.y*blockDim.y + threadIdx.y;
     const int jm = (j-1 + defines::ny) % defines::ny;
     const int jp = (j+1 + defines::ny) % defines::ny;
     //            for(int i=0; i<defines::nx; i++) {
     const int i = blockIdx.x*blockDim.x + threadIdx.x;
     const int im = (i-1 + defines::nx) % defines::nx;
     const int ip = (i+1 + defines::nx) % defines::nx;

     const int ji  = Index::index_xy(i,j);
     const int jim = Index::index_xy(im,j);
     const int jip = Index::index_xy(ip,j);
     const int jim2 = Index::index_xy(i,jm);
     const int jip2 = Index::index_xy(i,jp);

     fn[ji] = f[ji]
         + defines::coef_diff * (f[jim] - 4*f[ji] + f[jip] + f[jim2] + f[jip2] );
};
};

#endif

