
//#ifdef _OPENMP
//#include <omp.h>
//#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
//#include <cutil.h>

#include "ValuesDiffusion.h"
#include "Index.h"
#include "defines.h"
#include <cuda.h>

namespace  kernel{

    __global__  void DiffusionEq(const real* fn, const real* f){

     //        for(int j=0; j<defines::ny; j++) {
     const int j = blockIdx.y*blockDim.y + threadIdx.y;
     const int jm = (j-1 + defines::ny) % defines::ny;
     const int jp = (j+1 + defines::ny) % defines::ny;
     //            for(int i=0; i<defines::nx; i++) {
     const int i = blockIdx.x*blockDim.x + threadIdx.x;
     const int im = (i-1 + defines::nx) % defines::nx;
     const int ip = (i+1 + defines::nx) % defines::nx;
 
     const int ji  = index::index_xy(i,j);
     const int jim = index::index_xy(im,j);
     const int jip = index::index_xy(ip,j);
     const int jim2 = index::index_xy(i,jm);
     const int jip2 = index::index_xy(i,jp);
 
     fn[ji] = f[ji]
         + defines::coef_diff * (f[jim] - 4*f[ji] + f[jip] + f[jim2] + f[jip2] );
};
};



void ValuesDiffusion::
allocate_values() {
    x_ = (real*)std::malloc(sizeof(real) * nx_);
    y_ = (real*)std::malloc(sizeof(real) * ny_);
    f_ = (real *)aligned_alloc( defines::alignment, ncell_*sizeof(real));
}

void ValuesDiffusion::
deallocate_values() {
    std::free(x_);
    std::free(y_);
    std::free(f_);
}

void ValuesDiffusion::
init_values() {
//#pragma omp parallel 
//    { 
//#pragma omp for  
        for (int j=0; j<ny_; j++){
            const real yi = (j - defines::ncy) *  defines::dx; 
            y_[j] = yi;
            for(int i=0; i<nx_; i++) {
                const real xi = (i - defines::ncx) *  defines::dx; 
                x_[i] = xi;
                const int ji = index::index_xy(i, j);
                //f_[ji] = defines::fmax * std::exp( - (xi*xi + yi*yi ));
                f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*M_PI + yi/defines::lx*2.0*M_PI);
                //f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*defines::pi)*cos(yi/defines::lx*2.0*defines::pi) ;
            }
        }
//    }
}

//__global__  void ValuesDiffusion::
//DiffusionEq(const real* fn, const real* f){
//
//    //        for(int j=0; j<defines::ny; j++) {
//    const int j = blockIdx.y*blockDim.y + threadIdx.y;                                                   
//    const int jm = (j-1 + defines::ny) % defines::ny;
//    const int jp = (j+1 + defines::ny) % defines::ny;
//    //            for(int i=0; i<defines::nx; i++) {
//    const int i = blockIdx.x*blockDim.x + threadIdx.x;                                                   
//    const int im = (i-1 + defines::nx) % defines::nx;
//    const int ip = (i+1 + defines::nx) % defines::nx;
//
//    const int ji  = index::index_xy(i,j);
//    const int jim = index::index_xy(im,j);
//    const int jip = index::index_xy(ip,j);
//    const int jim2 = index::index_xy(i,jm);
//    const int jip2 = index::index_xy(i,jp);
//
//    fn[ji] = f[ji] 
//        + defines::coef_diff * (f[jim] - 4*f[ji] + f[jip] + f[jim2] + f[jip2] );
//
//
//}

void ValuesDiffusion::
time_integrate(const ValuesDiffusion& valuesDiffusion) {
    real* fn = f_;
    const real* f = valuesDiffusion.f_;
    real* d_f, d_fn;  

    cudaMalloc(&d_f ,defines::ncell*sizeof(real));
    cudaMalloc(&d_fn,defines::ncell*sizeof(real));

    cudaMemcpy(d_fn, fn, defines::ncell*sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f , f , defines::ncell*sizeof(real), cudaMemcpyHostToDevice);


    dim3 block(defines::blocksizex,defines::blocksizey,1 ); 
        dim3 grid (defines::nx/block.x, defines::ny/block.y,1);

        kernel::DiffusionEq <<<grid, block>>> (fn, f);

    cudaMemcpy(fn, d_fn, defines::ncell*sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_f);
    cudaFree(d_fn);

}


//void ValuesDiffusion::
//time_integrate(const ValuesDiffusion& valuesDiffusion) {
//    real* fn = f_;
//    const real* f = valuesDiffusion.f_;
//#pragma omp parallel  
//    { 
//        //#pragma omp for schedule(static,8) 
//#pragma omp for 
//        for(int j=0; j<defines::ny; j++) {
//            const int jm = (j-1 + defines::ny) % defines::ny;
//            const int jp = (j+1 + defines::ny) % defines::ny;
//#pragma GCC ivdep
//            for(int i=0; i<defines::nx; i++) {
//                const int im = (i-1 + defines::nx) % defines::nx;
//                const int ip = (i+1 + defines::nx) % defines::nx;
//
//                const int ji  = index::index_xy(i,j);
//                const int jim = index::index_xy(im,j);
//                const int jip = index::index_xy(ip,j);
//                const int jim2 = index::index_xy(i,jm);
//                const int jip2 = index::index_xy(i,jp);
//
//                fn[ji] = f[ji] 
//                    + defines::coef_diff * (f[jim] - 4*f[ji] + f[jip] + f[jim2] + f[jip2] );
//            }
//        }
//    }
//}

void ValuesDiffusion::
copy_values(const ValuesDiffusion& valuesDiffusion) {
    x_ = valuesDiffusion.x_;
    y_ = valuesDiffusion.y_;
    f_ = valuesDiffusion.f_;
}

void ValuesDiffusion::
swap(ValuesDiffusion* v0, ValuesDiffusion* v1) {
    ValuesDiffusion tmp(*v0);
    v0->copy_values(*v1);
    v1->copy_values(tmp);
}

