
//#ifdef _OPENMP
//#include <omp.h>
//#endif

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
//#include <cutil.h>

#include "ValuesDiffusion.h"
#include "Index.h"
#include "defines.h"
#include "Kernel.h"

void ValuesDiffusion::
allocate_values() {
    x_ = (real*)std::malloc(sizeof(real) * nx_);
    y_ = (real*)std::malloc(sizeof(real) * ny_);
    f_ = (real*)std::malloc(sizeof(real) * ncell_);
    cudaMalloc(&d_f_ ,defines::ncell*sizeof(real));
}

void ValuesDiffusion::
deallocate_values() {
    std::free(x_);
    std::free(y_);
    std::free(f_);
    cudaFree(d_f_);
}

void ValuesDiffusion::
init_values() {
    #pragma omp parallel 
        { 
    #pragma omp for  
    for (int j=0; j<ny_; j++){
        const real yi = (j - defines::ncy) *  defines::dx; 
        y_[j] = yi;
        for(int i=0; i<nx_; i++) {
            const real xi = (i - defines::ncx) *  defines::dx; 
            x_[i] = xi;
            const int ji = Index::index_xy(i, j);
            //f_[ji] = defines::fmax * std::exp( - (xi*xi + yi*yi ));
            f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*M_PI + yi/defines::lx*2.0*M_PI);
            //f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*M_PI);
            //f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*defines::pi)*cos(yi/defines::lx*2.0*defines::pi) ;
        }
    }
    }
    cudaMemcpy(d_f_ , f_ , defines::ncell*sizeof(real), cudaMemcpyHostToDevice);

}

void ValuesDiffusion::
d2h(void){
       cudaMemcpy(f_, d_f_, defines::ncell*sizeof(real), cudaMemcpyDeviceToHost);
}


void ValuesDiffusion::
time_integrate_gpu(const ValuesDiffusion& valuesDiffusion) {
    real* d_fn = d_f_;
    const real* d_f = valuesDiffusion.d_f_;
    //real* d_f,* d_fn;  
    cudaError_t error;

    dim3 block(defines::blocksizex,defines::blocksizey,1 ); 
    dim3 grid ((defines::nx+block.x-1)/block.x, (defines::ny+block.y-1)/block.y,1);

    kernel::DiffusionEq <<<grid, block>>> (d_fn, d_f);

    error = cudaGetLastError ();
    if(error != cudaSuccess ){
        printf("cuda error stop \n"); 
        exit(1);
    };

    cudaDeviceSynchronize();

}

void ValuesDiffusion::
time_integrate_cpu(const ValuesDiffusion& valuesDiffusion) {
     real* fn = f_;
     const real* f = valuesDiffusion.f_;
#pragma omp parallel
    {
#pragma omp for 
             for(int j=0; j<defines::ny; j++) {
      const int jm = (j-1 + defines::ny) % defines::ny;
      const int jp = (j+1 + defines::ny) % defines::ny;
              for(int i=0; i<defines::nx; i++) {
      const int im = (i-1 + defines::nx) % defines::nx;
      const int ip = (i+1 + defines::nx) % defines::nx;
 
      const int ji  = Index::index_xy(i,j);
      const int jim = Index::index_xy(im,j);
      const int jip = Index::index_xy(ip,j);
      const int jim2 = Index::index_xy(i,jm);
      const int jip2 = Index::index_xy(i,jp);

            fn[ji] = f[ji] 
                    + defines::coef_diff * (f[jim] - 4*f[ji] + f[jip] + f[jim2] + f[jip2] );
         }
     }
 }
}


void ValuesDiffusion::
copy_values(const ValuesDiffusion& valuesDiffusion) {
    x_ = valuesDiffusion.x_;
    y_ = valuesDiffusion.y_;
    f_ = valuesDiffusion.f_;
    d_f_ = valuesDiffusion.d_f_;
}

void ValuesDiffusion::
swap(ValuesDiffusion* v0, ValuesDiffusion* v1) {
    ValuesDiffusion tmp(*v0);
    v0->copy_values(*v1);
    v1->copy_values(tmp);
}

