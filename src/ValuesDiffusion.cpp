
#ifdef _OPENMP
#include <omp.h>
#endif

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
#ifdef USE_NVCC
    cudaMallocManaged(&f_, defines::ncell* sizeof(real));
#else
    f_ = reinterpret_cast<real*>(std::aligned_alloc(32, defines::ncell * sizeof(real)));
#endif
}

void ValuesDiffusion::
deallocate_values() {
    std::free(x_);
    std::free(y_);
#ifdef USE_NVCC
    cudaFree(f_);
#endif
    std::free(f_);
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
                f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*M_PI + yi/defines::lx*2.0*M_PI);
                //f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*M_PI);
                //f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*defines::pi)*cos(yi/defines::lx*2.0*defines::pi) ;
            }
        }
    }
//#ifdef USE_NVCC
//    cudaMemcpy(d_f_ , f_ , defines::ncell*sizeof(real), cudaMemcpyHostToDevice);
//#endif
}

//void ValuesDiffusion::
//d2h(void){
//    cudaMemcpy(f_, d_f_, defines::ncell*sizeof(real), cudaMemcpyDeviceToHost);
//}


void ValuesDiffusion::
time_integrate(const ValuesDiffusion& valuesDiffusion) {

    real* fn = f_;
    const real* f = valuesDiffusion.f_;
    kernel::exec2d<kernel::opti>(
            // dim3
            defines::nx, defines::ny,
            // lambda
            [=] __HD__ () {
            kernel::DiffusionEq (fn, f);
            }
            );
}



void ValuesDiffusion::
copy_values(const ValuesDiffusion& valuesDiffusion) {
    x_ = valuesDiffusion.x_;
    y_ = valuesDiffusion.y_;
    f_ = valuesDiffusion.f_;
    //d_f_ = valuesDiffusion.d_f_;
}

void ValuesDiffusion::
swap(ValuesDiffusion* v0, ValuesDiffusion* v1) {
    ValuesDiffusion tmp(*v0);
    v0->copy_values(*v1);
    v1->copy_values(tmp);
}

