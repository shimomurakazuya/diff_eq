
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
                const int ji = Index::index_xy(i, j);
                //f_[ji] = defines::fmax * std::exp( - (xi*xi + yi*yi ));
                f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*M_PI + yi/defines::lx*2.0*M_PI);
                //f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*defines::pi)*cos(yi/defines::lx*2.0*defines::pi) ;
            }
        }
//    }
}

void ValuesDiffusion::
time_integrate(const ValuesDiffusion& valuesDiffusion) {
    real* fn = f_;
    const real* f = valuesDiffusion.f_;
    real* d_f,* d_fn;  

    cudaMalloc(&d_f ,defines::ncell*sizeof(real));
    cudaMalloc(&d_fn,defines::ncell*sizeof(real));

    cudaMemcpy(d_fn, fn, defines::ncell*sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f , f , defines::ncell*sizeof(real), cudaMemcpyHostToDevice);


    dim3 block(defines::blocksizex,defines::blocksizey,1 ); 
        dim3 grid (defines::nx/block.x, defines::ny/block.y,1);

        kernel::DiffusionEq <<<grid, block>>> (d_fn, d_f);

    cudaMemcpy(fn, d_fn, defines::ncell*sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_f);
    cudaFree(d_fn);

}

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

