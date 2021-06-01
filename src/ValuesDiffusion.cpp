#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <math.h>

#include "ValuesDiffusion.h"
#include "Index.h"
#include "defines.h"

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
    for(int i=0; i<nx_; i++) {
        const real xi = (i - defines::ncx) * defines::dx;
        x_[i] = xi;
        for (int j=0; j<ny_; j++){
            const real yi = (j - defines::ncy) * defines::dx;
            y_[j] = yi;
            const int ij = index::index_xy(i, j);
            //f_[ij] = defines::fmax * std::exp( - (xi*xi + yi*yi ));
            f_[ij] = defines::fmax * cos(xi/defines::lx*2.0*M_PI + yi/defines::lx*2.0*M_PI);
        }
    }
}

void ValuesDiffusion::
time_integrate(const ValuesDiffusion& valuesDiffusion) {
    real* fn = f_;
    const real* f = valuesDiffusion.f_;
    for(int i=0; i<defines::nx; i++) {
        for(int j=0; j<defines::ny; j++) {
        const int ij = index::index_xy(i,j);
   //     const int im = (ij-ny_ + defines::ncell) % defines::ncell;
   //     const int ip = (ij+ny_ + defines::ncell) % defines::ncell;
   //     const int jm = (ij-1 + defines::ncell) % defines::ncell;
   //     const int jp = (ij+1 + defines::ncell) % defines::ncell;
         int im = ij- defines::ny  ;
         int ip = ij+ defines::ny  ;
         int jm = ij-1      ;
         int jp = ij+1      ;
    // periodic     
        if(j== 0            ) jm = jm + ny_;  
        if(j== defines::ny-1) jp = jp - ny_;  
        if(i== 0            ) im = im + defines::ncell;
        if(i== defines::nx-1) ip = ip - defines::ncell;
        fn[ij] = f[ij] + 
            + defines::c_dif * defines::dt / defines::dx / defines::dx * (f[im] - 2*f[ij] + f[ip] ) 
            + defines::c_dif * defines::dt / defines::dx / defines::dx * (f[jm] - 2*f[ij] + f[jp]);
        }
    }
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

