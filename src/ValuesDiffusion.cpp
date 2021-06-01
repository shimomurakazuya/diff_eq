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
    for (int j=0; j<ny_; j++){
        const real yi = (j - defines::ncy) * defines::dx;
        for(int i=0; i<nx_; i++) {
            const real xi = (i - defines::ncx) * defines::dx;
            x_[i] = xi;
            y_[j] = yi;
            const int ji = index::index_xy(i, j);
            //f_[ji] = defines::fmax * std::exp( - (xi*xi + yi*yi ));
            f_[ji] = defines::fmax * cos(xi/defines::lx*2.0*M_PI + yi/defines::lx*2.0*M_PI);
        }
    }
}

void ValuesDiffusion::
time_integrate(const ValuesDiffusion& valuesDiffusion) {
    real* fn = f_;
    const real* f = valuesDiffusion.f_;
    for(int j=0; j<defines::ny; j++) {
            const int jm = (j-1 + defines::ny) % defines::ny;
            const int jp = (j+1 + defines::ny) % defines::ny;
        for(int i=0; i<defines::nx; i++) {
            const int im = (i-1 + defines::nx) % defines::nx;
            const int ip = (i+1 + defines::nx) % defines::nx;

            const int ji = index::index_xy(i,j);
            const int jim = index::index_xy(im,j);
            const int jip = index::index_xy(ip,j);
            const int jim2 = index::index_xy(i,jm);
            const int jip2 = index::index_xy(i,jp);

            fn[ji] = f[ji] + 
                + defines::c_dif * defines::dt / defines::dx / defines::dx * (f[jim] - 2*f[ji] + f[jip] ) 
                + defines::c_dif * defines::dt / defines::dx / defines::dx * (f[jim2] - 2*f[ji] + f[jip2]);
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

