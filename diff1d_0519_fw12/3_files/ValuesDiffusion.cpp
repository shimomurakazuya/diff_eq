#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#include "ValuesDiffusion.h"

#include "defines.h"

void ValuesDiffusion::
allocate_values() {
    x_ = (real*)std::malloc(sizeof(real) * nx_);
    f_ = (real*)std::malloc(sizeof(real) * nx_);
}

void ValuesDiffusion::
deallocate_values() {
    std::free(x_);
    std::free(f_);
}

void ValuesDiffusion::
init_values() {
    for(int i=0; i<nx_; i++) {
        const real xi = (i - defines::nc) * defines::dx / defines::delta;
        x_[i] = xi;
        f_[i] = defines::fmax * std::exp( - xi*xi);
    }
}

void ValuesDiffusion::
time_integrate(const ValuesDiffusion& valuesDiffusion) {
    real* fn = f_;
    const real* f = valuesDiffusion.f_;
    for(int i=0; i<defines::nx; i++) {
        const int im = (i-1 + defines::nx) % defines::nx;
        const int ip = (i+1 + defines::nx) % defines::nx;
        fn[i] = f[i] + 
            + defines::c_dif * defines::dt / defines::dx / defines::dx * (f[im] - 2*f[i] + f[ip]);
    }
}

void ValuesDiffusion::
copy_values(const ValuesDiffusion& valuesDiffusion) {
    x_ = valuesDiffusion.x_;
    f_ = valuesDiffusion.f_;
}

void ValuesDiffusion::
swap(ValuesDiffusion* v0, ValuesDiffusion* v1) {
    ValuesDiffusion tmp(*v0);
    v0->copy_values(*v1);
    v1->copy_values(tmp);
}

void ValuesDiffusion::
print_sum(const int t) {
    double sum = 0, mmm = f_[0];
    for(int i=0; i<defines::nx; i++) {
        sum += f_[i];
        mmm = std::fmax(mmm, f_[i]);
    }
    std::cout << "t=" << std::setw(8) << t 
        << " :    " << std::setw(8) << sum 
        << " ,    " << std::setw(8) << mmm << std::endl;

}
