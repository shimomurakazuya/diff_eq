#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>

// define.h
using real  = float;
namespace defines {
    // x, t
    constexpr real dx = 1.0;
    constexpr int nx = 51;
    constexpr int nc = nx / 2;
    //constexpr real lx = nx - 1;
    constexpr int iter = 100000;
    constexpr int iout = 100;
    constexpr real dt = 0.001; // torima

    // diffusion
    constexpr real fmax = 1;
    constexpr real delta = 4 * dx; // 4 dx
    constexpr real c_dif = 1;
};

real* allocate() {
    return (real*)std::malloc(defines::nx * sizeof(real));
}

void deallocate(real* f) {
    std::free(f);
}

void init(real* f, real* x) {
    for(int i=0; i<defines::nx; i++) {
        const real xi = (i - defines::nc) * defines::dx / defines::delta;
        x[i] = xi;
        f[i] = defines::fmax * std::exp( - xi*xi);
    }
}

void time_integrate(real* fn, const real* f) {
    for(int i=0; i<defines::nx; i++) {
        const int im = (i-1 + defines::nx) % defines::nx;
        const int ip = (i+1 + defines::nx) % defines::nx;
        fn[i] = f[i] + 
            + defines::c_dif * defines::dt / defines::dx / defines::dx * (f[im] - 2*f[i] + f[ip]);
    }
}

void print_sum(const int t, const real* f) {
    double sum = 0, mmm = f[0];
    for(int i=0; i<defines::nx; i++) {
        sum += f[i];
        mmm = std::fmax(mmm, f[i]);
    }
    std::cout << "t=" << std::setw(8) << t 
        << " :    " << std::setw(8) << sum 
        << " ,    " << std::setw(8) << mmm << std::endl;

}

void output_data(const int t, const real* f) {
    // not implemented
}

void swap_memory(real**f, real**fn) {
    real* tmp = *fn;
    *fn = *f;
    *f = tmp;
}

int main() {
    real* f  = allocate();
    real* fn = allocate();
    real* x  = allocate();
    real* xn = allocate();

    init(f , x );
    init(fn, xn);

    // main loop
    for(int t=0; t<defines::iter; t++) {
        // output
        if(t % defines::iout == 0) {
            const int fout_step = t / defines::iout;
            print_sum(fout_step, f);
            output_data(fout_step, f);
        }

        // 
        swap_memory(&fn, &f);
        time_integrate(fn, f);
    }

}
