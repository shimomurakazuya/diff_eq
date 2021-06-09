#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif


// define.h
using real  = double;
namespace defines {
    // x, t
    //    constexpr real dx = 1.0;
    constexpr int nx = 10000000;
    //    constexpr int nc = nx / 2;
    //constexpr real lx = nx - 1;
    constexpr int iter = 100;
    //    constexpr int iout = 100;
    //    constexpr real dt = 0.001; // torima

    // diffusion
    //    constexpr real fmax = 1;
    //    constexpr real delta = 4 * dx; // 4 dx
    constexpr real c_dif = 3;
    constexpr int thread_num = 24;
};

void init(real* f, real* x) {
#pragma omp parallel for
    for(int i=0; i<defines::nx; i++) {
        x[i] =4.0;
        f[i] = 2.0;
    }
}

void time_integrate(real* fn, const real* f, const real* x) {
#pragma omp parallel for
    for(int i=0; i<defines::nx; i++) {
        fn[i] = f[i] + defines::c_dif * x[i];
    }
}


int main() {
    real f[defines::nx] ;
    real fn[defines::nx];
    real x[defines::nx] ;
    real xn[defines::nx];
    real st_tloop, ed_tloop, ave_tloop=0, tloop=0;

    omp_set_num_threads(defines::thread_num);
    init(f , x );
    init(fn, xn);

    // main loop
    for(int t=0; t<defines::iter; t++) {
        // output

        st_tloop=omp_get_wtime();
        time_integrate(fn,f,x);
        ed_tloop=omp_get_wtime();
        tloop = ed_tloop-st_tloop;
        ave_tloop = ave_tloop + tloop/defines::iter; 

        printf("sec=%lf \n",tloop);
    }
        printf("nx = %d, seci_ave=%lf \n",defines::nx, ave_tloop);
}
