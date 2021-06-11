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
        constexpr real lx = 1.0;
    constexpr int nx = 100000000;
    constexpr int ncx = nx / 2;
    constexpr real dx = lx/ nx;
    constexpr int iter = 100;
    constexpr int iout = 10;

    // diffusion
    constexpr real c_dif = 3;
    constexpr int thread_num = 24;
    constexpr int data_num = 2;
};

namespace index{
    int index_xy(int  i , int j){
        return  ( i + defines::nx* j );
    }
};


real* allocate() {
    return (real*)std::malloc(defines::nx * sizeof(real));
}

void deallocate(real* f) {
    std::free(f);
}


void init(real* f, real* x) {
#pragma omp parallel for
    for(int i=0; i<defines::nx; i++) {
        const real xi = (i - defines::ncx) *  defines::dx;
        x[i] =xi;
         f[i] = 2.0;
    }
}

void time_integrate(real* fn, const real* f) {
    int im, ip;
#pragma omp parallel for private(im,ip) 
    for(int i=0; i<defines::nx; i++) { 
          im = (i-1 + defines::nx) % defines::nx;  
          ip = (i+1 + defines::nx) % defines::nx;  
        fn[i] = f[i] + defines::c_dif * (f[ip] - 2.0* f[i] + f[im]);
    }
}   

void output_bw(const double ave_tloop_min){
    double bw ; 
    bw = 1e-09 * defines::data_num *8.0 * defines::nx/ave_tloop_min;  
    std::cout << "sec_min(s)=" << ave_tloop_min  << std::endl
        << "band_width(GB/s)=" << bw <<std::endl ;


}

int main() {
    real* f  = allocate();
    real* fn = allocate();
    real* x  = allocate();
    real* xn = allocate();
    real st_tloop, ed_tloop, ave_tloop=100, ave_tloop_min=100,tloop;


    omp_set_num_threads(defines::thread_num);
    init(f , x );
    init(fn, xn);

    // main loop
    for(int t=0; t<defines::iter; t++) {
        // output
        if(t % (defines::iout) == 0) {
            printf("sec_ave=%lf \n",ave_tloop);
            ave_tloop_min = std::min(ave_tloop_min, ave_tloop);
            printf("sec_min=%lf \n",ave_tloop_min);
            ave_tloop =0; 
        }

        st_tloop=omp_get_wtime();
        time_integrate(fn,f);
        ed_tloop=omp_get_wtime();

        tloop = ed_tloop-st_tloop;
        ave_tloop = ave_tloop + tloop/defines::iout; 
        printf("sec=%lf \n",tloop);
    }
    output_bw(ave_tloop_min);

}
