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
    //constexpr int nx = 100000000;
    constexpr int nx = 2000000;
    constexpr int ny = 100;
    constexpr int ncell = nx*ny;
    constexpr int ncx = nx / 2;
    constexpr int ncy = ny / 2;
    constexpr real dx = lx/nx ;
    constexpr real dy = lx/ny ;
    constexpr int iter = 100;
    constexpr int iout = 10;
    //    constexpr real dt = 0.001; // torima

    // diffusion
    //    constexpr real fmax = 1;
    //    constexpr real delta = 4 * dx; // 4 dx
    constexpr real c_dif = 3;
    constexpr int thread_num = 24;
    constexpr int data_num = 4;
};

namespace index{
    int index_xy(int  i , int j){
        return  ( i + defines::nx* j );
    }
};


real* allocate() {
    return (real*)std::malloc(defines::ncell * sizeof(real));
}

void deallocate(real* f) {
    std::free(f);
}

// void init(real* f, real* x) {
// #pragma omp parallel for
//     for(int i=0; i<defines::nx; i++) {
//         const real xi = (i - defines::ncx) *  defines::dx;
//         x[i] =xi;
//          f[i] = 2.0;
//     }
// }



void init(real* f, real* x, real* y) {
#pragma omp parallel for
    for(int j=0; j<defines::ny; j++) {
          const real yi = (j - defines::ncy) *  defines::dy;
        y[j] = yi;
        for(int i=0; i<defines::nx; i++) {
            int ji = index::index_xy(i,j);
         const real xi = (i - defines::ncx) *  defines::dx;
            x[i] = xi;
            f[ji] = cos(xi/defines::lx*2.0*M_PI + yi/defines::lx*2.0*M_PI);
        }
    }
}

void time_integrate(real* fn, const real* f) {
#pragma omp parallel for 
    for(int j=0; j<defines::ny; j++) { 
        const int jm = (j-1 + defines::ny) % defines::ny;
        const int jp = (j+1 + defines::ny) % defines::ny;
        for(int i=0; i<defines::nx; i++) { 
            const int im = (i-1 + defines::nx) % defines::nx;
            const int ip = (i+1 + defines::nx) % defines::nx;

            const int ji  = index::index_xy(i,j);
            const int jim = index::index_xy(im,j);
            const int jip = index::index_xy(ip,j);
            const int jim2 = index::index_xy(i,jm);
            const int jip2 = index::index_xy(i,jp);

            fn[ji] = fn[ji]
                + defines::c_dif* (f[jim] - 4*f[ji] + f[jip] + f[jim2] + f[jip2] );

        }
    }
}

void output_bw(const double ave_tloop_min){
    double bw1, bw2 ; 
    bw1 = 1e-09 * (defines::data_num-2) *8.0 * defines::ncell/ave_tloop_min;  
    bw2 = 1e-09 * (defines::data_num) *8.0 * defines::ncell/ave_tloop_min;  
    std::cout << "sec_min(s)= , band_width(GB/s)= "<< std::endl
        << ave_tloop_min << "     " << bw1  <<"~" << bw2 <<std::endl ;
}


int main() {
    real* f  = allocate();
    real* fn = allocate();
    real* x  = allocate();
    real* xn = allocate();
    real* y  = allocate();
    real* yn = allocate();
    real st_tloop, ed_tloop, ave_tloop=100, ave_tloop_min=100,tloop;


    omp_set_num_threads(defines::thread_num);
    init(f , x ,y);
    init(fn, xn,yn);

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
