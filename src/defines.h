#ifndef DEFINE_H_
#define DEFINE_H_

#ifdef USE_NVCC
  #define __HOST__    __host__
  #define __DEVICE__  __device__
#else
  #define __HOST__ 
  #define __DEVICE__
#endif

#define __HD__ __HOST__ __DEVICE__

//using real = float;
using real = double;

namespace defines {
    // x, t
    constexpr real lx = 1.0;
    constexpr int nx = 100;
    constexpr int ny = 100;
    //constexpr int nx = 2000000;
    //constexpr int ny = 384;
    //constexpr int nx = 6144;
    //constexpr int ny = 6144 ;
    constexpr int ncell = nx*ny;
    constexpr int ncx = nx / 2;
    constexpr int ncy = ny / 2;
    constexpr double dx = lx / nx;
    constexpr int iter = 20000;
    constexpr int iout = 2000;
 
    //4 cal preformance
    //constexpr int iter = 10;
    //constexpr int iout = 1;

    // diffusion
    constexpr real fmax = 1;
    constexpr real delta = 4 * dx; // 4 dx
    constexpr real c_dif = 1;
    //constexpr real pi =3.14159265359;
    constexpr real dt = 0.25* dx*dx /c_dif; // torima
    constexpr real coef_diff= c_dif * dt /dx / dx; 

    // downsize
    constexpr int downsize = 5;
    
    // 
    constexpr int data_num   = 2;
    constexpr int flop_num   = 7;
    constexpr int thread_num = 2;
    #ifdef USE_NVCC
    constexpr int bandwidth = 900;
    #else
    constexpr int bandwidth = 140;
    #endif
    constexpr real loof_line    = bandwidth * flop_num/(8*data_num); 
    constexpr int alignment = 32;

    //cuda
     constexpr int blocksizex = 32;
     constexpr int blocksizey = 32;
     //constexpr int ntx        = 16;
     //constexpr int nty        = 16;

};

#define D() { \
        std::cout << __PRETTY_FUNCTION__ << " @ " << __FILE__ << ':' << __LINE__ << std::endl; \
}

#endif
