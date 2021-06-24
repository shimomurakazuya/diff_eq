#ifndef DEFINE_H_
#define DEFINE_H_

//using real = float;
using real = double;

namespace defines {
    // x, t
//    constexpr real dx = 4.0;
    constexpr real lx = 1.0;
//    constexpr int nx = 20000;
//    constexpr int ny = 10000;
    constexpr int nx = 200;
    constexpr int ny = 200;
//    constexpr int nx = 2000000;
//    constexpr int nx = 2097152;
//    constexpr int ny =96;
//    constexpr int nx = 6144;
//    constexpr int ny = 6144 ;
    constexpr int ncell = nx*ny;
    constexpr int ncx = nx / 2;
    constexpr int ncy = ny / 2;
    constexpr double dx = lx / nx;
    constexpr int iter = 20000;
    constexpr int iout = 20;
 
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
    constexpr int thread_num = 24;
    constexpr int cpubandwidth = 140;
    constexpr int gpubandwidth = 900;
    constexpr real loof_line    = gpubandwidth * flop_num/(8*data_num); 
    constexpr int alignment = 32;

    //cuda
     constexpr int blocksizex = 4;
     constexpr int blocksizey = 4;

};

#define D() { \
        std::cout << __PRETTY_FUNCTION__ << " @ " << __FILE__ << ':' << __LINE__ << std::endl; \
}

#endif
