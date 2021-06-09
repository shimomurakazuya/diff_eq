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
    constexpr int nx = 2000000;
    constexpr int ny =100;
    constexpr int ncell = nx*ny;
    constexpr int ncx = nx / 2;
    constexpr int ncy = ny / 2;
    constexpr real dx = lx / nx;
    //constexpr real lx = nx - 1;
    constexpr int iter = 50;
    constexpr int iout = 10;

    // diffusion
    constexpr real fmax = 1;
    constexpr real delta = 4 * dx; // 4 dx
    constexpr real c_dif = 1;
    //constexpr real pi =3.14159265359;
    constexpr real dt = 0.25* dx*dx /c_dif; // torima
    constexpr real coef_diff= c_dif * dt /dx / dx; 

    // downsize
    constexpr int downsize = 50;
    
    // 
    constexpr int thread_num = 4;

};


#endif
