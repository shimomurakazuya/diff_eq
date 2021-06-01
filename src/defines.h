#ifndef DEFINE_H_
#define DEFINE_H_

//using real = float;
using real = double;
namespace defines {
    // x, t
    constexpr real lx = 1.0;
    constexpr int nx = 50;
    constexpr int ny = 50;
    constexpr int ncell = nx*ny;
    constexpr int ncx = nx / 2;
    constexpr int ncy = ny / 2;
    constexpr real dx = lx/nx ;
    constexpr int iter = 10000;
    constexpr int iout = 100;

    // diffusion
    constexpr real fmax = 1;
    constexpr real c_dif = 1;
    constexpr real dt = 0.25* dx*dx/ c_dif;

    // downsize
    constexpr int downsize = 4;
    

};


#endif
