#ifndef DEFINE_H_
#define DEFINE_H_

using real = float;
namespace defines {
    // x, t
    constexpr real dx = 1.0;
    constexpr int nx = 51;
    constexpr int ny = 51;
    constexpr int ncell = nx*ny;
    constexpr int ncx = nx / 2;
    constexpr int ncy = ny / 2;
    //constexpr real lx = nx - 1;
    constexpr int iter = 10000;
    constexpr int iout = 100;
//    constexpr int iter = 5;
//    constexpr int iout = 2;
    constexpr real dt = 0.001; // torima

    // diffusion
    constexpr real fmax = 1;
    constexpr real delta = 4 * dx; // 4 dx
    constexpr real c_dif = 1;
    constexpr real pi =3.14;

    // downsize
    constexpr int downsize = 3;
    

};


#endif
