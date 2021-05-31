#ifndef DEFINE_H_
#define DEFINE_H_

#include<math.h>

using real = float;
namespace defines {
    // x, t
    constexpr real lx = 1.0 ;
    constexpr int nx = 800;
    constexpr int nc = nx / 2;
    constexpr real dx = lx / nx;
    constexpr int iter = 1000000;
    constexpr int iout = 100;
//    constexpr int iter = 1;
//    constexpr int iout = 1;
//    constexpr real dt = 0.00025; // torima

    // diffusion
    constexpr real fmax = 1;
    constexpr real delta = 4 * dx; // 4 dx
    constexpr real c_dif = 1;

    constexpr real dt = 0.25* dx*dx/ c_dif ; //  4 investigation err

    // downsize
          constexpr int downsize = 3;

};


#endif
