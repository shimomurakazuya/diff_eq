#ifndef KENREL_H_
#define KERNEL_H_

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <string>

#include "defines.h"
#include "Index.h"


#if defined(USE_INTEL)
#define PRAGMA_FOR_SIMD  _Pragma("ivdep")

#define ASSUME_ALIGNED64(VAL) __assume_aligned(VAL, 64)
#define ASSUME64(VAL) __assume(VAL%64 == 0)

#elif defined(USE_A64FX)
#define PRAGMA_FOR_SIMD  _Pragma("loop prefetch_sequential soft")

#define ASSUME_ALIGNED64(VAL)  
#define ASSUME64(VAL)  
#else
#define PRAGMA_FOR_SIMD  

#define ASSUME_ALIGNED64(VAL)  
#define ASSUME64(VAL)  
#endif


// macro
#ifdef __CUDA_ARCH__
#define FOR_EACH2D(I, J, NX, NY) \
    const auto I = threadIdx.x+blockIdx.x* blockDim.x; \
    const auto J = threadIdx.y+blockIdx.y* blockDim.y;


#define SKIP_FOR() return
#else
#define FOR_EACH2D(I, J, NX, NY) \
    PRAGMA_FOR_SIMD \
_Pragma("omp parallel for") \
for(int J=0; J<NY; J++) \
PRAGMA_FOR_SIMD \
for(int I=0; I<NX; I++) 

#define FOR_EACH1D(IJK, NN) \
    PRAGMA_FOR_SIMD \
for(int IJK=0; IJK<NN; IJK++) 

#define SKIP_FOR() continue
#endif


namespace kernel {

    struct backend {};

    struct openmp    : backend {};
    struct cuda      : backend {};


#ifdef USE_NVCC
    using opti = cuda;
#else
    using opti = openmp;
#endif

#ifdef USE_NVCC
    template<class Func, class... Args> 
        __global__ void exec2d_gpu(
                Func    func,
                Args... args
                ) 
        {
            func(args...);
        } 
#endif


    template<class Func, class... Args> 
        void exec2d_cpu(
                Func    func,
                Args... args
                )
        {
            func(args...);
        }


    template<class ExecutionPolicy, class Func, class... Args>
        void exec2d(
                const int nx,
                const int ny,
                Func    func,
                Args... args
                )
        {
            if (std::is_same<ExecutionPolicy, cuda>::value) {
#ifdef USE_NVCC
                constexpr int nth_x = defines::blocksizex, nth_y = defines::blocksizey;
                const int bx = (nx + nth_x - 1) / nth_x;
                const int by = (ny + nth_y - 1) / nth_y;
                exec2d_gpu<Func, Args...> <<<
                    dim3(bx, by), 
                    dim3(nth_x, nth_y)
                        >>> (func, args...);
                cudaDeviceSynchronize();
#endif
            }
            else if (std::is_same<ExecutionPolicy, openmp>::value) {
                exec2d_cpu<Func, Args...>(func, args...);
            }
            else {
                static_assert(std::is_base_of<backend, ExecutionPolicy>::value, "unexpected Execution Policy");
            }
        }

    //} // namespace foreach

    //namespace  kernel{
__HD__ 
void DiffusionEq(real* fn, const real* f){
    FOR_EACH2D(i,j,defines::nx,defines::ny){
        //for(int j=0; j<defines::ny; j++) {
        //const int j = blockIdx.y*blockDim.y + threadIdx.y;
        const int jm = (j-1 + defines::ny) % defines::ny;
        const int jp = (j+1 + defines::ny) % defines::ny;
        //for(int i=0; i<defines::nx; i++) {
        //const int i = blockIdx.x*blockDim.x + threadIdx.x;
        const int im = (i-1 + defines::nx) % defines::nx;
        const int ip = (i+1 + defines::nx) % defines::nx;

        const int ji  = Index::index_xy(i,j);
        const int jim = Index::index_xy(im,j);
        const int jip = Index::index_xy(ip,j);
        const int jim2 = Index::index_xy(i,jm);
        const int jip2 = Index::index_xy(i,jp);

        //printf("%d %d %d %d \n",i, jm,j,jp);
        if(j<defines::ny and i<defines::nx){ 
            fn[ji] = f[ji]
                + defines::coef_diff * (f[jim] - 4*f[ji] + f[jip] + f[jim2] + f[jip2] );
        }
    }
    };
    };

#endif
