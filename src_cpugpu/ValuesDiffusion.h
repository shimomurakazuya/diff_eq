#ifndef VALUES_DIFFUSION_H_
#define VALUES_DIFFUSION_H_

#include "defines.h"
#include "Index.h"

class ValuesDiffusion {
private:
    real* x_;
    real* y_;
 //   real* f_ =  aligned_alloc( defines::alignment, ncell_*sizeof(real));
    real* f_;
    real* d_f_;
    const int nx_;
    const int ny_;
    const int ncell_;

public:
    real*  xx()   { return  x_; };
    real*  yy()   { return  y_; };
    real*  ff()   { return  f_; };
    real*  d_f()  { return  d_f_;  };

    const real*  xx()  const { return  x_; };
    const real*  yy()  const { return  y_; };
    const real*  ff()   const { return  f_; };
    const real*  d_f()   const { return  d_f_; };

    ValuesDiffusion() = delete;
    ValuesDiffusion(int nx, int ny, int ncell): nx_(nx), ny_(ny), ncell_(ncell) {}
    ValuesDiffusion(const ValuesDiffusion&) = default;
    
    void allocate_values();
    void deallocate_values();
    void init_values();
    void d2h();
    void time_integrate_gpu(const ValuesDiffusion& valuesDiffusion);
    void time_integrate_cpu(const ValuesDiffusion& valuesDiffusion);
    void print_sum(int t);
    static void swap(ValuesDiffusion* v0, ValuesDiffusion* v1);
    


private:
    void copy_values(const ValuesDiffusion& valuesDiffusion);

};

#endif

