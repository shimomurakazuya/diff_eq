#ifndef VALUES_DIFFUSION_H_
#define VALUES_DIFFUSION_H_

class ValuesDiffusion {
private:
    float* x_;
    float* f_;
    const int nx_;

public:
    float*  xx()   { return  x_; };
    float*  ff()   { return  f_; };

    const float*  xx()  const { return  x_; };
    const float*  ff()   const { return  f_; };

    ValuesDiffusion() = delete;
    ValuesDiffusion(int nx): nx_(nx) {}
    ValuesDiffusion(const ValuesDiffusion&) = default;
    
    void allocate_values();
    void deallocate_values();
    void init_values();
    void time_integrate(const ValuesDiffusion& valuesDiffusion);
    void print_sum(int t);
    static void swap(ValuesDiffusion* v0, ValuesDiffusion* v1);

private:
    void copy_values(const ValuesDiffusion& valuesDiffusion);

};

#endif

