#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <string>


// define.h
using real = float;
namespace defines {
    // x, t
    constexpr real dx = 1.0;
    constexpr int nx = 51;
    constexpr int nc = nx / 2;
    //constexpr real lx = nx - 1;
//    constexpr int iter = 100000;
//    constexpr int iout = 100;
    constexpr int iter = 1000;
    constexpr int iout = 10;
    constexpr real dt = 0.001; // torima

    // diffusion
    constexpr real fmax = 1;
    constexpr real delta = 4 * dx; // 4 dx
    constexpr real c_dif = 1;
    constexpr real pi =3.14;
};

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
    void output(int t);
    static void swap(ValuesDiffusion* v0, ValuesDiffusion* v1);

private:
    void copy_values(const ValuesDiffusion& valuesDiffusion);

};

void ValuesDiffusion::
allocate_values() {
    x_ = (real*)std::malloc(sizeof(real) * nx_);
    f_ = (real*)std::malloc(sizeof(real) * nx_);
}

void ValuesDiffusion::
deallocate_values() {
    std::free(x_);
    std::free(f_);
}

void ValuesDiffusion::
init_values() {
    for(int i=0; i<nx_; i++) {
        const real xi = (i - defines::nc) * defines::dx / defines::delta;
//        const real xi = (i - defines::nc) * defines::dx;
        x_[i] = xi;
        f_[i] = defines::fmax * std::exp( - xi*xi);
//        f_[i] = defines::fmax * cos(defines::pi * xi/ (defines::dx * defines::nc));
    }
}

void ValuesDiffusion::
time_integrate(const ValuesDiffusion& valuesDiffusion) {
    real* fn = f_;
    const real* f = valuesDiffusion.f_;
    for(int i=0; i<defines::nx; i++) {
        const int im = (i-1 + defines::nx) % defines::nx;
        const int ip = (i+1 + defines::nx) % defines::nx;
        fn[i] = f[i] + 
            + defines::c_dif * defines::dt / defines::dx / defines::dx * (f[im] - 2*f[i] + f[ip]);
    }
}

void ValuesDiffusion::
copy_values(const ValuesDiffusion& valuesDiffusion) {
    x_ = valuesDiffusion.x_;
    f_ = valuesDiffusion.f_;
}

void ValuesDiffusion::
swap(ValuesDiffusion* v0, ValuesDiffusion* v1) {
    ValuesDiffusion tmp(*v0);
    v0->copy_values(*v1);
    v1->copy_values(tmp);
}

void ValuesDiffusion::
print_sum(const int t) {
    double sum = 0, mmm = f_[0];
    for(int i=0; i<defines::nx; i++) {
        sum += f_[i];
        mmm = std::fmax(mmm, f_[i]);
    }
    std::cout << "t=" << std::setw(8) << t 
        << " :    " << std::setw(8) << sum 
        << " ,    " << std::setw(8) << mmm << std::endl;

}

void ValuesDiffusion::
//output(const ValuesDiffusion &valuediffusion, const const int t) {
output( const int t) {

    std::ostringstream oss;
//    oss << "data/t=" << std::setfill("0")   << std::right << std::setw(4) <<t << ".dat";
    oss << "data/t=" <<t << ".dat";
    std::ofstream ofs(oss.str());
     ofs << "#t=" << std::setw(8) << t << "    x,    f" <<std::endl;
    for(int i=0; i<defines::nx; i++) {
     ofs << std::setw(8) << x_[i] << "  " 
          << std::setw(8) << f_[i] << std::endl ;
}
}

class Output{
private  :
    float* x__;
    float* f__;
    ValuesDiffusion* v__;
    float* f_ave__;

public   :

//     const ValuesDiffusion* v;
      void setout(ValuesDiffusion v, int t);
//      void setout(float* f, float* x , int t);
      float average(float f_ave, ValuesDiffusion v);
      float maximum(float f_ave, ValuesDiffusion v);

};

void Output::
setout(const ValuesDiffusion v, const int t) {
//setout(const float* f, const float*x const int t) {

     float f_ave,f_max;
      f_ave = average(f_ave,v);
      f_max = maximum(f_max,v);
    std::ostringstream oss;
    oss << "data/t=" << t << ".dat";
        
    std::ofstream ofs(oss.str());
     ofs << "#t=" << std::setw(8) << t << "    x,    f,      average,	maximum" << std ::endl;

    for(int i=0; i<defines::nx; i++) {
     ofs << std::setw(8) << v.xx()[i] << " "
         << std::setw(8) << v.ff()[i] << " "
         << std::setw(8) << f_ave     << " "  
         << std::setw(8) << f_max     << " " << std::endl;
    }
}

float Output::
average( float f_ave, const ValuesDiffusion v){
       f_ave = 0 ;
      for(int i=0; i<defines::nx; i++) {
      f_ave = f_ave + v.ff()[i]/defines::nx;
      }
 return f_ave;
}

float Output::
maximum( float f_max, const ValuesDiffusion v){
     f_max = 0; 
     for(int i=0; i<defines::nx; i++) { 
      f_max = std::fmax(f_max , v.ff()[i]);
      }
   return f_max;
}

int main() {
    ValuesDiffusion v (defines::nx);
    ValuesDiffusion vn(defines::nx);
    Output output;

    v .allocate_values();
    vn.allocate_values();

    v .init_values();
    vn.init_values();

    // main loop
    for(int t=0; t<defines::iter; t++) {
        // output
        if(t % defines::iout == 0) {
            const int fout_step = t / defines::iout;
            v.print_sum(fout_step);
//            v.output(fout_step);
            output.setout(v ,fout_step);
        }

        ValuesDiffusion::swap(&vn, &v);
        vn.time_integrate(v);
    }

}
