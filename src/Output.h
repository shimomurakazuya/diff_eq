#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "ValuesDiffusion.h"

class Output{
private  :
    float* x__;
    float* f__;
    ValuesDiffusion* v__;
    float* f_ave__;

public   :

//    const ValuesDiffusion* v;
//    void allocate_values();
//    void deallocate_values();
      void setout(ValuesDiffusion v, int t);
//     void setout(float* f, float* x , int t);
      float average(float f_ave, ValuesDiffusion v);
      float maximum(float f_ave, ValuesDiffusion v);

};
#endif    





