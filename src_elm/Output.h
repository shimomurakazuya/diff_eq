#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "ValuesDiffusion.h"

class Output{
public   :
      void setout(const ValuesDiffusion& v, int t);
      float average(const ValuesDiffusion& v);
      float maximum(const ValuesDiffusion& v);
      float analytic(const ValuesDiffusion& v, int t);
      void  print_sum(const int t, const ValuesDiffusion& v);


};
#endif    





