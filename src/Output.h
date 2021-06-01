#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "ValuesDiffusion.h"
#include "Index.h"

class Output{
public   :
      void OutputDiffusionData( const ValuesDiffusion& v, int t);
      real average(const ValuesDiffusion& v);
      real maximum(const ValuesDiffusion& v);
      real analytic(const ValuesDiffusion& v,int t);
      void print_sum(const ValuesDiffusion& v, int t);

};
#endif    





