#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "ValuesDiffusion.h"
#include "Index.h"

class Output{

public   :
      void OutputDiffusionData( const ValuesDiffusion& v, const ValuesDiffusion& vn, int t);
      real average(const ValuesDiffusion& v);
      real maximum(const ValuesDiffusion& v);
      real analytic(const ValuesDiffusion& v, int t);
      void print_sum(const ValuesDiffusion& v_gpu, const ValuesDiffusion& v_cpu ,int t);
      void parameter();
      void print_elapsetime(const double sttime, const double edtime,const double avetime);
      //void print_elapsetime();
      void print_elapsetime_loop(const double sttime, const double edtime);
};
#endif    





