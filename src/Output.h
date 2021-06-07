#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "ValuesDiffusion.h"
#include "Index.h"

class Output{

public   :
      void OutputDiffusionData( const ValuesDiffusion& v, int t);
      real average(const ValuesDiffusion& v);
      real maximum(const ValuesDiffusion& v);
      void print_sum(const ValuesDiffusion& v, int t);
      void print_elapsetime(const double sttime, const double edtime,const double avetime);
      //void print_elapsetime();
      void print_elapsetime_loop(const double sttime, const double edtime);
};
#endif    





