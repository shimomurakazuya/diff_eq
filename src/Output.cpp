#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <string>

#include "ValuesDiffusion.h"
#include "Index.h"
#include "defines.h"
#include "Output.h"


void Output::
OutputDiffusionData(const ValuesDiffusion& v, const int t) {

    double f_ave,f_max,f_ana;
    f_ave = average(v);
    f_max = maximum(v);

    std::ostringstream oss;
      oss << "data/ascii_value_step" << std::setfill('0') << std::right<< std::setw(4)<<  t << ".dat";
 
     std::ofstream ofs(oss.str());
      ofs << "#t=" << std::setw(8) << t << "    x,    f,      average,   maximum" << std ::endl;
 
     for(int i=0; i<defines::nx; i++) {
        for(int j=0; j<defines::ny; j++) {
            const int ij = index::index_xy(i,j);
              f_ana = defines::fmax * cos(v.xx()[i]/defines::lx*2.0*M_PI + v.yy()[j]/defines::lx*2.0*M_PI) 
                  * exp(-defines::c_dif * ((2.0 * M_PI/defines::lx)* (2.0*M_PI/defines::lx) + (2.0 * M_PI/defines::lx)* (2.0*M_PI/defines::lx) )* t * defines::dt *defines::iout  );
            ofs << std::setw(8) << v.xx()[i] << " "
                << std::setw(8) << v.yy()[j] << " "
                << std::setw(8) << v.ff()[ij] << " "
                << std::setw(8) << f_ana     << " "
                << std::setw(8) << f_ave     << " "
                << std::setw(8) << f_max     << " " << std::endl;
     }
        ofs << std::endl;
     }


        
    std::ostringstream oss2;
      oss2 << "data/ascii_value_step" << std::setfill('0') << std::right<< std::setw(4)<< t << "_downsize"<< defines::downsize << ".dat";
 
     std::ofstream ofs2(oss2.str());
      ofs2 << "#t=" << std::setw(8) << t << "    x,    f,      average,   maximum,   downsize =" << defines::downsize << std ::endl;
 
      for(int i=0; i<defines::nx; i+= defines::downsize) {
        for(int j=0; j<defines::ny; j+= defines::downsize) {
            const int ij = index::index_xy(i,j);
            ofs2 << std::setw(8) << v.xx()[i] << " "
                << std::setw(8) << v.yy()[j] << " "
                << std::setw(8) << v.ff()[ij] << " "
                << std::setw(8) << f_ave     << " "
                << std::setw(8) << f_max     << " " << std::endl;
 
      } 
      }
}


real Output::
average(const ValuesDiffusion& v){
    real f_ave = 0 ;
    for(int i=0; i<defines::ncell; i++) {
        f_ave = f_ave + v.ff()[i]/defines::ncell;
    }
    return f_ave;
}

real Output::
maximum(const ValuesDiffusion& v){
    real f_max = 0 ;
    for(int i=0; i<defines::ncell; i++) { 
        f_max = std::fmax(f_max , v.ff()[i]);
    }
    return f_max;
}

real Output::
 analytic( const ValuesDiffusion& v,int t){
      real f_ana = 0;
      real err_elm =0;
      for(int i=0; i<defines::nx; i++) {
          for(int j=0; j<defines::nx; j++) {
              const int ij = index::index_xy(i,j);
              f_ana = defines::fmax * cos(v.xx()[i]/defines::lx*2.0*M_PI + v.yy()[j]/defines::lx*2.0*M_PI) 
                  * exp(-defines::c_dif * ((2.0 * M_PI/defines::lx)* (2.0*M_PI/defines::lx) + (2.0 * M_PI/defines::lx)* (2.0*M_PI/defines::lx) )* t * defines::dt *defines::iout  );
                  err_elm = err_elm + fabs(f_ana - v.ff()[ij])/defines::ncell;
                  }
                  }
                  return err_elm;
}


void Output::
print_sum(const ValuesDiffusion& v, const int t) {
    double sum = 0, mmm = v.ff()[0];
    real err_nlm = 0;
    err_nlm = analytic(v,t); 
    for(int i=0; i<defines::ncell; i++) {
        sum += v.ff()[i];
        mmm = std::fmax(mmm, v.ff()[i]);
    }
    real Tout = t * defines::dt *defines::iout;
    std::cout << "t=" << std::setw(8) << Tout
        << " :    " << std::setw(8) << sum
        << " :    " << std::setw(8) << err_nlm
        << " ,    " << std::setw(8) << mmm << std::endl;
}
