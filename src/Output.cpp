#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <string>
#include <math.h>

#include "ValuesDiffusion.h"

#include "defines.h"
#include "Output.h"


void Output::
setout(const ValuesDiffusion& v, const int t) {

    float f_ave,f_max,f_ana, err_elm;
     f_ave = average(v);
     f_max = maximum(v);
     err_elm = analytic(v,t);


    std::ostringstream oss;
     oss << "data/ascii_value_step" << std::setfill('0') << std::right<< std::setw(4)<<  t << ".dat";
        
    std::ofstream ofs(oss.str());
     ofs << "#t=" << std::setw(8) << t << "    x,    f,      average,	maximum, err_elm" << std ::endl;

    for(int i=0; i<defines::nx; i++) {
      f_ana = defines::fmax * cos(v.xx()[i]/defines::lx*2.0*M_PI) * exp(- defines::c_dif* (2.0 * M_PI/defines::lx)* (2.0 * M_PI/defines::lx) * double(t)* defines::dt *defines::iout  );
     ofs << std::setw(10) << v.xx()[i] << "  "
         << std::setw(10) << v.ff()[i] << "  "
         << std::setw(10) << f_ana     << "  "
         << std::setw(10) << f_ave     << "  "  
         << std::setw(10) << f_max     << "  "  
         << std::setw(10) << err_elm     << " " << std::endl;
    }

    ofs.close(); 
   
    std::ostringstream oss2;
     oss2 << "data/ascii_value_step" << std::setfill('0') << std::right<< std::setw(4)<< t << "_downsize"<< defines::downsize << ".dat";

    std::ofstream ofs2(oss2.str());
     ofs2 << "#t=" << std::setw(8) << t << "    x,    f,      average,   maximum,	downsize =" << defines::downsize << std ::endl;

     for(int i=0; i<defines::nx; i+= defines::downsize) {
     ofs2 << std::setw(8) << v.xx()[i] << " "
          << std::setw(8) << v.ff()[i] << " "
          << std::setw(8) << f_ave     << " "
          << std::setw(8) << f_max     << " " << std::endl;
     
    }

}

float Output::
average( const ValuesDiffusion& v){
     real f_ave = 0 ;
     for(int i=0; i<defines::nx; i++) {
      f_ave = f_ave + v.ff()[i]/defines::nx;
      }
     return f_ave;
}

float Output::
maximum( const ValuesDiffusion& v){
     real f_max = 0; 
     for(int i=0; i<defines::nx; i++) { 
      f_max = std::fmax(f_max , v.ff()[i]);
      }
     return f_max;
}


float Output::
analytic( const ValuesDiffusion& v,int t){
     real f_ana = 0; 
     real err_elm =0;
     for(int i=0; i<defines::nx; i++) { 
      f_ana = defines::fmax * cos(v.xx()[i]/defines::lx*2.0*M_PI) * exp(-defines::c_dif * (2.0 * M_PI/defines::lx)* (2.0*M_PI/defines::lx)* t * defines::dt *defines::iout  );
      err_elm = err_elm + fabs(f_ana - v.ff()[i]);
     }
     return err_elm;
}

void Output::
print_sum(const int t, const ValuesDiffusion& v) {
     double sum = 0, mmm = v.ff()[0];
     for(int i=0; i<defines::nx; i++) {
         sum += fabs(v.ff()[i]);
         mmm = std::fmax(mmm, v.ff()[i]);
     }
     float err_elm, elm, err_num ;
     err_elm = analytic(v,t);
     elm = err_elm/sum;
     err_num = err_elm/defines::nx;
     float Tout = t * defines::dt * defines::iout;

     std::cout << "t=" << std::setw(8) << Tout
         << " :    " << std::setw(8) << mmm
         << " :    " << std::setw(8) << sum
         << " :    " << std::setw(8) << err_elm
         << " :    " << std::setw(8) << err_num
         << " ,    " << std::setw(8) << elm << std::endl;
}

