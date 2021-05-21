#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <string>

#include "ValuesDiffusion.h"

#include "defines.h"
#include "Output.h"


void Output::
setout(const ValuesDiffusion v, const int t) {

     float f_ave,f_max;
      f_ave = average(f_ave,v);
      f_max = maximum(f_max,v);

    std::ostringstream oss;
    oss << "data/t=" << std::setfill('0') << std::right<< std::setw(4)<<  t << ".dat";
        
    std::ofstream ofs(oss.str());
     ofs << "#t=" << std::setw(8) << t << "    x,    f,      average,	maximum" << std ::endl;

    for(int i=0; i<defines::nx; i++) {
     ofs << std::setw(8) << v.xx()[i] << " "
         << std::setw(8) << v.ff()[i] << " "
         << std::setw(8) << f_ave     << " "  
         << std::setw(8) << f_max     << " " << std::endl;
    }

    ofs.close(); 
   
    std::ostringstream oss2;
    oss2 << "data/t=" << std::setfill('0') << std::right<< std::setw(4)<< t << "_downsize"<< defines::downsize << ".dat";

     std::ofstream ofs2(oss2.str());
     ofs2 << "#t=" << std::setw(8) << t << "    x,    f,      average,   maximum,	downsize =" << defines::downsize << std ::endl;

    for(int i=0; i<defines::nx; i++) {
     if(  i % defines::downsize  == 0){
     ofs2 << std::setw(8) << v.xx()[i] << " "
         << std::setw(8) << v.ff()[i] << " "
         << std::setw(8) << f_ave     << " "
         << std::setw(8) << f_max     << " " << std::endl;
     }
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




