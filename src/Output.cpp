#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <string>

#include "ValuesDiffusion.h"
#include "Index.h"
#include "defines.h"
#include "Output.h"


void Output::
OutputDiffusionData(const ValuesDiffusion& v, const int t) {

    float f_ave,f_max;
        f_ave = average(v);
        f_max = maximum(v);

    std::ostringstream oss;
        oss << "data/ascii_value_step" << std::setfill('0') << std::right<< std::setw(4)<<  t << ".dat";
        
    std::ofstream ofs(oss.str());
        ofs << "#t=" << std::setw(8) << t << "    x,   y,    f,      average,	maximum" << std ::endl;

    for(int i=0; i<defines::nx; i++) {
        for(int j=0; j<defines::ny; j++) {
            const int ij = index::index_xy(i,j);
              ofs << std::setw(8) << v.xx()[i]  << " "
                  << std::setw(8) << v.yy()[j]  << " "
                  << std::setw(8) << v.ff()[ij] << " "
                  << std::setw(8) << f_ave      << " "  
                  << std::setw(8) << f_max      << " " << std::endl;
            }
         ofs << std::endl;
    }
    ofs.close(); 
   
    std::ostringstream oss2;
        oss2 << "data/ascii_value_step" << std::setfill('0') << std::right<< std::setw(4)<< t << "_downsize"<< defines::downsize << ".dat";

    std::ofstream ofs2(oss2.str());
        ofs2 << "#t="  << std::setw(8) << t << "    x,  y,    f,      average,   maximum,	downsize =" << defines::downsize << std ::endl;

    for(int i=0; i<defines::nx; i+= defines::downsize ) {
        for(int j=0; j<defines::ny; j++) {
            const int ij = index::index_xy(i,j); 
                ofs2 << std::setw(8) << v.xx()[i]  << " "
                     << std::setw(8) << v.yy()[j]  << " "
                     << std::setw(8) << v.ff()[ij] << " "
                     << std::setw(8) << f_ave      << " "
                     << std::setw(8) << f_max      << " " << std::endl;
         }
       }

}

real Output::
average(const ValuesDiffusion& v){
    real f_ave = 0 ;
        for(int i=0; i<defines::ncell; i++) {
            f_ave = f_ave + v.ff()[i]/defines::nx;
        }
    return f_ave;
}

real Output::
maximum(const ValuesDiffusion& v){
    real f_max = 0; 
        for(int i=0; i<defines::ncell; i++) { 
            f_max = std::fmax(f_max , v.ff()[i]);
        }
    return f_max;
}

void Output::
print_sum(const ValuesDiffusion& v, const int t) {
    double sum = 0, mmm = v.ff()[0];
      for(int i=0; i<defines::ncell; i++) {
            sum += v.ff()[i];
            mmm = std::fmax(mmm, v.ff()[i]);
            }
           std::cout << "t="     << std::setw(8) << t
                     << " :    " << std::setw(8) << sum
                     << " ,    " << std::setw(8) << mmm << std::endl;
}
