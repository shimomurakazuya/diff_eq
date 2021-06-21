//#ifdef _OPENMP
//#include <omp.h>
//#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>


#include "defines.h"
#include "ValuesDiffusion.h"
#include "Output.h"


int main() {
   printf("0");
    ValuesDiffusion v (defines::nx,defines::ny,defines::ncell);
    ValuesDiffusion vn(defines::nx,defines::ny,defines::ncell);
    Output output;
//    real st_time, ed_time;
//    real st_tloop, ed_tloop,ave_tloop=10,ave_tloop_min=10;
   printf("1");
    v .allocate_values();
    vn.allocate_values();
   printf("2");

    v .init_values();
    vn.init_values();
   printf("3");

//    output.parameter();
   printf("4\n");

    // main loop
    for(int t=0; t<defines::iter; t++) {
        // output
        if(t % defines::iout == 0) {
//            printf("sec_ave=%lf \n",ave_tloop);
//            ave_tloop_min = std::min(ave_tloop_min, ave_tloop);
//            printf("sec_min=%lf \n",ave_tloop_min);
//            ave_tloop =0;
            const int fout_step = t / defines::iout;
            //  output.print_sum(v,fout_step);
   printf("5\n");
            output.OutputDiffusionData(v ,fout_step);
   printf("6\n");
        }
        ValuesDiffusion::swap(&vn, &v);
//        st_tloop=omp_get_wtime();
        vn.time_integrate(v);
//        ed_tloop=omp_get_wtime();

//        ave_tloop = ave_tloop +  (ed_tloop - st_tloop)/defines::iout; 
//        if(t % defines::iout == 0) {
//            output.print_elapsetime_loop(st_tloop,ed_tloop);
//        }
    }
//    ed_time=omp_get_wtime();
//    output.print_elapsetime(st_time,ed_time, ave_tloop_min);

}
