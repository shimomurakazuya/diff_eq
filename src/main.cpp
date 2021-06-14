#ifdef _OPENMP
#include <omp.h>
#endif
 #include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>


#include "defines.h"
#include "ValuesDiffusion.h"
#include "Output.h"


int main() {
    ValuesDiffusion v (defines::nx,defines::ny,defines::ncell);
    ValuesDiffusion vn(defines::nx,defines::ny,defines::ncell);
    Output output;
    real st_time, ed_time;
    real st_tloop, ed_tloop,ave_tloop=10,ave_tloop_min=10;

    v .allocate_values();
    vn.allocate_values();

    v .init_values();
    vn.init_values();

    output.parameter();
    omp_set_num_threads(defines::thread_num);

    st_time=omp_get_wtime();

    // main loop
    for(int t=0; t<defines::iter; t++) {
        // output
        if(t % defines::iout == 0) {
            printf("sec_ave=%lf \n",ave_tloop);
                        ave_tloop_min = std::min(ave_tloop_min, ave_tloop);
                        printf("sec_min=%lf \n",ave_tloop_min);
                        ave_tloop =0;

            const int fout_step = t / defines::iout;
          //  output.print_sum(v,fout_step);
            output.OutputDiffusionData(v ,fout_step);
        }
        ValuesDiffusion::swap(&vn, &v);
        st_tloop=omp_get_wtime();
        vn.time_integrate(v);
        ed_tloop=omp_get_wtime();
 
        ave_tloop = ave_tloop +  (ed_tloop - st_tloop)/defines::iout; 
        if(t % defines::iout == 0) {
            output.print_elapsetime_loop(st_tloop,ed_tloop);
        }
    }
    ed_time=omp_get_wtime();
    output.print_elapsetime(st_time,ed_time, ave_tloop_min);
    
}
