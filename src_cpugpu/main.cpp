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
    ValuesDiffusion v_cpu (defines::nx,defines::ny,defines::ncell);
    ValuesDiffusion vn_cpu(defines::nx,defines::ny,defines::ncell);
    ValuesDiffusion v_gpu (defines::nx,defines::ny,defines::ncell);
    ValuesDiffusion vn_gpu(defines::nx,defines::ny,defines::ncell);
    Output output;
    real st_time, ed_time;
    real st_tloop, ed_tloop,ave_tloop=10,ave_tloop_min=10;
    v_cpu .allocate_values();
    vn_cpu.allocate_values();
    v_gpu .allocate_values();
    vn_gpu.allocate_values();

    v_cpu .init_values();
    vn_cpu.init_values();
    v_gpu .init_values();
    vn_gpu.init_values();

    omp_set_num_threads(defines::thread_num);

    //output.parameter();
    st_time = omp_get_wtime();
        // main loop
        for(int t=0; t<defines::iter; t++) {
            ValuesDiffusion::swap(&vn_cpu, &v_cpu);
            ValuesDiffusion::swap(&vn_gpu, &v_gpu);
            // output
            if(t % defines::iout == 0) {
                printf("sec_ave=%lf \n",ave_tloop);
                ave_tloop_min = std::min(ave_tloop_min, ave_tloop);
                printf("sec_min=%lf \n",ave_tloop_min);
                ave_tloop =0;
                const int fout_step = t / defines::iout;
                v_gpu.d2h();
                output.print_sum(v_gpu,v_cpu, fout_step);
                output.OutputDiffusionData(v_gpu ,v_cpu, fout_step);
            }
            st_tloop=omp_get_wtime();
            vn_cpu.time_integrate_cpu(v_cpu);
            vn_gpu.time_integrate_gpu(v_gpu);
            ed_tloop=omp_get_wtime();

            ave_tloop = ave_tloop +  (ed_tloop - st_tloop)/defines::iout; 
            if(t % defines::iout == 0) {
                output.print_elapsetime_loop(st_tloop,ed_tloop);
            }
        }
    ed_time=omp_get_wtime();
    output.print_elapsetime(st_time,ed_time, ave_tloop_min);

}
