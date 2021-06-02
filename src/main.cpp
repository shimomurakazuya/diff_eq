#ifdef _OPENMP
#include <omp.h>
#endif

#include "defines.h"
#include "ValuesDiffusion.h"
#include "Output.h"


int main() {
    ValuesDiffusion v (defines::nx,defines::ny,defines::ncell);
    ValuesDiffusion vn(defines::nx,defines::ny,defines::ncell);
    Output output;
    real st_time, ed_time;

    v .allocate_values();
    vn.allocate_values();

    v .init_values();
    vn.init_values();

    omp_set_num_threads(defines::thread_num);

   st_time=omp_get_wtime();

    // main loop
    for(int t=0; t<defines::iter; t++) {
        // output
        if(t % defines::iout == 0) {
            const int fout_step = t / defines::iout;
            output.print_sum(v,fout_step);
            output.OutputDiffusionData(v ,fout_step);
        }
        ValuesDiffusion::swap(&vn, &v);
        vn.time_integrate(v);
    }
    ed_time=omp_get_wtime();
       output.print_elapsetime(st_time,ed_time);
    
}
