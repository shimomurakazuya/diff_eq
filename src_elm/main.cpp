#include "defines.h"
#include "ValuesDiffusion.h"
#include "Output.h"


int main() {
    ValuesDiffusion v (defines::nx);
    ValuesDiffusion vn(defines::nx);
    Output output;

    v .allocate_values();
    vn.allocate_values();

    v .init_values();
    vn.init_values();

    // main loop
    for(int t=0; t<defines::iter; t++) {
        // output
        if(t % defines::iout == 0) {
            const int fout_step = t / defines::iout;
            output.print_sum(fout_step,v);
            output.setout(v ,fout_step);
        }
         
        ValuesDiffusion::swap(&vn, &v);
        vn.time_integrate(v);
    }

}
