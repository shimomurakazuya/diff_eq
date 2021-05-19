#include "defines.h"
#include "ValuesDiffusion.h"

int main() {
    ValuesDiffusion v (defines::nx);
    ValuesDiffusion vn(defines::nx);

    v .allocate_values();
    vn.allocate_values();

    v .init_values();
    vn.init_values();

    // main loop
    for(int t=0; t<defines::iter; t++) {
        // output
        if(t % defines::iout == 0) {
            const int fout_step = t / defines::iout;
            v.print_sum(fout_step);
        }

        // 
        ValuesDiffusion::swap(&vn, &v);
        vn.time_integrate(v);
    }

}
