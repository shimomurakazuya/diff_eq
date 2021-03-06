//#ifdef _OPENMP
//#include <omp.h>
//#endif
    
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

    real f_ave,f_max;
    real f_ana, diff;
    f_ave = average(v);
    f_max = maximum(v);
    real time = double(t)* defines::dt *defines::iout;

    char filename[1024];
    sprintf(filename, "data/ascii_value_step%04d.dat",t ) ;
    FILE* fp = fopen( filename ,"w");
    fprintf(fp,"#t = %04d    x,   y,    f,      average, maximum \n",t  ); 

    for(int j=0; j<defines::ny; j++) {
        for(int i=0; i<defines::nx; i++) {
            const int ji = Index::index_xy(i,j);
            fprintf(fp,"%8.3f %8.3f %8.8f %8.3f %8.3f\n",v.xx()[i],v.yy()[j], v.ff()[ji], f_ave, f_max ); 
        }
        fprintf(fp,"\n" ); 
    }    

    char filename2[1024];
    sprintf(filename2, "data/ascii_value_step%04d_downsize%01d.dat",t,defines::downsize ) ;
    FILE* fp2 = fopen( filename2 ,"w");
    fprintf(fp2,"#t = %8.8f    x,   y,    f,      average, maximum \n",time  ); 

    for(int j=0; j<defines::ny; j+=defines::downsize) {
        for(int i=0; i<defines::nx; i+= defines::downsize) {
            const int ji = Index::index_xy(i,j);
            f_ana = defines::fmax * cos(v.xx()[i]/defines::lx*2.0*M_PI + v.yy()[j]/defines::lx*2.0*M_PI) * exp(- defines::c_dif* ((2.0 * M_PI/defines::lx)* (2.0 * M_PI/defines::lx)+ (2.0 * M_PI/defines::lx)* (2.0 * M_PI/defines::lx) ) * double(t)* defines::dt *defines::iout  );
            diff = f_ana - v.ff()[ji];
            fprintf(fp2,"%8.3f %8.3f %8.8f %8.3f %8.3f %10.8f  %10.8f \n",v.xx()[i],v.yy()[j], v.ff()[ji], f_ave, f_max,f_ana, diff ); 
        }
        fprintf(fp2,"\n" ); 
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

void Output::
parameter(){

//#pragma omp single  
    //printf("nx = %d, ny= %d, iter = %d, \n dx = %lf, nthread = %d, nthread_max = %d", defines::nx, defines::ny, defines::iter, defines::dx, defines::thread_num, omp_get_max_threads());
    printf("nx = %d, ny= %d, iter = %d, \n dx = %lf \n, nthread = %d \n", defines::nx, defines::ny, defines::iter, defines::dx, defines::thread_num);
    //printf("nx = %d, ny= %d, iter = %d, \n dx = %lf, nthread = %d", defines::nx, defines::ny, defines::iter, defines::dx, defines::thread_num);
}

void Output::
print_sum(const ValuesDiffusion& v, const int t) {
    double sum = 0, mmm = v.ff()[0];
    for(int i=0; i<defines::ncell; i++) {
        sum += v.ff()[i];
        mmm = std::fmax(mmm, v.ff()[i]);
    }
    std::cout << "t=" << std::setw(8) << t
       << " :    " << std::setw(8) << sum
       << " ,    " << std::setw(8) << mmm << std::endl;
}

void Output::
print_elapsetime(const double st_time,const double ed_time, const double ave_time){
    double elapse_time, bw;
    double cal_ratio, gf; 
    elapse_time = ed_time - st_time;
    bw = 1e-09* defines::data_num * 8 * defines::ncell / ave_time;
    gf = 1e-09* defines::flop_num * defines::ncell / ave_time;    
    cal_ratio = gf/ defines::loof_line;

    //    printf ("    elapse time = %lf \n ave time = %lf \n number_thread = %d \n",elapse_time,ave_time, defines::thread_num);
    std:: cout << "elapse time = " << std::setw(8) << elapse_time
        << ", elapse time loop avetage  = " << std::setw(8) << ave_time << std::endl
        << "thread_num =" << defines::thread_num << ",  peak Flops = "  <<defines::loof_line  << std::endl
        << "bandwidth(GB/s) = "<< bw <<", Flops(Gflops) = "<< gf << std::endl
        << "calcuration ratio = "<< cal_ratio << std::endl; 
}


void Output::
print_elapsetime_loop(const double st_time,const double ed_time){
    double elapse_time;
    elapse_time = ed_time - st_time; 

    printf ("    elapse time loop = %lf \n ",elapse_time);
//    std::cout << "elapse time loop = " << std::setw(8) << elapse_time << std::endl
//        << "number_thread = " << std::setw(8) <<defines::thread_num
//        << std::endl; 
}



