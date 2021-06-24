#ifdef _OPENMP
#include <omp.h>
#endif

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
OutputDiffusionData(const ValuesDiffusion& v_gpu, const ValuesDiffusion& v_cpu,  const int t) {

    real f_ave,f_max;
    real f_ana, diff;
    f_ave = average(v_gpu);
    f_max = maximum(v_gpu);
    real time = double(t)* defines::dt *defines::iout;
    //real err_elm = analytic(v,t);

    char filename[1024];
    sprintf(filename, "data/ascii_value_step%04d.dat",t ) ;
    FILE* fp = fopen( filename ,"w");
    fprintf(fp,"#t = %04d    x,   y,    f,      average, maximum \n",t  ); 

    for(int j=0; j<defines::ny; j++) {
        for(int i=0; i<defines::nx; i++) {
            const int ji = Index::index_xy(i,j);
            diff = v_cpu.ff()[ji] - v_gpu.ff()[ji];
            fprintf(fp,"%8.3f %8.3f %8.8lf %8.8lf %20.20e  %8.3f %8.3f\n",v_gpu.xx()[i],v_gpu.yy()[j], v_cpu.ff()[ji], v_gpu.ff()[ji], diff, f_ave, f_max ); 
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
            diff = v_cpu.ff()[ji] - v_gpu.ff()[ji];
            fprintf(fp,"%8.3f %8.3f %8.8lf %8.8lf %20.10e  %8.3f %8.3f\n",v_gpu.xx()[i],v_gpu.yy()[j], v_cpu.ff()[ji], v_gpu.ff()[ji], diff, f_ave, f_max ); 
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

//void Output::
//parameter(){
//
//    #pragma omp single  
//    //printf("nx = %d, ny= %d, iter = %d, \n dx = %lf, nthread = %d, nthread_max = %d", defines::nx, defines::ny, defines::iter, defines::dx, defines::thread_num, omp_get_max_threads());
//    printf("nx = %d, ny= %d, iter = %d, \n dx = %lf \n, nthread = %d \n", defines::nx, defines::ny, defines::iter, defines::dx, defines::thread_num);
//    //printf("nx = %d, ny= %d, iter = %d, \n dx = %lf, nthread = %d", defines::nx, defines::ny, defines::iter, defines::dx, defines::thread_num);
//}

void Output::
print_elapsetime(const double st_time,const double ed_time, const double ave_time){
    real elapse_time, bw;
    real cal_ratio, gf; 
    elapse_time = ed_time - st_time;
    bw = 1e-09* defines::data_num * sizeof(real) * defines::ncell / ave_time;
    gf = 1e-09* defines::flop_num * defines::ncell / ave_time;    
    cal_ratio = gf/ defines::loof_line;

    char filename[1024];
    sprintf(filename, "data/log_GPU%04d_%04d.dat", defines::blocksizex, defines::blocksizey ) ;
    FILE* fp = fopen( filename ,"w");

    fprintf(fp, "nx = %d, ny= %d, iter = %d, \n dx = %lf \n, nthread = %d, blocksizex = %d, blocksizey = %d  \n ",
            defines::nx, defines::ny, defines::iter, defines::dx, defines::thread_num, defines::blocksizex, defines::blocksizey);
    fprintf(fp, "elapse time = %lf, elapse time loop avetage= %lf,\n", elapse_time, ave_time);
    fprintf(fp, "OMP_thread_num = %d, peak Flops = %lf, \n", defines::thread_num, defines::loof_line);
    fprintf(fp, "bandwidth(GB/s) = %lf, Flops = %lf, \n calcuration ratio = %lf  \n ", bw, gf, cal_ratio);
    //std:: cout << "elapse time = " << std::setw(8) << elapse_time
    //    << ", elapse time loop avetage  = " << std::setw(8) << ave_time << std::endl
    //    << "thread_num =" << defines::thread_num << ",  peak Flops = "  <<defines::loof_line  << std::endl
    //    << "bandwidth(GB/s) = "<< bw <<", Flops(Gflops) = "<< gf << std::endl
    //    << "calcuration ratio = "<< cal_ratio << std::endl; 
}


void Output::
print_elapsetime_loop(const double st_time,const double ed_time){
    double elapse_time;
    elapse_time = ed_time - st_time; 

    printf ("    elapse time loop = %lf \n ",elapse_time);
    std::cout << "elapse time loop = " << std::setw(8) << elapse_time << std::endl
        << "number_thread = " << std::setw(8) <<defines::thread_num
        << std::endl; 
}

real Output::
analytic( const ValuesDiffusion& v,int t){
    real f_ana = 0;
    real err_elm =0;
    for(int j=0; j<defines::ny; j++) {
        for(int i=0; i<defines::nx; i++) {
            const int ji = Index::index_xy(i,j);
            f_ana = defines::fmax * cos(v.xx()[i]/defines::lx*2.0*M_PI + v.yy()[j]/defines::lx*2.0*M_PI) * exp(- defines::c_dif* ((2.0 * M_PI/defines::lx)* (2.0 * M_PI/defines::lx)+ (2.0 * M_PI/defines::lx)* (2.0 * M_PI/defines::lx) ) * double(t)* defines::dt *defines::iout  );
            err_elm = err_elm + fabs(f_ana - v.ff()[ji]);
        }
    }
    return err_elm;
}



void Output::
print_sum( const ValuesDiffusion& v_gpu, const ValuesDiffusion& v_cpu, const int t ) {
    real sum = 0, mmm = v_gpu.ff()[0];
    for(int i=0; i<defines::ncell; i++) {
        //sum += fabs(v_gpu.ff()[i] );
        sum += fabs(v_gpu.ff()[i] - v_cpu.ff()[i]);
        mmm = std::fmax(mmm, v_gpu.ff()[i]);
    }
    real err_elm, elm, err_num ;
    err_elm = analytic(v_cpu,t);
    elm = err_elm/sum;
    err_num = err_elm/defines::ncell;
    real Tout = t * defines::dt * defines::iout;

    char filename[1024];
    sprintf(filename, "data/err_nx%08d.dat", defines::nx ) ;
    FILE* fp = fopen( filename ,"w");

    fprintf(fp, "t = %lf, dt= %lf, fmax = %lf, lx = %lf, mmm = %lf, sum = %lf, err_elm = %lf, err_num = %lf, elm = %lf \n ",
            Tout, defines::dt, defines::fmax, defines::lx, mmm, sum, err_elm, err_num, elm);
    //fprintf(fp, "elapse time = %lf, elapse time loop avetage= %lf,\n", elapse_time, ave_time);
    //fprintf(fp, "OMP_thread_num = %d, peak Flops = %lf, \n", defines::thread_num, defines::loof_line);
    //fprintf(fp, "bandwidth(GB/s) = %lf, Flops = %lf, \n calcuration ratio = %lf  \n ", bw, gf, cal_ratio);


    std::cout << "t= " << std::setw(8) << Tout
        << ":    " << std::setw(8) << t
        << ":    " << std::setw(8) << defines::dt
        << ":    " << std::setw(8) << defines::iout
        << ":    " << std::setw(8) << defines::fmax
        << ":    " << std::setw(8) << defines::lx
        << ":    " << std::setw(8) << mmm
        << ":    " << std::setw(8) << sum
        << ":    " << std::setw(8) << err_elm
        << ":    " << std::setw(8) << err_num
        << ",    " << std::setw(8) << elm << std::endl;
}



