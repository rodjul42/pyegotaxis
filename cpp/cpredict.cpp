#include <stdio.h>
#include <math.h>
#include "cpredict.h"
#include <iostream>
#include <algorithm>
#include <string>
#include <iostream>
#include "cnpy.h"




int calc_Drot::load_kernel_val(std::string fname){

    cnpy::npz_t my_npz = cnpy::npz_load(fname);
    int i = 0;
    for (i=0; i<my_npz.size(); i++){
        cnpy::NpyArray arr_mv1 = my_npz[std::to_string(i)];
        if (arr_mv1.shape.size() != 1) throw std::invalid_argument("input array is not a vector");

        double* loaded_data = arr_mv1.data<double>();
        std::vector<float> kernel_tmp;
        for (int j=0;j<arr_mv1.shape[0];j++){
            kernel_tmp.push_back(loaded_data[j]);
        }
        kernel_val.push_back(kernel_tmp);
    }
    return i;
}
int calc_Drot::load_kernel_idx(std::string fname){

    cnpy::npz_t my_npz = cnpy::npz_load(fname);
    int i = 0;
    for (i=0; i<my_npz.size(); i++){
        cnpy::NpyArray arr_mv1 = my_npz[std::to_string(i)];
        if (arr_mv1.shape.size() != 1) throw std::invalid_argument("input array is not a vector");

        long* loaded_data = arr_mv1.data<long>();
        std::vector<int> kernel_tmp;
        for (int j=0;j<arr_mv1.shape[0];j++){
            kernel_tmp.push_back(loaded_data[j]);
        }
        kernel_idx.push_back(kernel_tmp);
    }
    return i;
}

int calc_Drot::predict(int nrows, int ncols, double* likelihood, int nrows_r, int ncols_r, double* res ){
    for (unsigned int i = 0; i<nrows*ncols; i++){
        double li = likelihood[i];
        //if (abs(li)>minL){
            unsigned int size = kernel_idx[i].size();
            for (unsigned int j = 0;j<size;j++) {
                res[kernel_idx[i][j]] += kernel_val[i][j] * li;
            }
        //}
    }
    return 0;
}



inline void calc_adv(int ind_y, int ind_y_p1, int ind_y_m1, int ind_xc, int ind_x_p1 , int ind_x_m1,
             double* likelihood, double* res, double dx,double dy,double space_discretization_step, double dt){
        double Nl[] = {likelihood[ind_y + ind_x_p1], likelihood[ind_y + ind_x_m1],
                       likelihood[ind_xc + ind_y_p1], likelihood[ind_xc + ind_y_m1],
                       likelihood[ind_x_m1 + ind_y_m1], likelihood[ind_x_p1 + ind_y_p1],likelihood[ind_xc+ind_y] };
        double grad_x = ( likelihood[ind_y + ind_x_p1] - likelihood[ind_y + ind_x_m1] ) / (2*space_discretization_step);
        double grad_y = ( likelihood[ind_xc + ind_y_p1] - likelihood[ind_xc + ind_y_m1] ) / (2*space_discretization_step);
        double H_xx =  ( likelihood[ind_y + ind_x_p1] + likelihood[ind_y + ind_x_m1] - 2*likelihood[ind_xc+ind_y]) / (space_discretization_step*space_discretization_step);
        double H_yy =  ( likelihood[ind_xc + ind_y_p1] + likelihood[ind_xc + ind_y_m1] - 2*likelihood[ind_xc+ind_y]) / (space_discretization_step*space_discretization_step);
        double H_xy = ( likelihood[ind_x_m1 + ind_y_m1] + likelihood[ind_x_p1 + ind_y_p1] + 2*likelihood[ind_xc+ind_y] - 
                        likelihood[ind_x_m1 + ind_y] - likelihood[ind_x_p1 + ind_y] -likelihood[ind_xc + ind_y_m1] -likelihood[ind_xc + ind_y_p1]) 
                        / (2*space_discretization_step*space_discretization_step);
        double tmp = likelihood[ind_xc+ind_y] + dx*grad_x + dy*grad_y + 0.5*H_xx*dx*dx + 0.5*H_yy*dy*dy + H_xy*dx*dy;
        double min = 10000;
        double max =-10000;
        for (auto const el : Nl){
            min = std::min(min,el);
            max = std::max(max,el);
        }
        //auto [min, max] = std::ranges::minmax(Nl);
        if (tmp<min) tmp = min;
        else if (tmp>max) tmp = max;
        res[ind_xc+ind_y] = tmp ;
        return;
}

int predict_adv(int nrows, int ncols, double* likelihood,int nrows_r, int ncols_r, double* res,
            double dx,double dy,double space_discretization_step, double dt  )
{
    for (int ind_x=1;ind_x<nrows-1;ind_x++){
        int ind_x_m1=ncols*(ind_x-1);
        int ind_x_p1=ncols*(ind_x+1);
        int ind_xc = ind_x*ncols;
        for (int ind_y=1;ind_y<ncols-1;ind_y++){
            int ind_y_m1=ind_y-1;
            int ind_y_p1=ind_y+1;
            calc_adv( ind_y, ind_y_p1, ind_y_m1, ind_xc, ind_x_p1, ind_x_m1,
                     likelihood, res,dx, dy, space_discretization_step, dt);
        }
    }

    return 0;
}





inline void calc_all(int ind_y, int ind_y_p1, int ind_y_m1, int ind_x, int ind_xc, int ind_x_p1 , int ind_x_m1,
             double* likelihood, double* res, double space_discretization_step, double dx, double dy , double D,double Drot, double cx,double cy, double dt){
        double Nl[] = {likelihood[ind_y + ind_x_p1], likelihood[ind_y + ind_x_m1],
                       likelihood[ind_xc + ind_y_p1], likelihood[ind_xc + ind_y_m1],
                       likelihood[ind_x_m1 + ind_y_m1], likelihood[ind_x_p1 + ind_y_p1],likelihood[ind_xc+ind_y] };
        double grad_x = ( likelihood[ind_y + ind_x_p1] - likelihood[ind_y + ind_x_m1] ) / (2*space_discretization_step);
        double grad_y = ( likelihood[ind_xc + ind_y_p1] - likelihood[ind_xc + ind_y_m1] ) / (2*space_discretization_step);
        double H_xx =  ( likelihood[ind_y + ind_x_p1] + likelihood[ind_y + ind_x_m1] - 2*likelihood[ind_xc+ind_y]) / (space_discretization_step*space_discretization_step);
        double H_yy =  ( likelihood[ind_xc + ind_y_p1] + likelihood[ind_xc + ind_y_m1] - 2*likelihood[ind_xc+ind_y]) / (space_discretization_step*space_discretization_step);
        double H_xy = ( likelihood[ind_x_m1 + ind_y_m1] + likelihood[ind_x_p1 + ind_y_p1] + 2*likelihood[ind_xc+ind_y] - 
                        likelihood[ind_x_m1 + ind_y] - likelihood[ind_x_p1 + ind_y] -likelihood[ind_xc + ind_y_m1] -likelihood[ind_xc + ind_y_p1]) 
                        / (2*space_discretization_step*space_discretization_step);
        double x = ind_x*space_discretization_step - cx;
        double y = ind_y*space_discretization_step - cy;
        double diff   = dt*(D*(H_xx+H_yy) +             Drot*( -x*grad_x - y*grad_y - 2*x*y*H_xy + x*x*H_yy + y*y*H_xx));
        double adv = dx*grad_x + dy*grad_y + 0.5*H_xx*dx*dx + 0.5*H_yy*dy*dy + H_xy*dx*dy ;
        double tmp  = likelihood[ind_xc+ind_y] + diff + adv;        

        //auto [min, max] = std::ranges::minmax(Nl);
        double min = 10000;
        double max =-10000;
        for (auto const el : Nl){
            min = std::min(min,el);
            max = std::max(max,el);
        }
        if (tmp<min) tmp = min;
        else if (tmp>max) tmp = max;  
        res[ind_xc+ind_y] = tmp;

        return;
}

int predict_all(int nrows, int ncols, double* likelihood,int nrows_r, int ncols_r, double* res,
            double space_discretization_step, double dx, double dy , double D, double Drot, double cx,double cy, double dt  )
{
    for (int ind_x=1;ind_x<nrows-1;ind_x++){
        int ind_x_m1=ncols*(ind_x-1);
        int ind_x_p1=ncols*(ind_x+1);
        int ind_xc = ind_x*ncols;
        for (int ind_y=1;ind_y<ncols-1;ind_y++){
            int ind_y_m1=ind_y-1;
            int ind_y_p1=ind_y+1;
            calc_all( ind_y, ind_y_p1, ind_y_m1, ind_x, ind_xc, ind_x_p1, ind_x_m1,
                     likelihood, res, space_discretization_step, dx, dy , D, Drot, cx, cy, dt);
        }
    }

    return 0;
}

