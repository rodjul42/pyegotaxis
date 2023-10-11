#include <cmath>
#include "cfunctions.h"

#include <iostream>


double fieldi(double lambda, double& x, double& y, double& bound){
/* input:
 ... lambda_ : [m/s] field strength
# ... R       : [m] distance from target    
*/
    auto R = sqrt(x*x+y*y);
    if (R<bound/2) R = bound/2;
    return lambda/R;
}


void grad_field(double lambda, double& x, double& y, double& res_x, double& res_y, double& bound){
    auto R = sqrt(x*x+y*y);
    if (R<bound/2) R = bound/2;
    res_x =  x * -lambda/std::pow(R,3);
    res_y =  y * -lambda/std::pow(R,3);
}

void hess_field(double lambda, double& x, double& y, double& res_h00, double& res_h10, double& res_h01, double& res_h11, double& bound){
    auto R = sqrt(x*x+y*y);
    if (R<bound/2) R = bound/2;
    double R5 = std::pow(R,5);
    double R3 = std::pow(R,3);
    res_h00 = -lambda/(R3)+3*lambda*x*x/(R5);
    res_h11 = -lambda/(R3)+3*lambda*y*y/(R5);
    double cross = 3*lambda*x*y/(R5);
    res_h10 = cross;
    res_h01 = cross;
}

void grad_laplace_field(double lambda, double& x, double& y, double& res_x, double& res_y, double& bound){
    auto R = sqrt(x*x+y*y);
    if (R<bound/2) R = bound/2;
    res_x = x*-3*lambda/std::pow(R,5);
    res_y = y*-3*lambda/std::pow(R,5);
}


void fieldv(double lambda,int irows, int icols, double* input, int len, double* result1d, double bound){
/* input:
 ... lambda_ : [m/s] field strength
# ... R       : [m] distance from target    
*/
    for(int i=0; i<irows; i++){
    double x = input[i*icols];    
    double y = input[i*icols+1];    
    result1d[i] = fieldi(lambda, x, y, bound);
    }
}


void distances_ij(int nrows, int ncols, double* result,double center,double space_discretization_step)
{
    #pragma omp parallel for
    for (int i=0;i<nrows;i++){
        int irow = i*ncols;
        for (int j=0;j<ncols;j++){
            auto x = i*space_discretization_step-center;
            auto y = j*space_discretization_step-center;
            result[irow+j] = sqrt(x*x+y*y);
        }
    }
}

void field_ij(int nrows, int ncols, double* result,double lambda ,double center,double space_discretization_step)
{
    #pragma omp parallel for
    for (int i=0;i<nrows;i++){
        int irow = i*ncols;
        for (int j=0;j<ncols;j++){
            auto x = i*space_discretization_step-center;
            auto y = j*space_discretization_step-center;
            result[irow+j] = fieldi(lambda, x ,y, space_discretization_step);
        }
    }
}

void grad_field_ij(int mnrows, int mncols, int mndim, double* resultm,double lambda ,double center,double space_discretization_step)
{
    #pragma omp parallel for
    for (int i=0;i<mnrows;i++){
        for (int j=0;j<mncols;j++){
            int idx = i*mncols*mndim + j*mndim;
            auto x = i*space_discretization_step-center;
            auto y = j*space_discretization_step-center;
            double res_x,res_y;
            grad_field( lambda, x, y, res_x, res_y, space_discretization_step);
            resultm[idx] = res_x;
            resultm[idx+1] = res_y;
        }
    }
}

void grad_laplace_field_ij(int mnrows, int mncols, int mndim, double* resultm,double lambda ,double center,double space_discretization_step)
{
    #pragma omp parallel for
    for (int i=0;i<mnrows;i++){
        for (int j=0;j<mncols;j++){
            int idx = i*mncols*mndim + j*mndim;
            auto x = i*space_discretization_step-center;
            auto y = j*space_discretization_step-center;
            double res_x,res_y;
            grad_laplace_field( lambda, x, y, res_x, res_y, space_discretization_step);
            resultm[idx] = res_x;
            resultm[idx+1] = res_y;
        }
    }
}

void hess_field_ij(int mnrows, int mncols, int mndim, double* resultm,int nrows, int ncols, double* result,double lambda ,double center,double space_discretization_step)
{
    #pragma omp parallel for
    for (int i=0;i<nrows;i++){
        for (int j=0;j<ncols;j++){
            int idxm = i*mncols*mndim + j*mndim;
            int idx = i*ncols + j ;
            auto x = i*space_discretization_step-center;
            auto y = j*space_discretization_step-center;
            double res_h00,res_h10,res_h01,res_h11;
            hess_field( lambda, x, y, res_h00, res_h10, res_h01, res_h11, space_discretization_step);
            resultm[idxm] = res_h00;
            resultm[idxm+1] = res_h01;
            resultm[idxm+2] = res_h10;
            resultm[idxm+3] = res_h11;
            result[idx] = res_h00 + res_h11;
        }
    }
}