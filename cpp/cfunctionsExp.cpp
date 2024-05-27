#include <cmath>
#include "cfunctionsExp.h"

#include <iostream>


double fieldi(double A,double k, double& x, double& y, double& bound){
/* input:
 ... lambda_ : [m/s] field strength
# ... R       : [m] distance from target    
*/
    auto R = sqrt(x*x+y*y);
    if (R<bound/2) R = bound/2;
    return A * std::exp(- R * k);
}


void grad_field(double A,double k, double& x, double& y, double& res_x, double& res_y, double& bound){
    auto R = sqrt(x*x+y*y);
    if (R<bound/2) R = bound/2;
    res_x =   A * std::exp(- R * k) * (-x/R*k);
    res_y =   A * std::exp(- R * k) * (-y/R*k);
}

void hess_field(double A,double k, double& x, double& y, double& res_h00, double& res_h10, double& res_h01, double& res_h11, double& bound){
    auto R = sqrt(x*x+y*y);
    if (R<bound/2) R = bound/2;
    auto R3 = std::pow(R,3);
    auto tmp = A * std::exp(- R * k) * k / R3;
    res_h00 = tmp * ( -y*y + k*x*x*R );
    res_h11 = tmp * ( -x*x + k*y*y*R );
    double cross = tmp * x*y * (1+k*R);  
    res_h10 = cross;
    res_h01 = cross;
}

void grad_laplace_field(double A,double k, double& x, double& y, double& res_x, double& res_y, double& bound){
    auto R = sqrt(x*x+y*y);
    if (R<bound/2) R = bound/2;
    auto tmp = A * std::exp(- R * k) * (k/std::pow(R,3) + k*k/std::pow(R,2) - k*k*k/R);
    res_x = x*tmp;
    res_y = y*tmp;
}


void fieldv(double A,double k,int irows, int icols, double* input, int len, double* result1d, double bound){
/* input:
 ... lambda_ : [m/s] field strength
# ... R       : [m] distance from target    
*/
    for(int i=0; i<irows; i++){
    double x = input[i*icols];    
    double y = input[i*icols+1];    
    result1d[i] = fieldi(A, k, x, y, bound);
    }
}


void field_ij(int nrows, int ncols, double* result,double A,double k ,double center,double space_discretization_step)
{
    #pragma omp parallel for
    for (int i=0;i<nrows;i++){
        int irow = i*ncols;
        for (int j=0;j<ncols;j++){
            auto x = i*space_discretization_step-center;
            auto y = j*space_discretization_step-center;
            result[irow+j] = fieldi(A, k, x ,y, space_discretization_step);
        }
    }
}

void grad_field_ij(int mnrows, int mncols, int mndim, double* resultm,double A,double k ,double center,double space_discretization_step)
{
    #pragma omp parallel for
    for (int i=0;i<mnrows;i++){
        for (int j=0;j<mncols;j++){
            int idx = i*mncols*mndim + j*mndim;
            auto x = i*space_discretization_step-center;
            auto y = j*space_discretization_step-center;
            double res_x,res_y;
            grad_field( A, k, x, y, res_x, res_y, space_discretization_step);
            resultm[idx] = res_x;
            resultm[idx+1] = res_y;
        }
    }
}

void grad_laplace_field_ij(int mnrows, int mncols, int mndim, double* resultm,double A,double k ,double center,double space_discretization_step)
{
    #pragma omp parallel for
    for (int i=0;i<mnrows;i++){
        for (int j=0;j<mncols;j++){
            int idx = i*mncols*mndim + j*mndim;
            auto x = i*space_discretization_step-center;
            auto y = j*space_discretization_step-center;
            double res_x,res_y;
            grad_laplace_field( A, k, x, y, res_x, res_y, space_discretization_step);
            resultm[idx] = res_x;
            resultm[idx+1] = res_y;
        }
    }
}

void hess_field_ij(int mnrows, int mncols, int mndim, double* resultm,int nrows, int ncols, double* result,double A,double k ,double center,double space_discretization_step)
{
    #pragma omp parallel for
    for (int i=0;i<nrows;i++){
        for (int j=0;j<ncols;j++){
            int idxm = i*mncols*mndim + j*mndim;
            int idx = i*ncols + j ;
            auto x = i*space_discretization_step-center;
            auto y = j*space_discretization_step-center;
            double res_h00,res_h10,res_h01,res_h11;
            hess_field( A, k , x, y, res_h00, res_h10, res_h01, res_h11, space_discretization_step);
            resultm[idxm] = res_h00;
            resultm[idxm+1] = res_h01;
            resultm[idxm+2] = res_h10;
            resultm[idxm+3] = res_h11;
            result[idx] = res_h00 + res_h11;
        }
    }
}