#pragma ones
#include <stdio.h>
#include <vector>
#include <string>

class calc_Drot{
    public:
    std::vector<std::vector<int>> kernel_idx;
    std::vector<std::vector<float>> kernel_val;
    double minL;
    calc_Drot(double minL):minL(minL){};
    ~calc_Drot(){};
    int load_kernel_val(std::string fname);
    int load_kernel_idx(std::string fname);
    int predict(int nrows, int ncols, double* likelihood, int nrows_r, int ncols_r, double* res );
};



int predict_adv(int nrows, int ncols, double* likelihood,int nrows_r, int ncols_r, double* res,
            double dx,double dy,double space_discretization_step, double dt  );

int predict_all(int nrows, int ncols, double* likelihood,int nrows_r, int ncols_r, double* res,
            double space_discretization_step, double dx, double dy , double D, double Drot, double cx,double cy, double dt  );

int predict_rot(int nrows, int ncols, double* likelihood,int nrows_r, int ncols_r, double* res,
            double space_discretization_step, double Drot, double cx,double cy, double dt  );




