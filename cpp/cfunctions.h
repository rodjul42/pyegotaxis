#pragma ones


void fieldv(double lambda,int irows, int icols, double* input, int len, double* result1d, double bound);

void distances_ij(int nrows, int ncols, double* result,double center,double space_discretization_step);

void field_ij(int nrows, int ncols, double* result,double lambda ,double center,double space_discretization_step);
void grad_field_ij(int mnrows, int mncols, int mndim, double* resultm,double lambda ,double center,double space_discretization_step);

void grad_laplace_field_ij(int mnrows, int mncols, int mndim, double* resultm,double lambda ,double center,double space_discretization_step);

void hess_field_ij(int mnrows, int mncols, int mndim, double* resultm,int nrows, int ncols, double* result,double lambda ,double center,double space_discretization_step);

