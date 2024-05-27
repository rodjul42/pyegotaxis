#pragma ones


void fieldv(double A,double k,int irows, int icols, double* input, int len, double* result1d, double bound);

void field_ij(int nrows, int ncols, double* result,double A,double k ,double center,double space_discretization_step);
void grad_field_ij(int mnrows, int mncols, int mndim, double* resultm,double A,double k ,double center,double space_discretization_step);

void grad_laplace_field_ij(int mnrows, int mncols, int mndim, double* resultm,double A,double k ,double center,double space_discretization_step);

void hess_field_ij(int mnrows, int mncols, int mndim, double* resultm,int nrows, int ncols, double* result,double A,double k ,double center,double space_discretization_step);

