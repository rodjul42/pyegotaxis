%module cfunctionsExp
%{
  #define SWIG_FILE_WITH_INIT
  #include "cfunctionsExp.h"
%}

%include "numpy.i"
 // Get the STL typemaps
%include "stl.i"

// Handle standard exceptions
%include "exception.i"
%exception
{
  try
  {
    $action
  }
  catch (const std::invalid_argument& e)
  {
    SWIG_exception(SWIG_ValueError, e.what());
  }
  catch (const std::out_of_range& e)
  {
    SWIG_exception(SWIG_IndexError, e.what());
  }
}
%init %{
    import_array();
%}


%apply (int DIM1, double* INPLACE_ARRAY1 )
      {(int len, double* result1d)};


%apply (int DIM1 , int DIM2 , double* INPLACE_ARRAY2)
      {(int nrows, int ncols, double* result),
       (int irows, int icols, double* input)};

%apply (int DIM1 , int DIM2 , int DIM3 , double* INPLACE_ARRAY3)
      {(int mnrows, int mncols, int mndim, double* resultm)};

%include "cfunctionsExp.h"

