%module cpredict
%{
  #define SWIG_FILE_WITH_INIT
  #include "cpredict.h"
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




%apply (int DIM1 , int DIM2 , double* INPLACE_ARRAY2)
      {(int nrows, int ncols, double* likelihood),(int nrows_r, int ncols_r, double* res)};

%include "cpredict.h"

