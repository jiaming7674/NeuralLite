#include "core.h"

using namespace std;
using namespace Neural;
using namespace Eigen;


MatrixXd Core::RandomMatrix(int rows, int cols, float min, float max)
{
  MatrixXd m = MatrixXd::Random(rows, cols);
  m = (m + MatrixXd::Ones(rows, cols)) / 2;
  m = m * (max - min) + MatrixXd::Constant(rows, cols, min);

  return m;
}


MatrixXd Core::Correlate2D(const MatrixXd& input,
                                  const MatrixXd& filter,
                                  int stride,
                                  PaddingType padding)
{
  MatrixXd pad;
  int sizeFilter = filter.rows();

  if (padding == SAME)
  {
    MatrixXd res(input.rows(), input.cols());
    pad = Core::Padding(input, (filter.rows()-1)/2);
    int _i = 0;
    for (int i = 0; i <input.rows(); i+=stride)
    {
      int _j = 0;
      for (int j = 0; j < input.cols(); j+=stride)
      {
        MatrixXd temp = pad.block(i, j, sizeFilter, sizeFilter);
        double acc = temp.cwiseProduct(filter).sum();
        res(_i, _j) = acc;
        _j++;
      }
      _i++;
    }
    return res;
  }
  else {
    MatrixXd res(input.rows()-filter.rows()+1, input.cols()-filter.cols()+1);
    pad = input;
    int _i = 0;
    for (int i = 0; i < (input.rows() - sizeFilter); i+=stride)
    {
      int _j = 0;
      for (int j = 0; j < (input.cols() - sizeFilter); j+=stride)
      {
        MatrixXd temp = pad.block(i-1, j-1, sizeFilter, sizeFilter);
        double acc = temp.cwiseProduct(filter).sum();
        res(_i, _j) = acc;
        _j++;
      }
      _i++;
    }
    return res;
  }
}


MatrixXd Core::Padding(const MatrixXd& m, int p)
{
  MatrixXd a = MatrixXd::Zero(p*2 + m.rows(), p*2 + m.cols());
  a.block(p, p, m.rows(), m.cols()) = m;
  return a;
}