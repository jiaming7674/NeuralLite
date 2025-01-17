#ifndef __CORE_H__
#define __CORE_H__

#include <Eigen/Dense>

enum PaddingType  {
  SAME,
  VALID
};


namespace Neural
{
  class Core {
    public:
      Core() {};
    public:
      static Eigen::MatrixXd RandomMatrix(int rows, int cols, float min, float max);
      static Eigen::MatrixXd Correlate2D(const Eigen::MatrixXd& input, 
                                  const Eigen::MatrixXd& filter,
                                  int stride,
                                  PaddingType padding);
      static Eigen::MatrixXd Padding(const Eigen::MatrixXd& m, int p);
      
  };
}

#endif