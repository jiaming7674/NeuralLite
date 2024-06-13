#ifndef __CORE_H__
#define __CORE_H__

#include <Eigen/Dense>


namespace Neural
{
  class Core {
    public:
      Core() {};
      static Eigen::MatrixXd RandomMatrix(int rows, int cols, float min, float max);
  };
}

#endif