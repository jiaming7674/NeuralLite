#include "core.h"

using namespace std;
using namespace Neural;
using namespace Eigen;


Eigen::MatrixXd Core::RandomMatrix(int rows, int cols, float min, float max)
{
  MatrixXd m = MatrixXd::Random(rows, cols);

  return m;
}
