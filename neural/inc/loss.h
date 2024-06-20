#ifndef __LOSS_H__
#define __LOSS_H__

#include <Eigen/Dense>


namespace Neural
{
  class Loss {
    public:
      Loss() {};
      virtual double Compute(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred) = 0;
      virtual Eigen::MatrixXd ComputeDerivative(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred) = 0;
  };

  class Mse : public Loss {
    public:
      Mse() {};
      virtual double Compute(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred) {
        Eigen::MatrixXd diff = y_true-y_pred;
        return diff.array().pow(2).mean();
      }
      virtual Eigen::MatrixXd ComputeDerivative(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred) {
        Eigen::MatrixXd diff = y_pred-y_true;
        return (2*diff)/y_true.size();
      }
  };
}

#endif