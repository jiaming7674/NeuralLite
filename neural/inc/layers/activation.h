#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <string>
#include <Eigen/Dense>

namespace Neural
{
  enum class ActivationType
  {
    NONE, SIGMOID, RELU, LEAKY_RELU, ELU, TANH, SOFTMAX
  };

  class Activation {
    public:
      Activation() {};
      virtual Eigen::MatrixXd Compute(Eigen::MatrixXd x) = 0;
      virtual Eigen::MatrixXd ComputeDerivative(Eigen::MatrixXd x) = 0;
      ActivationType getType() {
        return this->m_type;
      }
      protected:
        ActivationType m_type;
  };

  class Than : public Activation {
    public:
      Than() {
        m_type = ActivationType::TANH;
      };

      virtual Eigen::MatrixXd Compute(Eigen::MatrixXd x) {
        return x.array().tanh();
      }

      virtual Eigen::MatrixXd ComputeDerivative(Eigen::MatrixXd x) {
        return 1-x.array().tanh().pow(2);
      }
  };
}

#endif