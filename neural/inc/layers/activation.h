#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <string>
#include <Eigen/Dense>
#include <cmath>

namespace Neural
{
  enum class ActivationType
  {
    NONE, SIGMOID, RELU, LEAKY_RELU, ELU, TANH, SOFTMAX
  };

  class Activation {
    public:
      Activation() {};
      virtual Eigen::MatrixXd Compute(const Eigen::MatrixXd& x) = 0;
      virtual Eigen::MatrixXd ComputeDerivative(const Eigen::MatrixXd& x) = 0;
      ActivationType getType() {
        return this->m_type;
      }
      protected:
        ActivationType m_type;
  };

  class Sigmoid : public Activation {
    public:
      Sigmoid() {
        m_type = ActivationType::SIGMOID;
      };

      virtual Eigen::MatrixXd Compute(const Eigen::MatrixXd& x) {
        return 1.0 / (1.0 + (-x.array()).exp());
      }

      virtual Eigen::MatrixXd ComputeDerivative(const Eigen::MatrixXd& x) {
        Eigen::MatrixXd s = Compute(x);
        return s.array() * (1 - s.array());
      }
  };


  class ReLU : public Activation {
    public:
      ReLU() {
        m_type = ActivationType::RELU;
      };

      virtual Eigen::MatrixXd Compute(const Eigen::MatrixXd& x) {
        return x.array().max(0);
      }

      virtual Eigen::MatrixXd ComputeDerivative(const Eigen::MatrixXd& x) {
        return (x.array() > 0).cast<double>();
      }
  };


  class LeakyReLU : public Activation {
    private:
      double alpha;    
    public:
      LeakyReLU(double alpha = 0.01) : alpha(alpha) {
        m_type = ActivationType::LEAKY_RELU;
      };
    
      virtual Eigen::MatrixXd Compute(const Eigen::MatrixXd& x) {
        return (x.array() < 0).select(alpha * x.array(), x.array());      
      }

      virtual Eigen::MatrixXd ComputeDerivative(const Eigen::MatrixXd& x) {
        return (x.array() < 0).select(Eigen::MatrixXd::Constant(x.rows(), x.cols(), alpha), Eigen::MatrixXd::Constant(x.rows(), x.cols(), 1.0));
      }
  };


  class ELU : public Activation {
    private:
      double alpha;
    public:
      ELU(double alpha = 1.0) : alpha(alpha) {
        m_type = ActivationType::ELU;
      };

      virtual Eigen::MatrixXd Compute(const Eigen::MatrixXd& x) {
        return (x.array() < 0).select(alpha * (x.array().exp() - 1), x.array());
      }

      virtual Eigen::MatrixXd ComputeDerivative(const Eigen::MatrixXd& x) {
        return (x.array() < 0).select(alpha * x.array().exp(), Eigen::MatrixXd::Constant(x.rows(), x.cols(), 1.0));
      }
  };


  class Tanh : public Activation {
    public:
      Tanh() {
        m_type = ActivationType::TANH;
      };

      virtual Eigen::MatrixXd Compute(const Eigen::MatrixXd& x) {
        return x.array().tanh();
      }

      virtual Eigen::MatrixXd ComputeDerivative(const Eigen::MatrixXd& x) {
        return 1-x.array().tanh().pow(2);
      }
  };


  class Softmax : public Activation {
    public:
      Softmax() {
        m_type = ActivationType::SOFTMAX;
      };

      virtual Eigen::MatrixXd Compute(const Eigen::MatrixXd& x) {
        Eigen::MatrixXd exp_x = x.array().exp();
        Eigen::VectorXd sum_exp = exp_x.rowwise().sum();
        return exp_x.array().colwise() / sum_exp.array();
      }

      virtual Eigen::MatrixXd ComputeDerivative(const Eigen::MatrixXd& x) {
        // Note: This is a simplified version. The actual Jacobian of Softmax is more complex.
        Eigen::MatrixXd s = Compute(x);
        return s.array() * (1 - s.array());
      }
  };
}

#endif