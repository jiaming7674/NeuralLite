#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include <string>
#include <Eigen/Dense>

namespace Neural
{
  class Activation {
    public:
      Activation() {};
      virtual Eigen::MatrixXd Compute(Eigen::MatrixXd x) = 0;
      virtual Eigen::MatrixXd ComputePrime(Eigen::MatrixXd x) = 0;
      std::string getType() {
        return this->m_type;
      }
      protected:
        std::string m_type;
  };

  class Than : public Activation {
    public:
      Than() {
        m_type = "Than";
      };

      virtual Eigen::MatrixXd Compute(Eigen::MatrixXd x) {
        return x.array().tanh();
      }

      virtual Eigen::MatrixXd ComputePrime(Eigen::MatrixXd x) {
        return 1-x.array().tanh().pow(2);
      }
  };
}

#endif