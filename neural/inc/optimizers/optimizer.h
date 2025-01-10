#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace Neural
{
  class Optimizer
  {
  public:
    virtual void UpdateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &grad_weights) = 0;
    virtual void UpdateBias(Eigen::MatrixXd &bias, const Eigen::MatrixXd &grad_bias) = 0;
    virtual std::unique_ptr<Optimizer> Clone() const = 0;  // 克隆接口
    virtual ~Optimizer() {}
  };

  class Adam : public Optimizer
  {
  private:
    double m_learning_rate;
    double m_beta1;
    double m_beta2;
    double m_epsilon;
    int m_t;
    Eigen::MatrixXd m_m_weights, m_v_weights;
    Eigen::MatrixXd m_m_bias, m_v_bias;

  public:
    Adam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : m_learning_rate(learning_rate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_t(0) {}

    void UpdateWeights(Eigen::MatrixXd &weights, const Eigen::MatrixXd &grad_weights) override
    {
      // 初始化動量和方差
      if (m_m_weights.size() == 0)
      {
        m_m_weights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
        m_v_weights = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
      }

      m_t++;
      // Updating momentum and variance 
      m_m_weights = m_beta1 * m_m_weights + (1 - m_beta1) * grad_weights;
      m_v_weights = m_beta2 * m_v_weights + (1 - m_beta2) * grad_weights.array().square().matrix();

      // Adjusted momentum and variance
      Eigen::MatrixXd m_hat = m_m_weights / (1 - std::pow(m_beta1, m_t));
      Eigen::MatrixXd v_hat = m_v_weights / (1 - std::pow(m_beta2, m_t));

      // Update weights
      weights -= m_learning_rate * (m_hat.array() / (v_hat.array().sqrt() + m_epsilon)).matrix();
    }

    void UpdateBias(Eigen::MatrixXd &bias, const Eigen::MatrixXd &grad_bias) override
    {
      // Initialize momentum and variance
      if (m_m_bias.size() == 0)
      {
        m_m_bias = Eigen::MatrixXd::Zero(bias.rows(), bias.cols());
        m_v_bias = Eigen::MatrixXd::Zero(bias.rows(), bias.cols());
      }

      m_t++;
      // Updating momentum and variance 
      m_m_bias = m_beta1 * m_m_bias + (1 - m_beta1) * grad_bias;
      m_v_bias = m_beta2 * m_v_bias + (1 - m_beta2) * grad_bias.array().square().matrix();

      // Adjusted momentum and variance
      Eigen::MatrixXd m_hat = m_m_bias / (1 - std::pow(m_beta1, m_t));
      Eigen::MatrixXd v_hat = m_v_bias / (1 - std::pow(m_beta2, m_t));

      // Update bias
      bias -= m_learning_rate * (m_hat.array() / (v_hat.array().sqrt() + m_epsilon)).matrix();
    }

    std::unique_ptr<Optimizer> Clone() const override {
      return std::make_unique<Adam>(m_learning_rate, m_beta1, m_beta2, m_epsilon);
    }

  };
};

#endif