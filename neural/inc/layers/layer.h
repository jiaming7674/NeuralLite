#ifndef __LAYER_H__
#define __LAYER_H__

#include <iostream>
#include <Eigen/Dense>

#include "activation.h"
#include "../optimizers/optimizer.h"

namespace Neural
{
  class Layer
  {
    protected:
      Eigen::MatrixXd m_input;
      Eigen::MatrixXd m_net_sum;
      Eigen::MatrixXd m_output;
      Eigen::MatrixXd m_weights;
      Eigen::MatrixXd m_bias;
      bool m_as_weight;
    public:
      std::unique_ptr<Optimizer> m_optimizer;

    public:
    Layer() {};

    public:
      virtual Eigen::MatrixXd FeedForward(const Eigen::MatrixXd& input) = 0;
      virtual Eigen::MatrixXd BackPropagation(const Eigen::MatrixXd& output_error, float learning_rate) = 0;
      virtual void SaveLayer(std::ofstream &outfile) = 0;
      virtual void SetWeights(Eigen::MatrixXd &weights) = 0;
      virtual void SetBias(Eigen::MatrixXd &bias) = 0;
  };
}

#endif