#ifndef __LAYER_H__
#define __LAYER_H__

#include <Eigen/Dense>

namespace Neural
{
  class Layer
  {
    protected:
      Eigen::MatrixXd m_input;
      Eigen::MatrixXd m_output;
      bool m_as_weight;

    public:
      Layer() {}

    public:
      virtual Eigen::MatrixXd FeedForward(Eigen::MatrixXd input) = 0;
      virtual Eigen::MatrixXd BackPropagation(Eigen::MatrixXd output_error, float learning_rate) = 0;
  };
}

#endif