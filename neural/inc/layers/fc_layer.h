#ifndef __FC_LAYER_H__
#define __FC_LAYER_H__

#include "layer.h"

namespace Neural
{
  class Fc_Layer : public Layer
  {
    protected:
      Eigen::MatrixXd m_weights;
      Eigen::MatrixXd m_bias;

    public:
      Fc_Layer(int input_size, int output_size);

      virtual Eigen::MatrixXd FeedForward(Eigen::MatrixXd input_data);
      virtual Eigen::MatrixXd BackPropagation(Eigen::MatrixXd output_error, float learning_rate);

  };
}

#endif