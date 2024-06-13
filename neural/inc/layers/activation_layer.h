#ifndef __ACTIVATION_LAYER_H__
#define __ACTIVATION_LAYER_H__

#include "layer.h"
#include "activation.h"

namespace Neural
{
  class Activation_Layer : public Layer
  {
    private:
      Activation *p_activation;

    public:
      Activation_Layer();
      Activation_Layer(Activation *a);
      ~Activation_Layer();

      virtual Eigen::MatrixXd FeedForward(Eigen::MatrixXd input_data);
      virtual Eigen::MatrixXd BackPropagation(Eigen::MatrixXd output_error, float learning_rate);
  };
}

#endif