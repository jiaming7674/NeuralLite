#ifndef __FC_LAYER_H__
#define __FC_LAYER_H__

#include "layer.h"

namespace Neural
{
  class Fc_Layer : public Layer
  {
    protected:
      Activation *p_activation;

    public:
      Fc_Layer(int input_size, int output_size, ActivationType activationType);

      virtual Eigen::MatrixXd FeedForward(const Eigen::MatrixXd& input_data);
      virtual Eigen::MatrixXd BackPropagation(const Eigen::MatrixXd& output_error, float learning_rate);

      virtual void SaveLayer(std::ofstream &outfile);
      static Fc_Layer* LoadLayer(std::ifstream &infile);
      void SetWeights(Eigen::MatrixXd &weights);
      void SetBias(Eigen::MatrixXd &bias);
  };
}

#endif