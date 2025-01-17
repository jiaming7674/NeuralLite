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

      Eigen::MatrixXd FeedForward(const Eigen::MatrixXd& input_data) override;
      Eigen::MatrixXd BackPropagation(const Eigen::MatrixXd& output_error, float learning_rate) override;

      void SaveLayer(std::ofstream &outfile) override;
      static Fc_Layer* LoadLayer(std::ifstream &infile);
      void SetWeights(Eigen::MatrixXd &weights);
      void SetBias(Eigen::MatrixXd &bias);
  };
}

#endif