#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "layer.h"


namespace Neural
{
  class Conv_Layer : public Layer
  {
    private:
      int m_depth;
      int m_height;
      int m_width;
      int m_filter_size;
      int m_nb_filters;
      int m_stride;
      int m_padding;

    public:
      Conv_Layer(std::tuple<int, int, int> dimensions,
            std::tuple<int, int, int> filter,
            int stride,
            int padding);

      Eigen::MatrixXd FeedForward(const Eigen::MatrixXd& input) override;
      Eigen::MatrixXd BackPropagation(const Eigen::MatrixXd& output, float learning_rate) override;
      void SaveLayer(std::ofstream &outfile) override;
      void SetWeights(Eigen::MatrixXd &weights) override;
      void SetBias(Eigen::MatrixXd &bias) override;
  };
}


#endif