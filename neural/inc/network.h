#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <vector>
#include <string>
#include "layers/fc_layer.h"
#include "loss.h"

namespace Neural
{
  class Network
  {
    private:
      Loss *m_loss;
      std::vector<Layer*> m_layer;
      std::vector<double> m_error;

    public:
      Network();
      ~Network();

      void Add(Layer *layer);
      void Use(Loss *l);
      void UseOptimizer(Optimizer* optimizer);
      void Fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train, int epochs, double learning_rate, int batch_size, int verbose = 1);
      void Evaluate(Eigen::MatrixXd y_tests, Eigen::MatrixXd y_true);

      std::vector<Eigen::MatrixXd> Predict(Eigen::MatrixXd input_data);
      void SaveModel(std::string name);
      static Network* LoadModel(std::string name);
  };
}


#endif