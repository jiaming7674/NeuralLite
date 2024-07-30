#include <iostream>
#include "network.h"
#include "layers/fc_layer.h"
#include "layers/activation_layer.h"

#include <Eigen/Eigen>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace Neural;


int main(int argc, char *argv[])
{
  MatrixXd x_data(4, 2);
  x_data <<
            0, 0,
            0, 1,
            1, 0,
            1, 1;

  MatrixXd x_train(4, 1);

  x_train << 0,
             1,
             1,
             0;

  MatrixXd x_test(4, 2);
  x_test <<
            1, 1,
            0, 1,
            0, 0,
            1, 0;
  
  Network *net = new Network();
  net->Use(new Mse());

  net->Add(new Fc_Layer(2, 5, ActivationType::TANH));
  net->Add(new Fc_Layer(5, 5, ActivationType::RELU));
  net->Add(new Fc_Layer(5, 1, ActivationType::TANH));

  net->Fit(x_data, x_train, 10000, 0.01, 1);
  net->Predict(x_test);

  net->SaveModel("test");

  cout << "Load Model ..." << endl;
  auto lnet = Network::LoadModel("test");

  lnet->Predict(x_test);

  return 0;
}
