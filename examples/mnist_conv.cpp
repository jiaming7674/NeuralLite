#include <iostream>
#include "network.h"
#include "layers/fc_layer.h"
#include "layers/activation_layer.h"
#include "layers/conv_layer.h"
#include "mnist.h"


using namespace std;
using namespace Eigen;
using namespace Neural;

Network *net;

int main(int argc, char *argv[])
{
  cout << "Convolutional Neural Networks (CNNs)" << endl;

  string train_image = "./dataset/MNIST/train-images-idx3-ubyte";
  string train_label = "./dataset/MNIST/train-images-idx3-ubyte";

  string test_image = "./dataset/MNIST/t10k-images-idx3-ubyte";
  string test_label = "./dataset/MNIST/t10k-labels-idx1-ubyte";    

  // string train_file = "D:/temp/dataset/MNIST/train-images-idx3-ubyte";
  // string test_file  = "D:/temp/dataset/MNIST/train-images-idx3-ubyte";  

  mnist train(train_image, train_label, 1000);
  mnist test(test_image, test_label, 1000);

  tuple<int, int, int> dimensions = make_tuple(28, 28, 1);
  tuple<int, int, int> filter = make_tuple(3, 3, 1);
  // TODO : Number of filter ??

  net = new Network();

  net->Use(new Mse());

  net->Add(new Conv_Layer(dimensions, filter, 1, 1));
  net->Add(new Fc_Layer(784, 100, ActivationType::TANH));
  net->Add(new Fc_Layer(100, 10, ActivationType::SOFTMAX));

  net->UseOptimizer(new Adam(0.001));

  cout << "Start Traning ..." << endl;

  net->Fit(train.data.images, train.data.labels, 35, 0.1, 1, 2);

  auto y_predit = net->Predict(test.data.images);

  cout << "Predict -----" << endl;
  int i = 0;
  for (auto y : y_predit) {
    cout << y << endl;
    if (++i >= 100) break;
  }

  return 0;
}