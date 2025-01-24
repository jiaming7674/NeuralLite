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


void Verify(vector<MatrixXd> &results, vector<int> answers)
{
  int index = 0;
  int correct_cnt = 0;

  for (auto res : results)
  {
    int pred_val = -1;
    for (int i = 0; i < res.cols(); i++) {
      if (res(0, i) > 0.8) pred_val = i;
    }

    int ans = answers[index++];
    cout << "predict val :  " << pred_val << "\tanswer : " << ans << endl;;
    if (pred_val == ans) correct_cnt++;
  }

  double accuracy = (double)correct_cnt / (double)answers.size() * 100;

  cout << "Correct : " << correct_cnt <<
          "\tError : " << answers.size() - correct_cnt <<
          "\tAccuracy : " << accuracy << " %" << endl;
}



int main(int argc, char *argv[])
{
  cout << "Convolutional Neural Networks (CNNs)" << endl;

  string train_image = "./dataset/MNIST/train-images-idx3-ubyte";
  string train_label = "./dataset/MNIST/train-labels-idx1-ubyte";

  string test_image = "./dataset/MNIST/t10k-images-idx3-ubyte";
  string test_label = "./dataset/MNIST/t10k-labels-idx1-ubyte";    

  // string train_image = "D:/temp/dataset/MNIST/train-images-idx3-ubyte";
  // string train_label  = "D:/temp/dataset/MNIST/train-labels-idx1-ubyte";

  // string test_image = "D:/temp/dataset/MNIST/t10k-images-idx3-ubyte";
  // string test_label  = "D:/temp/dataset/MNIST/t10k-labels-idx1-ubyte";      

  mnist train(train_image, train_label, 10000);
  mnist test(test_image, test_label, 1000);

  tuple<int, int, int> dimensions = make_tuple(28, 28, 1);
  tuple<int, int, int> filter = make_tuple(3, 3, 1);

  net = new Network();

  net->Use(new Mse());

  net->Add(new Conv_Layer(dimensions, filter, 1, 1));
  net->Add(new Fc_Layer(784, 100, ActivationType::TANH));
  net->Add(new Fc_Layer(100, 10, ActivationType::SOFTMAX));

  net->UseOptimizer(new Adam(0.001));

  cout << "Start Traning ..." << endl;

  net->Fit(train.data.images, train.data.labels, 30, 0.1, 1, 2);

  auto y_predit = net->Predict(test.data.images);

  cout << "Predict -----" << endl;

  Verify(y_predit, test.m_labels);

  return 0;
}