#include <iostream>
#include <fstream>
#include <chrono>
#include "network.h"


using namespace std;
using namespace Neural;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMajMat;


/**
 * @brief Construct a new Network:: Network object
 * 
 */
Network::Network()
{
  this->m_loss = nullptr;
}


/**
 * @brief Destroy the Network:: Network object
 * 
 */
Network::~Network()
{
  for (int i = 0; i < m_layer.size(); i++) {
    delete(m_layer[i]);
  }

  delete m_loss;
}


/**
 * @brief Adding a Layer to network.
 * 
 * @param layer The pointer of the Layer Mother (class Layer)
 */
void Network::Add(Layer *layer)
{
  m_layer.push_back(layer);
}


/**
 * @brief Adding a loss function to network.
 * 
 * @param l The pointer of th loss function
 */
void Network::Use(Loss *l)
{
  this->m_loss = l;
}


/**
 * @brief Adding a optimizer to network
 */
void Network::UseOptimizer(Optimizer* optimizer)
{
  for (auto layer : this->m_layer) {
    layer->m_optimizer = optimizer->Clone();
  }
}


/**
 * @brief Train the network on a set of data and a set of results, this is for set the good weights and bias.
 * 
 * @param x_train Matrix input data
 * @param y_train Matrix result data
 * @param epochs Number of iteration
 * @param learning_rate The step size at each iteration
 * @param batch_size 
 */
void Network::Fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train, int epochs, double learning_rate, int batch_size, int verbose)
{
    int samples = x_train.rows();
    int cols = x_train.cols();

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < epochs; i++) {
        double err = 0.0;
        auto t_start = chrono::high_resolution_clock::now();

        // Shuffle training data
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(samples);
        perm.setIdentity();
        std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
        x_train = perm * x_train;
        y_train = perm * y_train;

        // Mini-batch training
        for (int j = 0; j < samples; j += batch_size) {
            int batch_end = std::min(j + batch_size, samples);
            int current_batch_size = batch_end - j;

            Eigen::MatrixXd x_batch = x_train.block(j, 0, current_batch_size, cols);
            Eigen::MatrixXd y_batch = y_train.block(j, 0, current_batch_size, y_train.cols());

            Eigen::MatrixXd output = x_batch;

            // Forward pass
            for (int l = 0; l < m_layer.size(); l++) {
                output = m_layer[l]->FeedForward(output);
            }

            // Compute loss
            err += this->m_loss->Compute(y_batch, output);

            // Backward pass
            Eigen::MatrixXd error = this->m_loss->ComputeDerivative(y_batch, output);
            for (int k = m_layer.size() - 1; k >= 0; k--) {
                error = m_layer[k]->BackPropagation(error, learning_rate);
            }

            // Update progress (optional)
            if (verbose >= 2 && j % 100 == 0) {
                cout << "\rBatch " << j / batch_size + 1 << "/" << (samples + batch_size - 1) / batch_size 
                     << " | Loss: " << err / (j + 1) << flush;
            }
        }

        err /= samples;

        auto t_end = chrono::high_resolution_clock::now();
        double elapsed_time_s = chrono::duration<double>(t_end - t_start).count();

        // Epoch summary
        if (verbose >= 1) {
            cout << "\rEpoch " << i + 1 << "/" << epochs 
                 << " | Loss: " << err 
                 << " | Time: " << elapsed_time_s << "s" << endl;
        }

        m_error.push_back(err);
    }

    auto stop = chrono::high_resolution_clock::now();
    double duration = chrono::duration_cast<chrono::seconds>(stop - start).count();

    if (verbose >= 1) {
        cout << "\nTraining complete in " << duration << " seconds." << endl;
    }
}


/**
 * @brief 
 * 
 * @param y_train 
 * @param y_test 
 */
void Network::Evaluate(MatrixXd y_train, MatrixXd y_test) 
{

}


/**
 * @brief Predict data based on input data, forward propagation throughout the network.
 * 
 * @param input_data Matrix input data
 * @return vector<MatrixXd> The array of Matrix output res
 */
vector<MatrixXd> Network::Predict(MatrixXd input_data)
{
  int samples = input_data.rows();
  vector<MatrixXd> res;

  for (int i = 0; i < samples; i++) {
    MatrixXd output = input_data.row(i);

    for (int j = 0; j < m_layer.size(); j++) {
      output = m_layer[j]->FeedForward(output);
    }

    res.push_back(output);
  }

  return res;
}


void Network::SaveModel(string name)
{
  ofstream ofs(name.c_str(), ios::out | ios::binary | ios::trunc);

  int layer_size = m_layer.size();

  ofs.write(reinterpret_cast<const char*>(&layer_size), sizeof(int));
  
  for (int i = 0; i < m_layer.size(); i++) {
    m_layer[i]->SaveLayer(ofs);
  }

  ofs.close();
}


Network *Network::LoadModel(string name)
{
  Network *network = new Network();
  ifstream ifs(name.c_str(), ios::in | ios::binary);

  if (!ifs) {
    cerr << "Can't open file !!" << endl;
    delete network;
    return nullptr;
  }

  int layer_size = 0;
  ifs.read(reinterpret_cast<char*>(&layer_size), sizeof(layer_size));

  for (int i = 0; i < layer_size; i++) {
    Fc_Layer *layer = Fc_Layer::LoadLayer(ifs);
    network->Add(layer);
  }

  network->Use(new Mse());

  ifs.close();

  return network;
}

