#include "layers/fc_layer.h"
#include "core.h"
#include <fstream>

using namespace std;
using namespace Neural;
using namespace Eigen;
//using Eigen::MatrixXd;

/**
 * @brief Construct a new Fc_Layer::Fc_Layer object
 * 
 * @param input_size size of input data
 * @param output_size size of output data
 */
Fc_Layer::Fc_Layer(int input_size, int output_size, ActivationType activationType)
{
  this->m_as_weight = true;
  this->m_weights = Core::RandomMatrix(input_size, output_size, -1.0, 1.0);
  this->m_bias = Core::RandomMatrix(1, output_size, -0.5, 0.5);

  switch (activationType) {
    case ActivationType::SIGMOID:
      this->p_activation = new Sigmoid();
      break;
    case ActivationType::RELU:
      this->p_activation = new ReLU();
      break;
    case ActivationType::LEAKY_RELU:
      this->p_activation = new LeakyReLU();
      break;
    case ActivationType::ELU:
      this->p_activation = new ELU();
      break;
    case ActivationType::TANH:
      this->p_activation = new Tanh();
      break;
    case ActivationType::SOFTMAX:
      this->p_activation = new Softmax();
      break;
    default:
      this->p_activation = nullptr;
  }
}


/**
 * @brief Performs forward propagation on the current layer.
 * 
 * @param input The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 * @return MatrixXd Output Matrix of forward propagation results.
 */
MatrixXd Fc_Layer::FeedForward(const MatrixXd& input_data)
{
  this->m_input = input_data;
  this->m_net_sum = (input_data * this->m_weights) + this->m_bias;
  
  // calculate activation function output
  if (p_activation != nullptr)
    this->m_output = p_activation->Compute(m_net_sum);
  else
    this->m_output = m_net_sum;

  return m_output;
}


/**
 * @brief Performs backward propagation on the current layer.
 * 
 * @param output_error The error of the layer's output, which is the difference between
 *                     the expected output and the actual output.
 * @param learning_rate The step size at each iteration for updating weights and biases.
 * @return MatrixXd The error of the input layer, which will be propagated backward to 
 *                  the previous layer.
 */
MatrixXd Fc_Layer::BackPropagation(const MatrixXd& output_error, float learning_rate)
{
  Eigen::MatrixXd gradient;

  if (this->p_activation != nullptr)
    gradient = this->p_activation->ComputeDerivative(this->m_net_sum).array() * output_error.array();
  else
    gradient = output_error;

  MatrixXd input_error = gradient * m_weights.transpose();
  MatrixXd weight_error = m_input.transpose() * gradient;

  if (this->m_optimizer != nullptr) {
    MatrixXd bias_error = gradient;
    m_optimizer->UpdateWeights(m_weights, weight_error);
    m_optimizer->UpdateBias(m_bias, bias_error);
  }
  else {
    this->m_weights.noalias() -= learning_rate * weight_error;
    this->m_bias.noalias() -= learning_rate * gradient;
  }

  return input_error;
}


/**
 * @brief Saves the layer's configuration and parameters to an output file stream.
 * 
 * @param outfile The output file stream to which the layer's data will be written.
 */
void Fc_Layer::SaveLayer(ofstream &outfile)
{
  // write activation type
  ActivationType type;
  if (this->p_activation != nullptr) {
    type = this->p_activation->getType();
  }
  else {
    type = ActivationType::NONE;
  }

  outfile.write(reinterpret_cast<const char*>(&type), sizeof(type));

  // write the weights of the layer
  int rows = this->m_weights.rows();
  int cols = this->m_weights.cols();
  outfile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
  outfile.write(reinterpret_cast<const char*>(&cols), sizeof(int));
  outfile.write(reinterpret_cast<const char*>(m_weights.data()), rows * cols * sizeof(double));

  rows = this->m_bias.rows();
  cols = this->m_bias.cols();

  // write the bias of the layer
  outfile.write(reinterpret_cast<const char*>(&rows), sizeof(int));
  outfile.write(reinterpret_cast<const char*>(&cols), sizeof(int));
  outfile.write(reinterpret_cast<const char*>(m_bias.data()), rows * cols * sizeof(double));
}


/**
 * @brief Loads the layer's configuration and parameters from an input file stream.
 * 
 * @param infile The input file stream from which the layer's data will be read.
 * @return Fc_Layer* A pointer to the newly created Fc_Layer with the loaded parameters.
 */
Fc_Layer* Fc_Layer::LoadLayer(ifstream &infile)
{
  ActivationType type;
  infile.read(reinterpret_cast<char*>(&type), sizeof(type));

  int rows, cols;
  infile.read(reinterpret_cast<char*>(&rows), sizeof(int));
  infile.read(reinterpret_cast<char*>(&cols), sizeof(int));

  Fc_Layer *layer = new Fc_Layer(rows, cols, type);

  MatrixXd weights;
  weights.resize(rows, cols);
  infile.read(reinterpret_cast<char*>(weights.data()), rows * cols * sizeof(double));
  layer->SetWeights(weights);

  infile.read(reinterpret_cast<char*>(&rows), sizeof(int));
  infile.read(reinterpret_cast<char*>(&cols), sizeof(int));

  MatrixXd bias;
  bias.resize(rows, cols);
  infile.read(reinterpret_cast<char*>(bias.data()), rows * cols * sizeof(double));
  layer->SetBias(bias);

  return layer;
}


/**
 * @brief Sets the weights of the layer.
 * 
 * @param weights A matrix containing the new weights for the layer.
 */
void Fc_Layer::SetWeights(Eigen::MatrixXd &weights)
{
  this->m_weights = weights;
}


/**
 * @brief Sets the biases of the layer.
 * 
 * @param bias A matrix containing the new biases for the layer.
 */
void Fc_Layer::SetBias(Eigen::MatrixXd &bias)
{
  this->m_bias = bias;
}