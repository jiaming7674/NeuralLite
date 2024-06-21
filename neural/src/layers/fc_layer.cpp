#include "layers/fc_layer.h"
#include "core.h"

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
  this->m_weights = Core::RandomMatrix(input_size, output_size, -0.5, 0.5);
  this->m_bias = Core::RandomMatrix(1, output_size, -0.5, 0.5);

  switch (activationType) {
    case ActivationType::TANH:
      this->p_activation = new Than();
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
MatrixXd Fc_Layer::FeedForward(MatrixXd input_data)
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
 * @param output_error Ths inputs of the Layer = The outputs of the previous layer, or the 
 *                     data of the first layer.
 * @param learning_rate The step size at each iteration.
 * @return MatrixXd Matrix of input layer error.
 */
MatrixXd Fc_Layer::BackPropagation(MatrixXd output_error, float learning_rate)
{
  Eigen::MatrixXd gradient;

  if (this->p_activation != nullptr)
    gradient = this->p_activation->ComputeDerivative(this->m_net_sum).array() * output_error.array();
  else
    gradient = output_error;

  MatrixXd input_error = gradient * m_weights.transpose();
  MatrixXd weight_error = m_input.transpose() * gradient;

  this->m_weights.noalias() -= learning_rate * weight_error;
  this->m_bias.noalias() -= learning_rate * gradient;

  return input_error;
}


MatrixXd Fc_Layer::GetWeights(void)
{
  return this->m_weights;
}


MatrixXd Fc_Layer::GetBias(void)
{
  return this->m_bias;
}


void Fc_Layer::SetWeights(MatrixXd weights)
{
  this->m_weights = weights;
}


void Fc_Layer::SetBias(MatrixXd bias)
{
  this->m_bias = bias;
}