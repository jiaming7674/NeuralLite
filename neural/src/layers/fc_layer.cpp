#include "layers/fc_layer.h"
#include "core.h"

using namespace std;
using namespace Neural;
using namespace Eigen;
//using Eigen::MatrixXd;

/**
 * @brief Construct a new Fc_Layer::Fc_Layer object
 * 
 * @param input_size 
 * @param output_size 
 */
Fc_Layer::Fc_Layer(int input_size, int output_size)
{
  this->m_as_weight = true;
  this->m_weights = Core::RandomMatrix(input_size, output_size, -0.5, 0.5);
  this->m_bias = Core::RandomMatrix(1, output_size, -0.5, 0.5);
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
  this->m_output = (input_data * this->m_weights) + this->m_bias;

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
  // CPU
  MatrixXd input_error = output_error * m_weights.transpose();
  MatrixXd weight_error = m_input.transpose() * output_error;

  this->m_weights.noalias() -= learning_rate * weight_error;
  this->m_bias.noalias() -= learning_rate * output_error;

  return input_error;
}
