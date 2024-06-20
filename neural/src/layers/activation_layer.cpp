#include "layers/activation_layer.h"

using namespace Neural;
using namespace Eigen;

/**
 * @brief Construct a new Activation_Layer::Activation_Layer object
 * 
 */
Activation_Layer::Activation_Layer()
{
  this->m_as_weight = false;
}


/**
 * @brief Construct a new Activation_Layer::Activation_Layer object
 * 
 * @param a 
 */
Activation_Layer::Activation_Layer(Activation *a)
{
  this->m_as_weight = false;
  this->p_activation = a;
}


/**
 * @brief Destroy the Activation_Layer::Activation_Layer object
 * 
 */
Activation_Layer::~Activation_Layer()
{
  delete this->p_activation;
}


/**
 * @brief Performs forward propagation on the current layer.
 * 
 * @param input The inputs of the Layer = The outputs of the previous Layer, or The data of the first Layer 
 * @return MatrixXd Output Matrix of forward propagation results.
 */
MatrixXd Activation_Layer::FeedForward(MatrixXd input_data)
{
  this->m_input = input_data;
  return this->p_activation->Compute(input_data);
}


/**
 * @brief Performs backward propagation on the current layer.
 * 
 * @param output_error Ths inputs of the Layer = The outputs of the previous layer, or the 
 *                     data of the first layer.
 * @param learning_rate The step size at each iteration.
 * @return MatrixXd Matrix of input layer error.
 */
MatrixXd Activation_Layer::BackPropagation(MatrixXd output_error, float learning_rate)
{
  return this->p_activation->ComputeDerivative(this->m_input).array() * output_error.array();
}