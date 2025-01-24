#include "layers/conv_layer.h"
#include "core.h"
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace Neural;

Conv_Layer::Conv_Layer(tuple<int, int, int> dimensions,
                    tuple<int, int, int> filter,
                    int stride,
                    int padding)
{
  this->m_depth = get<2>(dimensions);
  this->m_height = get<0>(dimensions);
  this->m_width = get<1>(dimensions);
  this->m_filter_size = get<0>(filter);
  this->m_nb_filters = get<2>(filter);
  this->m_stride = stride;
  this->m_padding = padding;

  MatrixXd filters(m_nb_filters, m_filter_size*m_filter_size);
  for (int i = 0; i < m_nb_filters; i++) {
    MatrixXd mat = Core::RandomMatrix(m_filter_size, m_filter_size, 0, 1);
    Map<RowVectorXd> v1(mat.data(), mat.size());
    filters.row(i) = v1;
  }

  this->m_weights = filters;
}

#if 1

MatrixXd Conv_Layer::FeedForward(const MatrixXd& input_data)
{
  this->m_input = input_data;
  MatrixXd out(this->m_nb_filters, input_data.size());
  for (int i = 0; i < m_nb_filters; i++) {
    Map<MatrixXd> filterMatrix(m_weights.row(i).data(), m_filter_size, m_filter_size);

    MatrixXd temp = Core::Correlate2D(input_data, filterMatrix, 1, SAME);
    Map<MatrixXd> temp_mapped(temp.data(), 1, input_data.size());
    out.row(i) = temp_mapped;
  }
  this->m_output = out;
  return this->m_output;
}


MatrixXd Conv_Layer::BackPropagation(const MatrixXd& output_error, float learning_rate)
{
  MatrixXd in_error(m_nb_filters, m_input.size());
  MatrixXd d_weights(m_weights.rows(), m_weights.cols());
  // Copy output_error
  MatrixXd _output_error = output_error;

  for (int i = 0; i < m_nb_filters; i++) {
    Map<MatrixXd> filterMatrix(m_weights.row(i).data(), m_filter_size, m_filter_size);
    Map<MatrixXd> outerr(_output_error.data(), this->m_height, this->m_width);
    
    // Calculate input error
    MatrixXd temp_err = Core::Correlate2D(outerr, filterMatrix, 1, SAME);
    Map<MatrixXd> temperr(temp_err.data(), 1, m_input.size());
    in_error.row(i) = temperr;

    // Calculate weights and gradients
    Map<MatrixXd> output_matrix(_output_error.row(i).data(), m_filter_size, m_filter_size);
    MatrixXd d_w = Core::Correlate2D(m_input, output_matrix, 1, SAME);
    Map<MatrixXd> d_w_mapped(d_w.data(), 1, m_filter_size * m_filter_size);
    d_weights.row(i) = d_w_mapped;
  }

  this->m_weights.noalias() -= learning_rate * d_weights;
  return in_error;
}

#else

MatrixXd Conv_Layer::FeedForward(const MatrixXd& input_data)
{
  this->m_input = input_data;
  int batch_size = input_data.rows();
  
  // initialzie output matrix
  MatrixXd out(batch_size, m_nb_filters * m_height * m_width);

  for (int b = 0; b < batch_size; b++) {
    Map<MatrixXd> input_reshaped(m_input.row(b).data(), m_height, m_width);

    for (int i = 0; i < m_nb_filters; i++) {
      Map<MatrixXd> filterMatrix(m_weights.row(i).data(), m_filter_size, m_filter_size);
      MatrixXd temp = Core::Correlate2D(input_reshaped, filterMatrix, 1, SAME);
      Map<MatrixXd> temp_mapped(temp.data(), 1, m_height * m_width);
      out.block(b, i * m_height * m_width, 1, m_height * m_width) = temp_mapped;
    }
  }

  this->m_output = out;

  return this->m_output;
}


MatrixXd Conv_Layer::BackPropagation(const MatrixXd &output_error, float learning_rate)
{
  int batch_size = output_error.rows();

  MatrixXd in_error(batch_size, m_input.size() * this->m_nb_filters);
  MatrixXd d_weights = MatrixXd::Zero(m_weights.rows(), m_weights.cols());

  MatrixXd _output_error = output_error;

  for (int b = 0; b < batch_size; b++)
  {
    Map<MatrixXd> outerr(_output_error.row(b).data(), this->m_height, this->m_width);

    for (int i = 0; i < m_nb_filters; i++)
    {
      Map<MatrixXd> filterMatrix(m_weights.row(i).data(), m_filter_size, m_filter_size);
      MatrixXd temp_err = Core::Correlate2D(outerr, filterMatrix, 1, SAME);

      Map<MatrixXd> temperr(temp_err.data(), 1, m_height * m_width);
      in_error.block(b, i * m_height * m_width, 1, m_height * m_width) = temperr;

      Map<MatrixXd> output_matrix(_output_error.row(b).data(), m_filter_size, m_filter_size);
      Map<MatrixXd> input_reshaped(m_input.row(b).data(), m_height, m_width);
      MatrixXd d_w = Core::Correlate2D(input_reshaped, output_matrix, 1, SAME);
      Map<MatrixXd> d_w_mapped(d_w.data(), 1, m_filter_size * m_filter_size);
      d_weights.row(i) += d_w_mapped;
    }
  }

  this->m_weights.noalias() -= learning_rate * d_weights / batch_size;
  return in_error;
}

#endif

void Conv_Layer::SaveLayer(std::ofstream &outfile)
{

}

void Conv_Layer::SetWeights(Eigen::MatrixXd &weights)
{

}

void Conv_Layer::SetBias(Eigen::MatrixXd &bias)
{

}