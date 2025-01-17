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


// MatrixXd Conv_Layer::FeedForward(const MatrixXd& input_data)
// {
//   this->m_input = input_data;
//   MatrixXd out(this->m_nb_filters, input_data.size());
//   for (int i = 0; i < m_nb_filters; i++) {
//     MatrixXd filterArray(m_filter_size, m_filter_size);
//     filterArray = this->m_weights.row(i);
//     Map<MatrixXd> filterMatrix(filterArray.data(), m_filter_size, m_filter_size);
//     Map<MatrixXd> temp(Core::Correlate2D(input_data, filterMatrix, 1, SAME).data(), 1, input_data.size());
//     out.row(i) = temp;
//   }
//   this->m_output = out;
//   return this->m_output;
// }

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


void Conv_Layer::SaveLayer(std::ofstream &outfile)
{

}

void Conv_Layer::SetWeights(Eigen::MatrixXd &weights)
{

}

void Conv_Layer::SetBias(Eigen::MatrixXd &bias)
{

}