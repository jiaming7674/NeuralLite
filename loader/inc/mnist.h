#ifndef __MNIST_H__
#define __MNIST_H__

#include <vector>
#include <string>
#include <Eigen/Dense>

class mnist {
public:
  std::vector<std::vector<double>> m_images;
  std::vector<int> m_labels;
  int m_size;
  int m_rows;
  int m_cols;

  struct {
    Eigen::MatrixXd images;
    Eigen::MatrixXd labels;
  } data;

public:
  mnist(std::string image_file, std::string label_file, int num);
  mnist(std::string image_file, std::string label_file);
  ~mnist();

  void load_images(std::string image_file, int num = 0);
  void load_labels(std::string label_file, int num = 0);
  int to_int(const char *p);
};



#endif