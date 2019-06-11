#ifndef MAT_FIFO_H
#define MAT_FIFO_H

#include <Eigen/Core>
#include <Eigen/LU>

using namespace Eigen;

class MAT_FIFO
{
public:
  MAT_FIFO();

  void      init();
  void      resize(size_t data_row, size_t data_col, size_t fifo_len);
  void      fill(MatrixXd *data);
  void      save(MatrixXd *data);
  MatrixXd  peek(size_t sample);

private:
  MatrixXd  m_;
  size_t    row_;
  size_t    col_;
  size_t    len_;
  size_t    idx_;

  void  idxNext_();
  size_t idxBack_(size_t idx, size_t sample);
};

#endif // MAT_FIFO_H
