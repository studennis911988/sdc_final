
#include "mat_fifo.h"
#include <Eigen/Core>
#include <Eigen/LU>

#include <iostream>

using namespace Eigen;

#define MAT_FIFO_CHECK_SIZE 1

MAT_FIFO::MAT_FIFO(){

}

void MAT_FIFO::init(){
  idx_ = len_;
}

void MAT_FIFO::resize(size_t data_row, size_t data_col, size_t fifo_len){
  row_ = data_row;
  col_ = data_col;
  len_ = fifo_len;
  this->m_ = MatrixXd::Zero(row_*len_, col_);
  this->init();
}

void MAT_FIFO::fill(MatrixXd *data){
#if MAT_FIFO_CHECK_SIZE
  if(data->rows() != row_ || data->cols() != col_){
    std::cout << "fifo size err" << std::endl;
    return;
  }
#endif

  for(int ii=0; ii < len_; ii++){
    this->save(data);
  }

}

void MAT_FIFO::save(MatrixXd *data){
#if MAT_FIFO_CHECK_SIZE
  if(data->rows() != row_ || data->cols() != col_){
    std::cout << "fifo size err" << std::endl;
    return;
  }
#endif

  this->idxNext_();
  this->m_.block(idx_*row_,0, row_, col_) << *data;

}

MatrixXd MAT_FIFO::peek(size_t sample){

#if MAT_FIFO_CHECK_SIZE
  if(sample>len_ || sample==0){
    std::cout << "fifo kn err" << std::endl;
    return MatrixXd::Zero(row_, col_);
  }
#endif

  size_t idx_kn = idxBack_(idx_, sample-1);

  return m_.block(idx_kn*row_, 0, row_, col_);

}

void MAT_FIFO::idxNext_(){
  this->idx_++;
  if(idx_ >= len_){
    idx_ -= len_;
  }
}

size_t MAT_FIFO::idxBack_(size_t idx, size_t sample){

  int32_t idx_kn = idx - sample;

  while( idx_kn < 0 ){
    idx_kn += len_;
  }

  return (size_t)idx_kn;
}
