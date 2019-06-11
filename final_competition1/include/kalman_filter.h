#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <Eigen/Core>
#include <Eigen/LU>
#include "mat_fifo.h"

using namespace Eigen;

typedef struct{
  double  sample_hz;
  double  delay_sec;
  MatrixXd  F;
  MatrixXd  B;
  MatrixXd  H;
  MatrixXd  Q;
  MatrixXd  R;
}KALMAN_FILTER_CfgTypeDef;

class KALMAN_FILTER
{
public:
  KALMAN_FILTER(KALMAN_FILTER_CfgTypeDef* cfg);

  void      init(MatrixXd *x0);
  void      init();
  void      update(MatrixXd* z, size_t kn);
  void      update(MatrixXd* z);
  void      update();
  void      predict(MatrixXd* u);
  void      predict();
  void      connect2(KALMAN_FILTER* next);
  void      connect0(KALMAN_FILTER* last);
  size_t    get_kn();
  MatrixXd* get_z_k();
  MatrixXd* get_x_k();
  MatrixXd* get_x_k_1();
  MatrixXd* get_P_k();
  MatrixXd* get_P_k_1();
  double    get_z_k(size_t idx);
  double    get_x_k(size_t idx);

private:
  /* Time */
  // Sampling rate
  double Fs;
  // Sampling time
  double Ts;
  // Delayed time
  double Td;
  // Delayed samples
  size_t kn;

  /* System */
  // state numbers
  size_t    nx;
  // sensor numbers
  size_t    ns;
  // estimated state
  MatrixXd  x_k;
  // predicted state
  MatrixXd  x_k_1;
  // estimated covariance
  MatrixXd  P_k;
  // predicted covariance
  MatrixXd  P_k_1;
  // measurement value
  MatrixXd  z_k;
  // System
  MatrixXd  F;
  MatrixXd  B;
  MatrixXd  H;

  /* Covariance */
  // Process covariance
  MatrixXd  Q;
  // Measurement covariance
  MatrixXd  R;
  // Random Matrix with variance=1e-14
  MatrixXd  O;

  /* History */
  MAT_FIFO  x_;
  MAT_FIFO  P_;
  MAT_FIFO  M_;

  /* Fast */
  MatrixXd  I;
  MatrixXd  Ft;
  MatrixXd  Ht;

  /* Function */
  void      update_(size_t kn);
  void      update_();

};
#endif // KALMAN_FILTER_H
