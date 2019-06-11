
#include "kalman_filter.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <math.h>
#include <iostream>
#include <cmath>

using namespace Eigen;
using namespace std;

#define AVOID_SINGULAR_ZONE 1e-20

#define WARN_HINT 1
#define SHOW_DISP 0

#if WARN_HINT
  #define WARN(str) cout << str << endl;
#else
  #define WARN(str)
#endif
#if SHOW_DISP
  #define DISP(str) cout << str << endl;
  #define SHOW(x)   cout << "------"#x"------" << endl << x << endl;
#else
  #define DISP(str)
  #define SHOW(x)
#endif

template<typename Derived>
inline bool is_inf(const Eigen::MatrixBase<Derived>& x)
{
   return !( ((x - x).array() == (x - x).array()).all() );
}

template<typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
{
   return !( ((x.array() == x.array())).all() );
}

KALMAN_FILTER::KALMAN_FILTER(KALMAN_FILTER_CfgTypeDef* cfg){

  /* Time */
  this->Fs  = cfg->sample_hz;
  this->Ts  = 1.0f / this->Fs;
  this->Td  = cfg->delay_sec;
  if(Td > 0.0f){
    this->kn  = (size_t)( std::round(Fs*Td) );
    if(kn == 0){
      kn = 1;
    }
  }else{
    this->kn  = 0;
  }
  SHOW(Fs);
  SHOW(Ts);
  SHOW(Td);
  SHOW(kn);

  /* System */
  // resize
  this->nx = cfg->H.cols();
  this->ns = cfg->H.rows();
  this->x_k   = MatrixXd::Zero(nx, 1);
  this->x_k_1 = MatrixXd::Zero(nx, 1);
  this->P_k   = MatrixXd::Zero(nx,nx);
  this->P_k_1 = MatrixXd::Zero(nx,nx);
  this->z_k   = MatrixXd::Zero(ns, 1);
  this->F = cfg->F;
  this->B = cfg->B;
  this->H = cfg->H;
  SHOW(nx);
  SHOW(ns);
  SHOW(x_k);
  SHOW(x_k_1);
  SHOW(P_k);
  SHOW(P_k_1);
  SHOW(z_k);
  SHOW(F);
  SHOW(B);
  SHOW(H);

  /* Covariance */
  // resize
  this->Q = cfg->Q;
  this->R = cfg->R;
  this->O = MatrixXd::Identity(nx,nx) * 1e-7f;
  SHOW(Q);
  SHOW(R);
  SHOW(O);

  /* History */
  if(kn){
    this->x_.resize(nx, 1,kn);
    this->P_.resize(nx,nx,kn);
    this->M_.resize(nx,nx,kn);
  }

  /* Fast */
  this->I   = MatrixXd::Identity(nx,nx);
  this->Ft  = F.transpose();
  this->Ht  = H.transpose();
  SHOW(I);
  SHOW(Ft);
  SHOW(Ht);
  DISP("==============Creat================");

  /* Init */
  this->init();

}

void KALMAN_FILTER::init(MatrixXd *x0){
  /* System */
  this->x_k_1 = *x0;
  this->P_k_1 = B*Q*B.transpose() + O;
  this->x_k = x_k_1;
  this->P_k = P_k_1;
  SHOW(x_k_1);
  SHOW(P_k_1);
  DISP("==============Init=================");
  /* History */
  if(kn){
    this->x_.fill(&x_k_1);
    this->P_.fill(&P_k_1);
    this->M_.fill(&F);
  }

}

void KALMAN_FILTER::init(){
  MatrixXd x0 = MatrixXd::Zero(nx, 1);
  this->init(&x0);
}

void KALMAN_FILTER::update(MatrixXd *z, size_t kn){

  /***** Measurement *****/
  this->z_k = *z;

  /***** Update *****/
  /* Delayed Type */
  if(this->kn){

    // Check kn
    if(kn > this->kn){
      kn = this->kn;
      WARN("kn overflow");
    }else if(kn == 0){
      kn = 1;
    }

  /* Instant Type */
  }else{

    // Update
    this->update_();
    DISP("=============Instant Update===========");

  }

}

void KALMAN_FILTER::update(MatrixXd *z){

  this->update(z, kn);

}

void KALMAN_FILTER::update(){

  /* Pass Prediction to Estimation */
  x_k = x_k_1;
  P_k = P_k_1;

  /* FIFO */
  if(kn){
    M_.save(&F);
  }

  DISP("============== No Update =============");

}

void KALMAN_FILTER::predict(MatrixXd *u){

  /* State Prediction */
  x_k_1 = F*x_k + B * (*u);
  SHOW(x_k_1);

  /* Covariance Prediction */
  P_k_1 = F*P_k*Ft + Q;
  SHOW(P_k_1);

  /* FIFO */
  if(kn){
    x_.save(&x_k_1);
    P_.save(&P_k_1);
  }

  DISP("============== Predict ===============");

}

void KALMAN_FILTER::predict(){

  /* Zero Control Input */
  MatrixXd u = MatrixXd::Zero(nx, 1);
  this->predict(&u);

}

void KALMAN_FILTER::connect2(KALMAN_FILTER* next){
  next->x_k_1 = this->x_k;
  next->P_k_1 = this->P_k;
}
void KALMAN_FILTER::connect0(KALMAN_FILTER* last){
  last->x_k_1 = this->x_k_1;
  last->P_k_1 = this->P_k_1;
}
size_t KALMAN_FILTER::get_kn(){
  return this->kn;
}
MatrixXd* KALMAN_FILTER::get_z_k(){
  return &(this->z_k);
}
MatrixXd* KALMAN_FILTER::get_x_k(){
  return &(this->x_k);
}
MatrixXd* KALMAN_FILTER::get_x_k_1(){
  return &(this->x_k_1);
}
MatrixXd* KALMAN_FILTER::get_P_k(){
  return &(this->P_k);
}
MatrixXd* KALMAN_FILTER::get_P_k_1(){
  return &(this->P_k_1);
}
double KALMAN_FILTER::get_z_k(size_t idx){
  return (idx < this->ns) ? this->z_k(idx,0) : NAN;
}
double KALMAN_FILTER::get_x_k(size_t idx){
  return (idx < this->nx) ? this->x_k(idx,0) : NAN;
}



void KALMAN_FILTER::update_(){

  /* Innovation Residual */
  MatrixXd vu_k(ns, 1);
  vu_k = z_k - H*x_k_1;
  SHOW(vu_k);

  /* Innovation Covariance */
  MatrixXd Pvv_k(ns,ns);
  Pvv_k = H*P_k_1*Ht + R;
  SHOW(Pvv_k);
  SHOW(Pvv_k.determinant());
  if( Pvv_k.determinant() < AVOID_SINGULAR_ZONE || is_nan(Pvv_k) ){
    Pvv_k = R;
    WARN("Pvv_k Singular");
  }

  /* Kalman Gain */
  MatrixXd W_k(nx,ns);
  W_k = P_k_1*Ht*Pvv_k.inverse();
  SHOW(W_k);

  /* State Estimation */
  x_k = x_k_1 + W_k*vu_k;
  SHOW(x_k);

  /* Covariance Estimation */
  P_k = (I - W_k*H)*P_k_1;
  SHOW(P_k);
  SHOW(P_k.determinant());
  if( P_k.determinant() < AVOID_SINGULAR_ZONE || is_nan(P_k) ){
    P_k = O;
    WARN("P_k Singular");
  }

}
