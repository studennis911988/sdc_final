/***
 *   SDC Final Competition 1 - localization TEAM 11
 *   Date : 2019/6/9
 *   Try to fuse imu and gps
***/
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
/* sensor message */
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>

/* kalman filter */
#include "kalman_filter.h"
#include "MadgwickAHRS.h"

using namespace Eigen;

/* parameters */
double sampling_rate = 200.0f;
double imu_covariance = 0.0f;
double gps_covariance = 0.0f;

/* global object */
// publisher
ros::Publisher fusion_pub;
ros::Publisher fusion_marker_pub;
geometry_msgs::PoseStamped fusion_data;
visualization_msgs::Marker fusion_marker;

/* kalman filter */
bool is_init = false;
// IMU
KALMAN_FILTER* accelKF_x = NULL;
KALMAN_FILTER* accelKF_y = NULL;
KALMAN_FILTER* accelKF_z = NULL;
// GPS
KALMAN_FILTER* gpsKF_x = NULL;
KALMAN_FILTER* gpsKF_y = NULL;
KALMAN_FILTER* gpsKF_z = NULL;
// lidar
KALMAN_FILTER* lidarKF_x = NULL;
KALMAN_FILTER* lidarKF_y = NULL;
KALMAN_FILTER* lidarKF_z = NULL;

/* sensor data */
bool imu_new = false;
bool gps_new = false;
bool lidar_new = false;

// IMU
//geometry_msgs::Quaternion imu_q;
Vector3d                  imu_accel;
Vector3d                  imu_accel_map;
Vector3d                  imu_gyro;
ros::Time                 imu_time;
// GPS
geometry_msgs::Quaternion gps_q;
Vector3d                  gps_p;
Vector3d*                 gps_init_p = NULL;
bool                      get_init_gps = false;
//Vector3d                  gps_last_p;
ros::Time                 gps_time;

// LIDAR
geometry_msgs::Quaternion lidar_q;
Vector3d                  lidar_p;
ros::Time                 lidar_time;





/**
 * @brief get gps data for kalman filter
 * @param gps_msg
 * ***********   I supose tha
 */
static void gpsMsgCallback(const geometry_msgs::PointStampedPtr& gps_msg) {
  gps_new = true;
  gps_p << gps_msg->point.x,
           gps_msg->point.y,
           gps_msg->point.z;
  gps_time = gps_msg->header.stamp;
//  ROS_INFO("gps msg x:%f \t y:%f \t z:%f", gps_p[0], gps_p[1], gps_p[2] );
  // initial pose
  if(gps_init_p == NULL) {
    gps_init_p = new Vector3d(gps_p[0],
                              gps_p[1],
                              gps_p[2]);
    get_init_gps = true;
  }

}

/**
 * @brief get imu data for kalman filter
 * @param imu_msg
 */
static void imuMsgCallback(const sensor_msgs::ImuPtr& imu_msg) {
  imu_new = true;
//  imu_q = imu_msg->orientation;

  imu_accel << imu_msg->linear_acceleration.x,
               imu_msg->linear_acceleration.y,
               imu_msg->linear_acceleration.z;

  imu_gyro <<  imu_msg->angular_velocity.x,
               imu_msg->angular_velocity.y,
               imu_msg->angular_velocity.z;


  imu_time = imu_msg->header.stamp;
}

/**
 * @brief fusion data by kalman filter
 */
static void fusionTask();
// init state for kalman
static void initState();
static void FusionandPublish();

static void imu_body2map();

static Matrix3d quaternion2DCM(geometry_msgs::Quaternion* q);


int main(int argc, char **argv)
{
  ros::init(argc, argv, "localization");
  ros::NodeHandle nh;

  /* parameters */
  ros::NodeHandle param_nh("~");
  param_nh.getParam("sampling_rate"  , sampling_rate);
  param_nh.getParam("imu_covariance" , imu_covariance);
  param_nh.getParam("gps_covariance" , gps_covariance);



  /* subscriber */
  // imu subscriber
  ros::Subscriber imu_sub = nh.subscribe("/imu/data", 1, imuMsgCallback);
  // gps subscriber
  ros::Subscriber gps_sub = nh.subscribe("/gps_transformed", 1, gpsMsgCallback);

  /* publisher */
  // publish fusion data marker
  fusion_pub = nh.advertise<geometry_msgs::PoseStamped>("/fusion/data", 1);
  fusion_marker_pub = nh.advertise<visualization_msgs::Marker>("/fusion/marker", 1);

  /* initalized kalman filter */
  KALMAN_FILTER_CfgTypeDef cfg;
  double T      = 1 / sampling_rate;
  cfg.sample_hz = sampling_rate;
  cfg.F.resize(3,3);
  cfg.B.resize(3,3);
  cfg.H.resize(1,3);
  cfg.Q.resize(3,3);
  cfg.R.resize(1,1);
  cfg.F <<  1,  T,  0.5*T*T,
            0,  1,  T,
            0,  0,  1;
  cfg.B <<  0.5*T*T,  0,  0,
            0,        T,  0,
            0,        0,  1;
  cfg.Q <<  0,  0,  0,
            0,  0,  0,
            0,  0,  1;

  // accel
  cfg.delay_sec = 0.0f;
  cfg.H << 0, 0, 1;
  cfg.R << imu_covariance;
  accelKF_x = new KALMAN_FILTER(&cfg);
  accelKF_y = new KALMAN_FILTER(&cfg);

  // gps
  cfg.delay_sec = 0.00f;
  cfg.H << 1, 0, 0;
  cfg.R << gps_covariance;
  gpsKF_x = new KALMAN_FILTER(&cfg);
  gpsKF_y = new KALMAN_FILTER(&cfg);

  /* infinite loop */
  ros::Rate loop_rate(sampling_rate);
  while(ros::ok()) {
    /* KALMAN START here */
    fusionTask();

    ros::spinOnce();
    loop_rate.sleep();
  }
  ros::spin();
  return 0;
}


static void fusionTask(){
  if(get_init_gps && imu_new && gps_new && !is_init) {
    initState();
    ROS_INFO("============ finish initalization ===============");
  }
  else if(is_init) {
    // transformed to map frame
    if(imu_new) {
      imu_body2map();
    }
    /** main fusion part **/
    FusionandPublish();
  }
  else {
    ROS_INFO("============ didn't init yet ================");
  }
}











/* small local function */
static void initState() {
    MatrixXd initial_pose_x;
    initial_pose_x.resize(3,1);
    initial_pose_x << (*gps_init_p)[0],
                      0.0f,
                      0.0f;
    MatrixXd initial_pose_y;
    initial_pose_y.resize(3,1);
    initial_pose_y << (*gps_init_p)[1],
                      0.0f,
                      0.0f;
    accelKF_x->init(&initial_pose_x);
    accelKF_y->init(&initial_pose_y);
    gpsKF_x->init(&initial_pose_x);
    gpsKF_y->init(&initial_pose_y);
    is_init = true;
}


static void FusionandPublish() {
  MatrixXd ax = imu_accel.row(0);
  MatrixXd ay = imu_accel.row(1);
  MatrixXd px = gps_p.row(0);
  MatrixXd py = gps_p.row(1);
  MatrixXd* xx;
  MatrixXd* xy;

  // measurement update
  if(imu_new) {
    accelKF_x->update(&ax);
    accelKF_y->update(&ay);
  }else {
    accelKF_x->update();
    accelKF_y->update();
  }

  // imu connect to gps
  accelKF_x->connect2(gpsKF_x);
  accelKF_y->connect2(gpsKF_y);


  if(gps_new) {
    double dT = (imu_time - gps_time).toSec();
    size_t knn = (size_t)std::round(dT * sampling_rate);

    gpsKF_x->update(&px);
    gpsKF_y->update(&py);
  }else {
    gpsKF_x->update();
    gpsKF_y->update();
  }

  // predict
  gpsKF_x->predict();
  gpsKF_y->predict();

  // connect back to imu
  gpsKF_x->connect0(accelKF_x);
  gpsKF_y->connect0(accelKF_y);

  // reset flag;
  imu_new = false;
  gps_new = false;

  /* publish data */
  fusion_data.header.frame_id = "/map";
  fusion_data.pose.position.x = gpsKF_x->get_x_k(0);
  fusion_data.pose.position.y = gpsKF_y->get_x_k(0);

  fusion_pub.publish(fusion_data);

  /* publish marker */
  fusion_marker.header.frame_id = "/map";
  fusion_marker.ns = "linestrip";
  fusion_marker.action = visualization_msgs::Marker::ADD;
  fusion_marker.type = visualization_msgs::Marker::LINE_STRIP;
  fusion_marker.scale.x = 1.0f;
  fusion_marker.color.b = 1.0f;
  fusion_marker.color.a = 1.0f;

  geometry_msgs::Point position;
  position.x = gpsKF_x->get_x_k(0);
  position.y = gpsKF_y->get_x_k(0);
  position.z = gps_p[2];
  fusion_marker.points.push_back(position);

  fusion_marker_pub.publish(fusion_marker);
}

static void imu_body2map() {

  /* imu -> MadgwickFIlter -> quaterion */
  MadgwickAHRSupdateIMU((float)imu_gyro[0], (float)imu_gyro[1], (float)imu_gyro[2],
                        (float)imu_accel[0], (float)imu_accel[1], (float)imu_accel[2]);

  /* imu body frame to map frame */
  // return quaternion
  geometry_msgs::Quaternion q;
  q.w = (double)q0;
  q.x = (double)q1;
  q.y = (double)q2;
  q.z = (double)q3;

  Matrix3d DCM = quaternion2DCM(&q);
  imu_accel_map = DCM * imu_accel;
}



static Matrix3d quaternion2DCM(geometry_msgs::Quaternion* q){
  double q0q0 = q->w * q->w;
  double q0q1 = q->w * q->x;
  double q0q2 = q->w * q->y;
  double q0q3 = q->w * q->z;
  double q1q1 = q->x * q->x;
  double q1q2 = q->x * q->y;
  double q1q3 = q->x * q->z;
  double q2q2 = q->y * q->y;
  double q2q3 = q->y * q->z;
  double q3q3 = q->z * q->z;
  Matrix3d DCM;
  DCM(0,0) = q0q0 + q1q1 - q2q2 - q3q3;
  DCM(0,1) = 2.0f*( q1q2 - q0q3 );
  DCM(0,2) = 2.0f*( q0q2 + q1q3 );
  DCM(1,0) = 2.0f*( q1q2 + q0q3 );
  DCM(1,1) = q0q0 - q1q1 + q2q2 - q3q3;
  DCM(1,2) = 2.0f*( q2q3 - q0q1 );
  DCM(2,0) = 2.0f*( q1q3 - q0q2 );
  DCM(2,1) = 2.0f*( q0q1 + q2q3 );
  DCM(2,2) = q0q0 - q1q1 - q2q2 + q3q3;
  return DCM;
}















