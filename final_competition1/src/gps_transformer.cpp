/*
 *  gps_transformer.cpp
 *
 *  Created on: May 25, 2019
 *  Purpose: SDC Final Competition
 */

#include <gps_transformer.h>
// gps data
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>


static double lat_origin = 24.7855644226;
static double lon_origin = 120.997009277;
static double alt_origin = 127.651;
static GeographicLib::Geocentric earth(GeographicLib::Constants::WGS84_a(), GeographicLib::Constants::WGS84_f());
static GeographicLib::LocalCartesian proj = GeographicLib::LocalCartesian(lat_origin, lon_origin, alt_origin);


ros::Publisher _pub_marker;
// gps transformed data
ros::Publisher gps_transformed_pub;

visualization_msgs::Marker line_strip;

static void gnss_callback(const sensor_msgs::NavSatFixConstPtr &msg)
{
  double x, y, z;
 
  // Warning: if the measurement of altitude is wrong, then the result of projection will be wrong.
  proj.Forward(msg->latitude, msg->longitude, alt_origin, x, y, z);

  geometry_msgs::PointStamped p;

  p.point.x = x;
  p.point.y = y;
  p.point.z = z;
  p.header.stamp = msg->header.stamp;
  // publish gps transformed data
  gps_transformed_pub.publish(p);


  geometry_msgs::Point p_;
  p_.x = p.point.x;
  p_.y = p.point.y;
  p_.z = p.point.z;
  line_strip.points.push_back(p_);
  if(line_strip.points.size()>15000)
    line_strip.points.clear();
  
  _pub_marker.publish(line_strip);

  return;
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "gps_transformer");
  ros::NodeHandle nh;

  ros::Subscriber gnss_sub = nh.subscribe("fix", 100000, gnss_callback);
  _pub_marker = nh.advertise<visualization_msgs::Marker>("/gps_marker",1);
  gps_transformed_pub = nh.advertise<geometry_msgs::PointStamped>("/gps_transformed", 1);

  line_strip.header.frame_id = "/map";
  line_strip.ns = "linestrip";
  line_strip.action = visualization_msgs::Marker::ADD;
  line_strip.pose.orientation.w = 1.0;
  line_strip.id = 1;
  line_strip.type = visualization_msgs::Marker::LINE_STRIP;


  line_strip.scale.x = 1.0;

  line_strip.color.r = 1.0;
  line_strip.color.a = 1.0;

  ros::spin();
}
