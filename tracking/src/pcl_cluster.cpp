#include <ros/ros.h>
#include <iostream>
#include <geometry_msgs/Pose.h>
#include <string>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Dense>
#include <pcl/features/fpfh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <visualization_msgs/MarkerArray.h>
#include <iomanip>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/console/time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/ColorRGBA.h>


typedef pcl::PointXYZINormal PointTypeFull;
using namespace boost::filesystem;

int WIDTH = 1280, HEIGHT = 720;
Eigen::Matrix4f l_t_c;
cv::Mat INTRINSIC_MAT(3, 3, cv::DataType<double>::type); // Intrinsics
cv::Mat DIST_COEFFS(5, 1, cv::DataType<double>::type); // Distortion vector

ros::Publisher plane_filtered_pub;
ros::Publisher cluster_pub;
ros::Publisher cloud_pub;
ros::Publisher marker_pub1;
ros::Publisher marker_pub2;
ros::Publisher marker_pub3;
ros::Publisher cube_pub;


void initializeGlobalParams() {
  l_t_c << 0.84592974185943604 , 0.53328412771224976  , -0.0033089336939156055, 0.092240132391452789,
           0.045996580272912979, -0.079141519963741302, -0.99580162763595581  , -0.35709697008132935,
           -0.53130710124969482, 0.84222602844238281  , -0.091477409005165100 , -0.16055910289287567,
           0,                                        0,                      0,                    1;

  INTRINSIC_MAT.at<double>(0, 0) = 698.939;
  INTRINSIC_MAT.at<double>(1, 0) = 0;
  INTRINSIC_MAT.at<double>(2, 0) = 0;

  INTRINSIC_MAT.at<double>(0, 1) = 0;
  INTRINSIC_MAT.at<double>(1, 1) = 698.939;
  INTRINSIC_MAT.at<double>(2, 1) = 0;

  INTRINSIC_MAT.at<double>(0, 2) = 641.868;
  INTRINSIC_MAT.at<double>(1, 2) = 385.788;
  INTRINSIC_MAT.at<double>(2, 2) = 1.0;

  DIST_COEFFS.at<double>(0) = -0.171466;
  DIST_COEFFS.at<double>(1) = 0.0246144;
  DIST_COEFFS.at<double>(2) = 0;
  DIST_COEFFS.at<double>(3) = 0;
  DIST_COEFFS.at<double>(4) = 0;
}

bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  if (squared_distance < 10000)
  {
    if (fabs (point_a.intensity - point_b.intensity) < 8.0f)
      return (true);
    if (fabs (point_a_normal.dot (point_b_normal)) < 0.06)
      return (true);
  }
  else
  {
    if (fabs (point_a.intensity - point_b.intensity) < 3.0f)
      return (true);
  }
  return (false);
}


void lidar_callback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  fromROSMsg(*msg, *cloud);
  pcl::console::TicToc tt;

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZI> vg;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.1, 0.1, 0.1);
  vg.filter (*cloud_filtered);

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZI> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZI> ());
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.3);

  // Segment the largest planar component from the remaining cloud
  seg.setInputCloud (cloud_filtered);
  seg.segment (*inliers, *coefficients);
  if (inliers->indices.size () == 0) {
    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  // Extract the planar inliers from the input cloud
  pcl::ExtractIndices<pcl::PointXYZI> extract;
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers);
  extract.setNegative (false);

  // Get the points associated with the planar surface
  extract.filter (*cloud_plane);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZI>);
  // Remove the planar inliers, extract the rest
  extract.setNegative (true);
  extract.filter (*cloud_f);
  *cloud_filtered = *cloud_f;

  sensor_msgs::PointCloud2 filtered_cloud;
  pcl::toROSMsg(*cloud_plane, filtered_cloud);
  filtered_cloud.header.frame_id = "velodyne";
  plane_filtered_pub.publish(filtered_cloud);

//  pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>);
//  pcl::search::KdTree<pcl::PointXYZI>::Ptr search_tree (new pcl::search::KdTree<pcl::PointXYZI>);
//  pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
//  // Set up a Normal Estimation class and merge data in cloud_with_normals
//    std::cerr << "Computing normals...\n", tt.tic ();
//    pcl::copyPointCloud (*cloud_filtered, *cloud_with_normals);
//    pcl::NormalEstimation<pcl::PointXYZI, PointTypeFull> ne;
//    ne.setInputCloud (cloud_filtered);
//    ne.setSearchMethod (search_tree);
//    ne.setRadiusSearch (10.0);
//    ne.compute (*cloud_with_normals);
//    std::cerr << ">> Done: " << tt.toc () << " ms\n";

//    // Set up a Conditional Euclidean Clustering class
//    std::cerr << "Segmenting to clusters...\n", tt.tic ();
//    pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
//    cec.setInputCloud (cloud_with_normals);
//    cec.setConditionFunction (&customRegionGrowing);
//    cec.setClusterTolerance (0.3);
//    cec.setMinClusterSize (cloud_with_normals->points.size () / 1000);
//    cec.setMaxClusterSize (cloud_with_normals->points.size () / 5);
//    cec.segment (*clusters);
//    cec.getRemovedClusters (small_clusters, large_clusters);
//    std::cerr << ">> Done: " << tt.toc () << " ms\n";

//    // Using the intensity channel for lazy visualization of the output
//    for (int i = 0; i < small_clusters->size (); ++i)
//      for (int j = 0; j < (*small_clusters)[i].indices.size (); ++j)
//        cloud_filtered->points[(*small_clusters)[i].indices[j]].intensity = -2.0;
//    for (int i = 0; i < large_clusters->size (); ++i)
//      for (int j = 0; j < (*large_clusters)[i].indices.size (); ++j)
//        cloud_filtered->points[(*large_clusters)[i].indices[j]].intensity = +10.0;
//    for (int i = 0; i < clusters->size (); ++i)
//    {
//      int label = rand () % 8;
//      for (int j = 0; j < (*clusters)[i].indices.size (); ++j)
//        cloud_filtered->points[(*clusters)[i].indices[j]].intensity = label;
//    }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
  ec.setClusterTolerance (0.3); // 2cm
  ec.setMinClusterSize (30);
  ec.setMaxClusterSize (1500);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);
  std::cout << "Cluster size: " << cluster_indices.size() << std::endl;

  // compute cluster centroid and publish
  pcl::PointCloud<pcl::PointXYZ>::Ptr centers (new pcl::PointCloud<pcl::PointXYZ>);
  visualization_msgs::Marker cube;
  visualization_msgs::Marker markerx;
  visualization_msgs::Marker markery;
  visualization_msgs::Marker markerz;
  for(int j=0;j<cluster_indices.size();j++){
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZI>);
      for(int i=0;i<cluster_indices[j].indices.size();i++){
    //    std::cout << "Cluster 0x: " << cloud_filtered->points[cluster_indices[0].indices[i]].x << std::endl;
        cloud0->push_back(cloud_filtered->points[cluster_indices[j].indices[i]]);
      }

      pcl::MomentOfInertiaEstimation <pcl::PointXYZI> feature_extractor;
      feature_extractor.setInputCloud (cloud0);
      feature_extractor.compute ();

      std::vector <float> moment_of_inertia;
      std::vector <float> eccentricity;
      pcl::PointXYZI min_point_AABB;
      pcl::PointXYZI max_point_AABB;
      pcl::PointXYZI min_point_OBB;
      pcl::PointXYZI max_point_OBB;
      pcl::PointXYZI position_OBB;
      Eigen::Matrix3f rotational_matrix_OBB;
      float major_value, middle_value, minor_value;
      Eigen::Vector3f major_vector, middle_vector, minor_vector;
      Eigen::Vector3f mass_center;

      feature_extractor.getMomentOfInertia (moment_of_inertia);
      feature_extractor.getEccentricity (eccentricity);
      feature_extractor.getAABB (min_point_AABB, max_point_AABB);
      feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
      feature_extractor.getEigenValues (major_value, middle_value, minor_value);
      feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
      feature_extractor.getMassCenter (mass_center);

      pcl::PointXYZ point (mass_center (0), mass_center (1), mass_center (2));
      centers->push_back(point);

      markerx.header.frame_id = markery.header.frame_id = markerz.header.frame_id = cube.header.frame_id = "/velodyne";
      markerx.header.stamp = markery.header.stamp = markerz.header.stamp = cube.header.stamp = ros::Time::now();
      markerx.type = markery.type = markerz.type = visualization_msgs::Marker::LINE_LIST;
      cube.type = visualization_msgs::Marker::LINE_LIST;
      markerx.scale.x = markery.scale.x = markerz.scale.x = cube.scale.x = 0.05;
      markerx.scale.y = markery.scale.y = markerz.scale.y = cube.scale.y = 0.05;
      markerx.scale.z = markery.scale.z = markerz.scale.z = cube.scale.z = 0.05;
      markerx.color.a = markery.color.a = markerz.color.a = cube.color.a = 1;
      geometry_msgs::Point ps;
      ps.x = mass_center(0) ;
      ps.y = mass_center(1) ;
      ps.z = mass_center(2) ;
      geometry_msgs::Point px;
      px.x = major_vector(0) + mass_center(0);
      px.y = major_vector(1) + mass_center(1);
      px.z = major_vector(2) + mass_center(2);
      geometry_msgs::Point py;
      py.x = middle_vector(0) + mass_center(0);
      py.y = middle_vector(1) + mass_center(1);
      py.z = middle_vector(2) + mass_center(2);
      geometry_msgs::Point pz;
      pz.x = minor_vector(0) + mass_center(0);
      pz.y = minor_vector(1) + mass_center(1);
      pz.z = minor_vector(2) + mass_center(2);
      markerx.color.b = 0;
      markerx.color.g = 0;
      markerx.color.r = 1;
      markery.color.b = 0;
      markery.color.g = 1;
      markery.color.r = 0;
      markerz.color.b = 1;
      markerz.color.g = 0;
      markerz.color.r = 0;
      markerx.points.push_back(ps);
      markerx.points.push_back(px);
      markery.points.push_back(ps);
      markery.points.push_back(py);
      markerz.points.push_back(ps);
      markerz.points.push_back(pz);
      cube.color.b = 0;
      cube.color.g = 255;
      cube.color.r = 0;


      Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
      Eigen::Vector3f dis (max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z);
      Eigen::Quaternionf quat (rotational_matrix_OBB);
      Eigen::Vector3f corner = quat*(dis/2);

      Eigen::Vector3f p1 (min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
      Eigen::Vector3f p2 (min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
      Eigen::Vector3f p3 (max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
      Eigen::Vector3f p4 (max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
      Eigen::Vector3f p5 (min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
      Eigen::Vector3f p6 (min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
      Eigen::Vector3f p7 (max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
      Eigen::Vector3f p8 (max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
      p1 = rotational_matrix_OBB * p1 + position;
      p2 = rotational_matrix_OBB * p2 + position;
      p3 = rotational_matrix_OBB * p3 + position;
      p4 = rotational_matrix_OBB * p4 + position;
      p5 = rotational_matrix_OBB * p5 + position;
      p6 = rotational_matrix_OBB * p6 + position;
      p7 = rotational_matrix_OBB * p7 + position;
      p8 = rotational_matrix_OBB * p8 + position;
      //      (major_vector(0)*dis(0) + middle_vector(0)*dis(0) + minor_vector(0)*dis(0))

      geometry_msgs::Point e1;
      e1.x = p1 (0);
      e1.y = p1 (1);
      e1.z = p1 (2);
      geometry_msgs::Point e2;
      e2.x = p2 (0);
      e2.y = p2 (1);
      e2.z = p2 (2);
      geometry_msgs::Point e3;
      e3.x = p3 (0);
      e3.y = p3 (1);
      e3.z = p3 (2);
      geometry_msgs::Point e4;
      e4.x = p4 (0);
      e4.y = p4 (1);
      e4.z = p4 (2);
      geometry_msgs::Point e5;
      e5.x = p5 (0);
      e5.y = p5 (1);
      e5.z = p5 (2);
      geometry_msgs::Point e6;
      e6.x = p6 (0);
      e6.y = p6 (1);
      e6.z = p6 (2);
      geometry_msgs::Point e7;
      e7.x = p7 (0);
      e7.y = p7 (1);
      e7.z = p7 (2);
      geometry_msgs::Point e8;
      e8.x = p8 (0);
      e8.y = p8 (1);
      e8.z = p8 (2);

      cube.points.push_back(e1);
      cube.points.push_back(e2);
      cube.points.push_back(e1);
      cube.points.push_back(e4);
      cube.points.push_back(e1);
      cube.points.push_back(e5);
      cube.points.push_back(e5);
      cube.points.push_back(e6);
      cube.points.push_back(e5);
      cube.points.push_back(e8);
      cube.points.push_back(e2);
      cube.points.push_back(e6);
      cube.points.push_back(e6);
      cube.points.push_back(e7);
      cube.points.push_back(e7);
      cube.points.push_back(e8);
      cube.points.push_back(e2);
      cube.points.push_back(e3);
      cube.points.push_back(e4);
      cube.points.push_back(e8);
      cube.points.push_back(e3);
      cube.points.push_back(e4);
      cube.points.push_back(e3);
      cube.points.push_back(e7);
  }
  marker_pub1.publish(markerx);
  marker_pub2.publish(markery);
  marker_pub3.publish(markerz);
  cube_pub.publish(cube);

  sensor_msgs::PointCloud2 cluster_center;
  pcl::toROSMsg(*centers, cluster_center);
  cluster_center.header.frame_id = "velodyne";
  cloud_pub.publish(cluster_center);

  int j = 50;
  int k = 0;

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_clusters (new pcl::PointCloud<pcl::PointXYZI>);

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {

    // extract clusters and save as a single point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
      cloud_filtered->points[*pit].intensity = j;
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    }
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;
    *cloud_clusters += *cloud_cluster;
    j+=2;

  }

  sensor_msgs::PointCloud2 cluster_cloud;
  pcl::toROSMsg(*cloud_clusters, cluster_cloud);
  cluster_cloud.header.frame_id = "velodyne";
  cluster_pub.publish(cluster_cloud);

}


int main(int argc, char **argv) {
  ros::init(argc, argv, "pcl_cluster");
  ros::NodeHandle nh;

  plane_filtered_pub = nh.advertise<sensor_msgs::PointCloud2 >("plane_filtered_pub_points", 10);
  cluster_pub = nh.advertise<sensor_msgs::PointCloud2 >("cluster_cloud", 10);
  cloud_pub = nh.advertise<sensor_msgs::PointCloud2 >("cluster_center", 10);
  marker_pub1 = nh.advertise<visualization_msgs::Marker>("axisx", 1000);
  marker_pub2 = nh.advertise<visualization_msgs::Marker>("axisy", 1000);
  marker_pub3 = nh.advertise<visualization_msgs::Marker>("axisz", 1000);
  cube_pub = nh.advertise<visualization_msgs::Marker>("bounding_boxs", 1000);

  ros::Subscriber lidar_sub = nh.subscribe("points_raw", 10, lidar_callback);
 ROS_INFO("FFFFFF");
 initializeGlobalParams();
ROS_INFO("GGGGG");

  ros::Rate loop_rate(1);
  while (ros::ok()) {
    loop_rate.sleep();
     ROS_INFO("DDDD");
    ros::spinOnce();
  }
  ros::spin();
  return 0;
}


