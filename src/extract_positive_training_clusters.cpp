#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

// ROS messages
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// Custom messages
#include <leg_tracker/laser_processor.h>
#include <leg_tracker/cluster_features.h>

#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>

#include <boost/foreach.hpp>

#include <math.h>       /* atan2 */


#define PI 3.14159265

/**
* @brief Extract positive training scan clusters from a rosbag 
*
* Reads in a rosbag, finds all scan clusters lying within user-specified bounding box (or alternatively min/max angles), 
* marks those clusters as positive examples and saves the result to a new rosbag
* Used to quickly get positive examples for training the leg detector. 
*/
class ExtractPositiveTrainingClusters
{
public:

  /**
  * @brief Constructor
  */
  ExtractPositiveTrainingClusters() 
  {
    // Get ROS parameters  
    ros::NodeHandle nh_private("~"); // to get private parameters
    
    // Required parameters
    if (!nh_private.getParam("load_bag_file", load_bag_file_))
      ROS_ERROR("Couldn't get bag_load_file from ros param server");
    if (!nh_private.getParam("save_bag_file", save_bag_file_))
      ROS_ERROR("Couldn't get bag_save_file from ros param server");
    if (!nh_private.getParam("scan_topic", scan_topic_))
      ROS_ERROR("Couldn't get scan_topic from ros param server");
    if (!nh_private.getParam("laser_frame", laser_frame_))
      ROS_ERROR("Couldn't get laser_frame from ros param server");
    if (!nh_private.getParam("tf_data", tf_data_))
      ROS_ERROR("Couldn't get tf_data from ros param server");

      

    // Optional parameters (i.e., params with defaults)
    nh_private.param("cluster_dist_euclid", cluster_dist_euclid_, 0.13);
    nh_private.param("min_points_per_cluster", min_points_per_cluster_, 3);        
    nh_private.param("max_points_per_cluster", max_points_per_cluster_, 50);        

    // Optional parameters - either the x-y coordinates of a bounding box can be specified 
    // or the min/max angle and length for an arc
    
    if(nh_private.getParam("polygon_x", pol_x_) and
      nh_private.getParam("polygon_y", pol_y_)) 
    {
      bounding_type = 2;
      ROS_INFO("---------USING POLYGON------------");
      if(pol_x_.size()!=pol_y_.size() or pol_x_.size()<=3){
        ROS_ERROR("Polygon does not have same amount of x and y coordinates");
      }
    }
    else
    {
      bounding_type = 1;
    }
    if (!nh_private.getParam("x_min", x_min_) or
        !nh_private.getParam("x_max", x_max_) or
        !nh_private.getParam("y_min", y_min_) or
        !nh_private.getParam("y_max", y_max_))                        
    {
      ROS_INFO("Couldn't get bounding box for positive clusters. Assuming you've specified a min/max scan angle and a max distance or a polygon instead.");
      if(bounding_type==1){
        bounding_type = 0;
      }
    }
    if (!nh_private.getParam("min_angle", min_angle_) or
        !nh_private.getParam("max_angle", max_angle_) or
        !nh_private.getParam("max_dist", max_dist_))                       
    {
      if (bounding_type)
      {
        ROS_INFO("Couldn't get min/max scan angle for positive clusters. Assuming you've specified a bounding box instead.");
      }
      else
      {
        ROS_ERROR("Couldn't get bounding box or scan min/max angle for positive clusters");
      }
    }

    // Print back params:
    printf("\nROS parameters: \n");
    printf("cluster_dist_euclid:%.2fm \n", cluster_dist_euclid_);
    printf("min_points_per_cluster:%i \n", min_points_per_cluster_);
    printf("max_points_per_cluster:%i \n", max_points_per_cluster_);

    printf("--------------\n");
  }

  /**
  * @brief Extract the positive clusters and record their position with a leg_cluster_positions message
  */
  void extract()
  {
    printf("TRIES TO EXTRACT");
    // Open rosbag we'll be saving to
    rosbag::Bag save_bag;
    save_bag.open(save_bag_file_.c_str(), rosbag::bagmode::Write);

    // Open rosbag we'll be loading from
    rosbag::Bag load_bag;
    load_bag.open(load_bag_file_.c_str(), rosbag::bagmode::Read);
    
    // Iterate through all scan messages in the loaded rosbag
    std::vector<std::string> topics;

    topics.push_back(std::string(scan_topic_));
    rosbag::View view(load_bag, rosbag::TopicQuery(topics));
    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {
      if(m.getDataType()=="sensor_msgs/LaserScan")
      {

        sensor_msgs::LaserScan::ConstPtr scan = m.instantiate<sensor_msgs::LaserScan>();
        if (scan != NULL)
        {          
          // Positive scan points
          laser_processor::ScanProcessor pos_processor(*scan,1,pol_x_,pol_y_);          

          pos_processor.splitConnected(cluster_dist_euclid_);
          printf("Number of clusters %ld.\n",pos_processor.getClusters().size());

          auto small_clusters = pos_processor.removeLessThan(min_points_per_cluster_);
          printf("Processor size %ld.\n",pos_processor.getClusters().size());

          geometry_msgs::PoseArray leg_cluster_positions;
          leg_cluster_positions.header.frame_id = laser_frame_;
          std::vector<laser_processor::SampleSet*> pos_clusters;
          for (std::list<laser_processor::SampleSet*>::iterator i = pos_processor.getClusters().begin();
            i != pos_processor.getClusters().end();
            ++i)
          {

            // Only use scan clusters that are in the specified positive cluster area
            tf::Point cluster_position = (*i)->getPosition();

            double x_pos = cluster_position[0];
            double y_pos = cluster_position[1];
            double angle = atan2(y_pos,x_pos) * 180 / PI;
            double dist_abs = sqrt(x_pos*x_pos + y_pos*y_pos);

            bool in_polygon = bounding_type==2 and pnpoly(pol_y_.size(),pol_x_,pol_y_,x_pos,y_pos);
            if(bounding_type == 2){
              pos_clusters.push_back(*i);
            }
            else{
              bool in_bounding_box = bounding_type==1 and x_pos > x_min_ and x_pos < x_max_ and y_pos > y_min_ and y_pos < y_max_;
              bool in_arc = !bounding_type and angle > min_angle_ and angle < max_angle_ and dist_abs < max_dist_;
              if (in_bounding_box or in_arc or in_polygon) 
              {         
                geometry_msgs::Pose new_leg_cluster_position;
                new_leg_cluster_position.position.x = cluster_position[0];
                new_leg_cluster_position.position.y = cluster_position[1];
                leg_cluster_positions.poses.push_back(new_leg_cluster_position);
                pos_clusters.push_back(*i);
              }
            }
          }

          auto neg_scan = *scan;
          for(auto& cluster : pos_clusters){
            for(auto& sample : *cluster){
              neg_scan.ranges[sample->index] = neg_scan.range_max + 1;

            }
          }

          for(auto& cluster : small_clusters){
            for(auto& sample : *cluster){
              neg_scan.ranges[sample->index] = neg_scan.range_max + 1;
            }
          }

          save_bag.write("/neg_scan",m.getTime(),neg_scan);
          if (!leg_cluster_positions.poses.empty())  // at least one leg has been found in current scan
          {
            // Save position of leg to be used later for training 
            save_bag.write("/leg_cluster_positions", m.getTime(), leg_cluster_positions); 

            // Save scan
            save_bag.write("/training_scan", m.getTime(), *scan);

            // Save a marker of the position of the cluster we extracted. 
            // Just used so we can playback the rosbag file 
            // and visually verify the correct clusters have been extracted
            visualization_msgs::MarkerArray ma;
            for (int i = 0;
                i < leg_cluster_positions.poses.size();
                i++)
            {
              visualization_msgs::Marker m;
              m.header.frame_id = "laser_frame";
              m.ns = "LEGS";
              m.id = i;
              m.type = m.SPHERE;
              m.pose.position.x = leg_cluster_positions.poses[i].position.x;
              m.pose.position.y = leg_cluster_positions.poses[i].position.y;
              m.pose.position.z = 0.1;
              m.scale.x = .2;
              m.scale.y = .2;
              m.scale.z = .2;
              m.color.a = 1;
              m.lifetime = ros::Duration(0.2);
              m.color.b = 0.0;
              m.color.r = 1.0;
              ma.markers.push_back(m);
            }
            save_bag.write("/visualization_marker_array", m.getTime(), ma);
          }
        }
      }
      // else
      // {
      //   tf2_msgs::TFMessage tf_data = m.instantiate<tf2_msgs::TFMessage>();
      //   save_bag.write(m.getTopic(),m.getTime(),tf_data);
      // }
    }
    load_bag.close();
  }


  // adapted from https://wrfranklin.org/Research/Short_Notes/pnpoly.html
  int pnpoly(int nvert, std::vector<double> vertsx, std::vector<double> vertsy, float testx, float testy)
  {
  int i, j, c = 0;
  for (i = 0, j = nvert-1; i < nvert; j = i++) {
    if ( ((vertsy[i]>testy) != (vertsy[j]>testy)) &&
    (testx < (vertsx[j]-vertsx[i]) * (testy-vertsy[i]) / (vertsy[j]-vertsy[i]) + vertsx[i]) )
        c = !c;
  }
  if(c)
  {
    printf("\nTrue for point(%.2f,%.2f)",testx,testy);
  }
  else{
    printf("\nFalse for(%.2f,%.2f)",testx,testy);
  }
  return c;
  }


private:
  tf::TransformListener tfl_;
  std::string laser_frame_;
  std::string scan_topic_;

  ros::NodeHandle nh_;

  std::string save_bag_file_;
  std::string load_bag_file_;  
  std::string tf_data_;

  double cluster_dist_euclid_;
  int min_points_per_cluster_;
  int max_points_per_cluster_;

  // to describe bounding box containing positive clusters
  int bounding_type;

  std::vector<double> pol_x_;
  std::vector<double> pol_y_;
  
  double x_min_;
  double x_max_;
  double y_min_;
  double y_max_;
  
  // to describe scan angles containing positive clusters
  int min_angle_;
  int max_angle_;
  int max_dist_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv,"extract_positive_leg_clusters");
  ExtractPositiveTrainingClusters eptc;
  eptc.extract();
  ROS_INFO("Finished successfully! (you still have press ctrl+c to terminate if you ran from a launch file)"); 
  /** @todo Automatically terminate after finishing successfully */
  return 0;
}

