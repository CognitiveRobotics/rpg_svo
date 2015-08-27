// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <ros/package.h>
#include <string>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/config.h>
#include <svo_ros/visualizer.h>
#include <vikit/params_helper.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/String.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <vikit/abstract_camera.h>
#include <vikit/camera_loader.h>
#include <vikit/user_input_thread.h>
#include <object_tracking_2d_ros/ObjectDetections.h>

ros::Time time_first,time_second;

Sophus::SE3 pose1,pose2;

int flag =0 ;
int flagforframe =1;

namespace svo {

/// SVO Interface
class VoNode
{
public:
  svo::FrameHandlerMono* vo_;
  svo::Visualizer visualizer_;
  bool publish_markers_;                 //!< publish only the minimal amount of info (choice for embedded devices)
  bool publish_dense_input_;
  boost::shared_ptr<vk::UserInputThread> user_input_thread_;
  ros::Subscriber sub_remote_key_;
  std::string remote_input_;
  vk::AbstractCamera* cam_;
  bool quit_;
  VoNode();
  ~VoNode();
  void imgCb(const sensor_msgs::ImageConstPtr& msg);
  void processUserActions();
  void ebtmessageCallback(const object_tracking_2d_ros::ObjectDetections &msg);
  void remoteKeyCb(const std_msgs::StringConstPtr& key_input);
};

VoNode::VoNode() :
  vo_(NULL),
  publish_markers_(vk::getParam<bool>("svo/publish_markers", true)),
  publish_dense_input_(vk::getParam<bool>("svo/publish_dense_input", false)),
  remote_input_(""),
  cam_(NULL),
  quit_(false)
{
  // Start user input thread in parallel thread that listens to console keys
  if(vk::getParam<bool>("svo/accept_console_user_input", true))
    user_input_thread_ = boost::make_shared<vk::UserInputThread>();

  // Create Camera
  if(!vk::camera_loader::loadFromRosNs("svo", cam_))
    throw std::runtime_error("Camera model not correctly specified.");

  // Get initial position and orientation
  visualizer_.T_world_from_vision_ = Sophus::SE3(
      vk::rpy2dcm(Vector3d(vk::getParam<double>("svo/init_rx", 0.0),
                           vk::getParam<double>("svo/init_ry", 0.0),
                           vk::getParam<double>("svo/init_rz", 0.0))),
      Eigen::Vector3d(vk::getParam<double>("svo/init_tx", 0.0),
                      vk::getParam<double>("svo/init_ty", 0.0),
                      vk::getParam<double>("svo/init_tz", 0.0)));

  // Init VO and start
  vo_ = new svo::FrameHandlerMono(cam_);
  vo_->start();
}

VoNode::~VoNode()
{
  delete vo_;
  delete cam_;
  if(user_input_thread_ != NULL)
    user_input_thread_->stop();
}

void VoNode::imgCb(const sensor_msgs::ImageConstPtr& msg)
{
  cv::Mat img;
  try {
    img = cv_bridge::toCvShare(msg, "mono8")->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  processUserActions();
  //<<"Geting image"<<std::endl;
  vo_->addImage(img, msg->header.stamp.toSec());


  if((vo_->stage() == FrameHandlerMono::STAGE_SECOND_FRAME) && !flag)
  {    //<<"the second frame"<<msg->header.stamp<<std::endl;
       time_first = msg->header.stamp;
       flag =1;
  }

   //<<"flag"<<flag<<std::endl;

  if((vo_->stage() == FrameHandlerMono::STAGE_DEFAULT_FRAME) && flag )
  {    //<<"the default frame"<<msg->header.stamp.toSec()<<std::endl;
       time_second = msg->header.stamp;
    //    //<<"flag1"<<flag<<std::endl;
       flag =0;
  }

  visualizer_.publishMinimal(img, vo_->lastFrame(), *vo_, msg->header.stamp.toSec());
   visualizer_.visualizeMarkers(vo_->lastFrame(), vo_->coreKeyframes(), vo_->map());

  if(publish_markers_ && vo_->stage() != FrameHandlerBase::STAGE_PAUSED)
    visualizer_.visualizeMarkers(vo_->lastFrame(), vo_->coreKeyframes(), vo_->map());

  if(publish_dense_input_)
    visualizer_.exportToDense(vo_->lastFrame());

  if(vo_->stage() == FrameHandlerMono::STAGE_PAUSED)
    usleep(100000);
}

void VoNode::ebtmessageCallback(const object_tracking_2d_ros::ObjectDetections &msg)
{

     //<<"in  ebt msg  callback "<<msg.header.stamp.toSec()<<std::endl;



     for(unsigned int i = 0; i < msg.detections.size(); ++i)
      {

       //   detection.id = msg.detections[i].id;

         //<<msg.detections[i].ns<<std::endl;
        if( ! msg.detections[i].ns.compare("ronzoni1.obj") )
        {  // detection.ns = msg.detections[i].ns;
       //   detection.good = msg.detections[i].good;
       //   detection.init = msg.detections[i].init;



          geometry_msgs::Pose m = msg.detections[i].pose;
          Eigen::Affine3d e = Eigen::Translation3d(m.position.x,
                                                   m.position.y,
                                                   m.position.z) *
                  Eigen::Quaterniond(m.orientation.w,
                                     m.orientation.x,
                                     m.orientation.y,
                                     m.orientation.z);

          //<<"in  ebt msg "<<msg.header.stamp<<std::endl;
          Eigen::Matrix3d rot = e.rotation();
          Eigen::Vector3d trans = e.translation();

      //    std::cout<<time_first<<std::endl<<time_second<<std::endl;

          if(abs((time_first - msg.header.stamp).toSec()) < 0.3 && flagforframe)
          {
              std::cout<<"in second frame ebt"<<(time_first - msg.header.stamp).toSec()<<std::endl;
              std::cout<<time_first<<std::endl<<time_second<<std::endl;
              pose1 = SE3(rot,trans);
              flagforframe=1;
   //           //<<"flagforframe1"<<flagforframe<<std::endl;

          }

          if(abs((time_second -msg.header.stamp).toSec()) < 0.3  && flagforframe)
          {      pose2 = SE3(rot,trans);
                 std::cout<<"in default frame ebt "<<msg.header.stamp.toSec()<<std::endl;
                 const SE3 current_trans = pose2*pose1.inverse();
                 //<<current_trans<<"the value"<<std::endl;
          //       current_trans = SE3 (Matrix3d::Identity(), Vector3d::Zero());
          //       //<<current_trans<<"the value with identity"<<std::endl;
                 visualizer_.InitPose(current_trans,vo_->lastFrame());
                 flagforframe=0;
                 //<<"flagforframe2"<<current_trans<<std::endl;

          }


      }

}


}

void VoNode::processUserActions()
{
  char input = remote_input_.c_str()[0];
  remote_input_ = "";

  if(user_input_thread_ != NULL)
  {
    char console_input = user_input_thread_->getInput();
    if(console_input != 0)
      input = console_input;
  }

  switch(input)
  {
    case 'q':
      quit_ = true;
      printf("SVO user input: QUIT\n");
      break;
    case 'r':
      vo_->reset();
      printf("SVO user input: RESET\n");
      break;
    case 's':
      vo_->start();
      printf("SVO user input: START\n");
      break;
    default: ;
  }
}

void VoNode::remoteKeyCb(const std_msgs::StringConstPtr& key_input)
{
  remote_input_ = key_input->data;
}

} // namespace svo

int main(int argc, char **argv)
{
  ros::init(argc, argv, "svo");
  ros::NodeHandle nh;
  // << "create vo_node" << std::endl;
  svo::VoNode vo_node;

  // subscribe to cam msgs
  std::string cam_topic(vk::getParam<std::string>("svo/cam_topic", "camera/image_raw"));
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber it_sub = it.subscribe(cam_topic, 5, &svo::VoNode::imgCb, &vo_node);

  // subscribe to remote input
  vo_node.sub_remote_key_ = nh.subscribe("svo/remote_key", 5, &svo::VoNode::remoteKeyCb, &vo_node);

  //subscribe to ebt msgs
  ros::Subscriber ebtmessage_sub_ = nh.subscribe ("/object_tracking_2d_ros/detections", 1, &svo::VoNode::ebtmessageCallback, &vo_node);


  // start processing callbacks
  while(ros::ok() && !vo_node.quit_)
  {
    ros::spinOnce();
    // TODO check when last image was processed. when too long ago. publish warning that no msgs are received!
  }

  printf("SVO terminated.\n");
  return 0;
}
