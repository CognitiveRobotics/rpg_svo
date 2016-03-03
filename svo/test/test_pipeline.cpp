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

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <vector>
#include <string>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/abstract_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/pinhole_camera.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <iostream>
#include "test_utils.h"
#include <dirent.h>

namespace svo {

class BenchmarkNode
{
  vk::AbstractCamera* cam_;
  svo::FrameHandlerMono* vo_;

public:
  BenchmarkNode();
  ~BenchmarkNode();
  void runFromFolder();
  std::fstream filename;

};

BenchmarkNode::BenchmarkNode()
{
  //cam_ = new vk::PinholeCamera(752, 480, 315.5, 315.5, 376.0, 240.0);
  cam_ = new vk::PinholeCamera(640, 480, 315.5, 315.5, 376.0, 240.0);
  vo_ = new svo::FrameHandlerMono(cam_);
  vo_->start();
  filename.open("/home/prateek/latestdata/ojflo_data/pose.txt",std::ofstream::out);
}

BenchmarkNode::~BenchmarkNode()
{
  delete vo_;
  delete cam_;
}

void BenchmarkNode::runFromFolder()
{
    char *dirName = "/home/prateek/latestdata/ojflo_data";
     DIR *dir;
     //dir = opendir(dirName.c_str());

     string imgName;
     struct dirent **ent;
       int img_id =0;
       int n = scandir(dirName,&ent,0,alphasort);
       std::cout<<"number of images"<<n<<std::endl;
       for(int i=4;i<n;i++)
       {
        imgName= ent[i]->d_name;
     //   std::cout<<imgName<<std::endl;
        //if(i >3)
           imgName= ent[i]->d_name;
         //   std::cout<<i<<" "<<imgName<<std::endl;

         cv::Mat img = cv::imread("/home/prateek/latestdata/ojflo_data/"+imgName,0);
       //  cv::imshow("img",img);
       //  cv::waitKey(10);
       // cv::Mat img;
      //   cv::cvtColor(img_new,img,CV_RGB2GRAY);
         //   cv::imshow("img",img);
      //   cv::waitKey(1000);
        std::cout<<"image Name "<<imgName<<std::endl;
         std::string::size_type sz;
         double time = i;


       // process frame
           vo_->addImage(img,time);

           // display tracking quality
           if(vo_->lastFrame() != NULL)
           {
               std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                         << "#Features: " << vo_->lastNumObservations() << " \t"
                         << "Proc. Time: " << vo_->lastProcessingTime()*1000 << "ms \t"
                         <<"Pose: "<<vo_->lastFrame()->T_f_w_.inverse() << "\n";

               Vector3d transl = vo_->lastFrame()->T_f_w_.inverse().translation();
               Matrix3d rot =  vo_->lastFrame()->T_f_w_.inverse().rotation_matrix();
               filename<<vo_->lastFrame()->id_<<" "<<transl(0)<<" "<<transl(1)<<" "<<transl(2)<<" "<<rot(0,0)<<" "<<rot(0,1)<<" "<<rot(0,2)<<" "<<rot(1,0)<<" "<<rot(1,1)<<" "<<rot(1,2)<<" "<<rot(2,0)<<" "<<rot(2,1)<<" "<<rot(2,2)<<std::endl;
               //access the pose of the camera via
           }
         }
           //img_id+=1;


    /*for(int img_id = 2; img_id < 188; ++img_id)
  {
    // load image
    std::stringstream ss;
    ss << svo::test_utils::getDatasetDir() << "/sin2_tex2_h1_v8_d/img/frame_"
       << std::setw( 6 ) << std::setfill( '0' ) << img_id << "_0.png";
    if(img_id == 2)
      std::cout << "reading image " << ss.str() << std::endl;
    cv::Mat img(cv::imread(ss.str().c_str(), 0));
    assert(!img.empty());

    // process frame
    vo_->addImage(img, 0.01*img_id);

    // display tracking quality
    if(vo_->lastFrame() != NULL)
    {
    	std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                  << "#Features: " << vo_->lastNumObservations() << " \t"
                  << "Proc. Time: " << vo_->lastProcessingTime()*1000 << "ms \n";



    	// access the pose of the camera via vo_->lastFrame()->T_f_w_.
    }





  }*/
}

} // namespace svo

int main(int argc, char** argv)
{
  {
    svo::BenchmarkNode benchmark;
    benchmark.runFromFolder();
  }
  printf("BenchmarkNode finished.\n");
  return 0;
}

