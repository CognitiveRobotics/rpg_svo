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
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>
#include "epnp.h"


#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"


#include <string>
#include <vector>




extern int keyframe_num =0;

namespace svo {
namespace initialization {

InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  reset();
  detectFeatures(frame_ref, px_ref_, f_ref_);

 /* string path = "/home/prateek/catkin_ws/src/object_tracking_2d_ros/data/ronzoni1";
  string path_cr= "/home/prateek/catkin_ws/src/object_tracking_2d_ros/data/crayola_64_ct";
  string path_or= "/home/prateek/catkin_ws/src/object_tracking_2d_ros/data/orange_juice_carton_flo";
  string path_ss = "/home/prateek/catkin_ws/src/object_tracking_2d_ros/data/soft_scrub";
  string path_td = "/home/prateek/catkin_ws/src/object_tracking_2d_ros/data/tide";*/



  string path = Config::objectname();

  std::cout<<"Frame number"<<frame_ref->id_<<std::endl;

  std::cout<<"loading keyframes"<<std::endl;
  loadKeyframes(path,1);
 // loadKeyframes(path_cr,2);
 // loadKeyframes(path_or,3);
  std::cout<<"loaded keyframes"<<std::endl;
  buildKdTree(0);
 // buildKdTree(1);
 // buildKdTree(2);
  int num_corr = 6;
  int num_corr1 = 0;
  int num_corr2 = 0;
  SE3 pose1 = estimatePose(num_corr,frame_ref,0);
  SE3 pose2,pose3;
//  pose2 = estimatePose(num_corr1,frame_ref,1);
 // pose3 = estimatePose(num_corr2,frame_ref,2);

  if(num_corr>=num_corr1 ){
     if(num_corr >= num_corr2)
         F_pose = pose1;
     else
         F_pose = pose3;
  }
  else{
      if(num_corr1 >= num_corr2)
          F_pose = pose2;
      else
          F_pose = pose3;

  }
 // std::cout<<F_pose<<std::endl;


  if(px_ref_.size() < 100)
  {
    SVO_INFO_STREAM("First image has less than 100 features. Retry in more textured environment."<<px_ref_.size());
    return FAILURE;
  }
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
  return SUCCESS;
}


Vector3d KltHomographyInit::triangulateFeatureNonLin1(const Matrix3d& R,  const Vector3d& t,
                         const Vector3d& feature1, const Vector3d& feature2 )
{
 // //<<"in triangulation1"<<std::endl;
  Vector3d f2 = R * feature2;
//  //<<"in triangulation2"<<std::endl;
  Vector2d b;
  b[0] = t.dot(feature1);
  b[1] = t.dot(f2);
 // //<<"in triangulation3"<<std::endl;
  Matrix2d A;
  A(0,0) = feature1.dot(feature1);
  A(1,0) = feature1.dot(f2);
  A(0,1) = -A(1,0);
  A(1,1) = -f2.dot(f2);
//  //<<"in triangulation4"<<std::endl;
  Vector2d lambda = A.inverse() * b;
  Vector3d xm = lambda[0] * feature1;
  Vector3d xn = t + lambda[1] * f2;
//  //<<"in triangulation5"<<std::endl;
  return ( xm + xn )/2;

}



double KltHomographyInit::computeInliers1(const vector<Vector3d>& features1, // c1
               const vector<Vector3d>& features2, // c2
               const Matrix3d& R,                 // R_c1_c2
               const Vector3d& t,                 // c1_t
               const double reproj_thresh,
               double error_multiplier2,
               vector<Vector3d>& xyz_vec,         // in frame c1
               vector<int>& inliers,
               vector<int>& outliers)
{
  //<<"triangulate all features and compute reprojection errors and inliers"<<std::endl;
  inliers.clear(); inliers.reserve(features1.size());
  outliers.clear(); outliers.reserve(features1.size());
  xyz_vec.clear(); xyz_vec.reserve(features1.size());
  double tot_error = 0;
  //<<"triangulate all features and compute reprojection errors and inliers"<<std::endl;
  for(size_t j=0; j<features1.size(); ++j)
  {
    xyz_vec.push_back(triangulateFeatureNonLin1(R, t, features1[j], features2[j] ));
    double e1 = vk::reprojError(features1[j], xyz_vec.back(), error_multiplier2);
    double e2 = vk::reprojError(features2[j], R.transpose()*(xyz_vec.back()-t), error_multiplier2);
    if(e1 > reproj_thresh || e2 > reproj_thresh)
      outliers.push_back(j);
    else
    {
      inliers.push_back(j);
      tot_error += e1+e2;
    }
    //<<"triangulate all features and compute reprojection errors and inliers"<<j<<std::endl;
  }
  return tot_error;

}


InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");
  keyframe_num++;

  if(disparities_.size() < Config::initMinTracked())
    return FAILURE;

  double disparity = vk::getMedian(disparities_);
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;
 /* else
  {
     return SUCCESS;
  }//*/


  input_keypoints_2d_.clear();

  int num_corr = 6;
  int num_corr1 = 0;
  int num_corr2 = 0;
  SE3 pose1 = estimatePose(num_corr,frame_cur,0);
  SE3 pose2,pose3;
  //    pose2 = estimatePose(num_corr1,frame_cur,1);
 // SE3 pose3 = estimatePose(num_corr2,frame_cur,2);
  SE3 pose;
  if(num_corr>=num_corr1 ){
     if(num_corr >= num_corr2)
         pose = pose1;
     else
         pose = pose3;
  }
  else{
      if(num_corr1 >= num_corr2)
         pose = pose2;
      else
         pose = pose3;

  }
 // std::cout<<pose<<std::endl;

 // int num_corr = 6;
 // SE3 pose = estimatePose(num_corr,frame_cur);
  //pose.rotation_matrix()
  //std::cout<<"in second frme"<<pose.rotation_matrix()<<"   "<<pose.translation() <<"  first frame "<<F_pose.rotation_matrix()<<" "<<F_pose.translation()<<std::endl;


  T_cur_from_ref_ = pose*F_pose.inverse();
  //std::cout<<"whole pose old"<<T_cur_from_ref_.translation()<<std::endl;
  //std::cout<<"whole pose diff"<<(pose.inverse()*F_pose).translation()<<std::endl;

  //T_cur_from_ref_ = T_cur_from_ref_.inverse();
 // std::cout<<"whole pose"<<T_cur_from_ref_.rotation_matrix()<<"  "<<T_cur_from_ref_.translation()<<std::endl;

  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);
  SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");
   //std::cout<<"whole pose without scale"<<T_cur_from_ref_<<std::endl;
  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  vector<double> depth_vec;
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  double scene_depth_median = vk::getMedian(depth_vec);
  //std::cout<<"scene depth "<<scene_depth_median<<"and keyframe number"<<keyframe_num<<"number of points"<<xyz_in_cur_.size()<<std::endl;
  double scale = Config::mapScale()/scene_depth_median;
  scale =1;
  //scale = 12;
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
 // frame_cur->T_f_w_.translation() =
  //    -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));
 //<<frame_cur->T_f_w_.translation()<<std::endl;

  // For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();
  //std::cout<<"The second frame is "<<T_world_cur<<std::endl;
  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
    {
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
      Point* new_point = new Point(pos);

      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
      frame_cur->addFeature(ftr_cur);
      new_point->addFrameRef(ftr_cur);

      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
      frame_ref_->addFeature(ftr_ref);     
      new_point->addFrameRef(ftr_ref);
    //  std::cout<<"actual point"<<new_point->pos_<<std::endl;
    }
  }
  return SUCCESS;
}


InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur,SE3 pose)
{
  //<<"klt with pose"<<std::endl;
 // trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");
  //<<"Init: KLT tracked "<< disparities_.size() <<" features"<<std::endl;
  keyframe_num++;

  if(disparities_.size() < Config::initMinTracked())
    return FAILURE;

  double disparity = vk::getMedian(disparities_);
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;

  //<<"Value after disparity"<<std::endl;
  T_cur_from_ref_ = pose;
//<<"Value after disparity"<<f_ref_.size()<<f_cur_.size()<<std::endl;


for(size_t i=0, i_max=f_ref_.size(); i<i_max; ++i)
{
//  uv_ref[i] = vk::project2d(f_ref_[i]);
//  uv_cur[i] = vk::project2d(f_cur_[i]);
    ////<<"in triangualtion"<<T_cur_from_ref_.translation()<<" "<<f_ref_[i]<<endl;

     Vector3d point =  triangulateFeatureNonLin1(T_cur_from_ref_.rotation_matrix(),T_cur_from_ref_.translation(), f_cur_[i],f_ref_[i]);
   //  //<<"points"<<point<<std::endl;
     xyz_in_cur_.push_back(point);
     inliers_.push_back(i);
}

//<<"after triaangualtion"<<xyz_in_cur_.size()<<T_cur_from_ref_.rotation_matrix()<<" "<<T_cur_from_ref_.translation()<<std::endl;
vector<int> outliers;

/*computeInliers1(f_cur_, f_ref_,
                     T_cur_from_ref_.rotation_matrix(), T_cur_from_ref_.translation(),
                     Config::poseOptimThresh(), frame_ref_->cam_->errorMultiplier2(),
                     xyz_in_cur_, inliers_, outliers);

  /*computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);*/

  //<<"Init: Homography RANSAC "<<inliers_.size()<<" inliers."<<std::endl;

  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  vector<double> depth_vec;
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  double scene_depth_median = vk::getMedian(depth_vec);
  //<<"scene depth new"<<scene_depth_median<<"and keyframe number new"<<keyframe_num<<std::endl;
  double scale = Config::mapScale()/scene_depth_median;
  //scale = 12
  //<<pose<<std::endl;
  //frame_cur->T_f_w_ = pose;
 // pose.translation() << 0,0,0;
 // frame_cur->T_f_w_.translation() = pose.translation();
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
  //<<"poses"<<T_cur_from_ref_ <<frame_ref_->T_f_w_<< "  "<<frame_cur->T_f_w_<<std::endl;
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));
 //<<frame_cur->T_f_w_.translation()<<std::endl;

  // For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();
  std::cout<<"The workd current "<<T_world_cur<<std::endl;

  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
    {
    // //<<"Pose"<<T_world_cur<<xyz_in_cur_[*it]<<std::endl;
     // T_world_cur(Matrix3d::Identity(), Vector3d::Zero());
  //    Matrix4d transform ;
  //    transform.topLeftCorner(3,3)= Matrix4d::Identity(3,3);
  //    transform.topRightCorner(1,)

      Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);

   //   //<<"3d points"<<pos<<std::endl;
      Point* new_point = new Point(pos);

      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
      frame_cur->addFeature(ftr_cur);
      new_point->addFrameRef(ftr_cur);

      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
      frame_ref_->addFeature(ftr_ref);
      new_point->addFrameRef(ftr_ref);
    }
  }
  return SUCCESS;
}



void KltHomographyInit::reset()
{
    px_cur_.clear();
    frame_ref_.reset();
    inliers_.clear();
    keyframe_keypoints_2d_.clear();
    keyframe_keypoints_3d_.clear();
    keyframe_descriptors_.clear();
    keyframe_images_.clear();
    input_keypoints_2d_.clear();
    keypoints2D.clear();
    keypoints3D.clear();

    kfd_.clear();
    keyframe_lut_.clear();
}


void KltHomographyInit::loadKeyframes(string obj_name,int num)
{
  // Load keyframes: count the number of keyframes, then dynamically load jpeg images and pose parameters (xml)

  // release the previous data



  /*for(int i=0; i<num_keyframes_; i++)
  {
    cvReleaseImage(&keyframe_images_[i]);
    cvReleaseMat(&keyframe_poses_[i]);
    cvReleaseMat(&keyframe_keypoints_2d_[i]);
    cvReleaseMat(&keyframe_keypoints_3d_[i]);
    cvReleaseMat(&keyframe_descriptors_[i]);
  }*/

  int num_keyframes_ = 0;
  std::string data_dir =  obj_name;
  fstream fsk;
  fsk.open((data_dir + "/" + "keyframe001.jpg").c_str());
  // count the number of keyframes
  while(fsk.is_open())
  {
    num_keyframes_++;
    fsk.close();
    std::stringstream ss;
    ss << data_dir << "/" << "keyframe" << std::setw(3) << std::setfill('0') << num_keyframes_ + 1 << ".jpg";
    fsk.open(ss.str().c_str());
  }

  //std::cout<<"(AKAN) Num of Keyframes: "<<num_keyframes_<<std::endl;

  // Read keyframes - jpg & xml files
  //keyframe_images_.resize(num_keyframes_);
  //keyframe_poses_.resize(num_keyframes_);
  std::vector<IplImage*> keyframe_images;
  std::vector<CvMat*> keyframe_keypoints_2d;
  std::vector<CvMat*> keyframe_keypoints_3d;
  std::vector<CvMat*> keyframe_descriptors;


  keyframe_keypoints_2d.resize(num_keyframes_);
  keyframe_keypoints_3d.resize(num_keyframes_);
  keyframe_descriptors.resize(num_keyframes_);
  keyframe_images.resize(num_keyframes_);


  std::stringstream ss;
  for(int i=0; i<num_keyframes_; i++)
  {
    ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "keyframe" << std::setw(3) << std::setfill('0') << i + 1 << ".jpg";
  //  keyframe_images[i] = cv::imread(ss.str().c_str());
    keyframe_images[i] = cvLoadImage(ss.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    assert(keyframe_images[i]);

  /*  ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "pose" << std::setw(3) << std::setfill('0') << i + 1 << ".xml";
    keyframe_poses_[i] = (CvMat*)cvLoad(ss.str().c_str());
    assert(keyframe_poses_[i]);*/

    ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "keypoint2d" << std::setw(3) << std::setfill('0') << i + 1 << ".xml";
    std::cout<<ss.str().c_str()<<std::endl;
    keyframe_keypoints_2d[i] = (CvMat*)cvLoad(ss.str().c_str());

    assert(keyframe_keypoints_2d[i]);

    ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "keypoint3d" << std::setw(3) << std::setfill('0') << i + 1 << ".xml";
    keyframe_keypoints_3d[i] = (CvMat*)cvLoad(ss.str().c_str());
    assert(keyframe_keypoints_3d[i]);

    ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "descriptor" << std::setw(3) << std::setfill('0') << i + 1 << ".xml";
    keyframe_descriptors[i] = (CvMat*)cvLoad(ss.str().c_str());
    assert(keyframe_descriptors[i]);
  }
  keyframe_keypoints_2d_.push_back(keyframe_keypoints_2d);
  keyframe_keypoints_3d_.push_back(keyframe_keypoints_3d);
  keyframe_descriptors_.push_back(keyframe_descriptors);
  keyframe_images_.push_back(keyframe_images);


}

void KltHomographyInit::buildKdTree(int num)
{
  if(keyframe_keypoints_2d_[num].size() == 0 || keyframe_keypoints_3d_[num].size() == 0 || keyframe_descriptors_[num].size() == 0)
    return;

  std::cout<<"the number of keyfrmaes are "<<num<<" "<<keyframe_keypoints_2d_.size()<<"numbers"<<keyframe_keypoints_3d_[num].size()<<" "<<keyframe_descriptors_[num].size()<<" "<<keyframe_keypoints_2d_[num].size()<<std::endl;
  //assert(keyframe.size() == keypoint2D.size());
  assert(keyframe_keypoints_2d_[num].size() == keyframe_keypoints_3d_[num].size());
  assert(keyframe_keypoints_3d_[num].size() == keyframe_descriptors_[num].size());

 // keyframe_images_ = keyframe;

  // Save keyframe descriptor into CvMat
  int dims = keyframe_descriptors_[num][0]->cols;
  int row = 0;
  for(int i=0; i < keyframe_keypoints_2d_[num].size(); i++)
  {
    row += keyframe_keypoints_2d_[num][i]->rows;
  }


  vector<int> keyframe_lut;
  vector<CvPoint2D32f> keypoints2D_;
  vector<CvPoint3D32f> keypoints3D_;

  keyframe_lut.resize(row);
  keypoints2D_.resize(row);
  keypoints3D_.resize(row);

  cv::Mat kfd(row, dims, CV_32F);
  cv::Mat labels,clusters;
  int k = 0;

  for(int h = 0; h < keyframe_keypoints_2d_[num].size(); h++)
  {

      int l=0;
      for(int i = 0; i < keyframe_keypoints_2d_[num][h]->rows; i++ )
      {

          keypoints2D_[l+k] = CV_MAT_ELEM(*keyframe_keypoints_2d_[num][h], CvPoint2D32f, i, 0);
          keypoints3D_[l+k] = CV_MAT_ELEM(*keyframe_keypoints_3d_[num][h], CvPoint3D32f, i, 0);
          keyframe_lut[l+k] = h;
          for(int j=0; j<dims; j++)
            kfd.at<float>(l+k, j) = CV_MAT_ELEM(*keyframe_descriptors_[num][h], float, i, j);
          l+=1;

        }

     k += keyframe_keypoints_2d_[num][h]->rows;
   }
     std::cout<<"number of 0 labels"<<kfd.rows<<std::endl;

   kfd_.push_back(kfd);
   keypoints2D.push_back(keypoints2D_);
   keypoints3D.push_back(keypoints3D_);
   keyframe_lut_.push_back(keyframe_lut);
}

SE3 KltHomographyInit::estimatePose(int &num_of_corr,FramePtr frame_ref,int num)
{
  // Using keyframes + EPnP RANSAC

  // Clear earlier storage and sequence
  /*if(seq_keypoints_)    cvClearSeq(seq_keypoints_);
  if(seq_descriptors_)  cvClearSeq(seq_descriptors_);
  cvClearMemStorage(ms_);*/

  CvMemStorage* ms_ = cvCreateMemStorage(0);
  CvSeq *seq_keypoints_ = NULL;
  CvSeq *seq_descriptors_ =  NULL;
  CvSURFParams surf_params_ = cvSURFParams(200, 3);;
  vector<int> corr_;

  IplImage* img = new IplImage(frame_ref->img());

  // Extract SURF features on test images.
  cvExtractSURF(img, 0, &seq_keypoints_, &seq_descriptors_, ms_, surf_params_);
  // Find the initial correspondence with Nearest Neighborhood
  std::cout<<"here after extraction"<<std::endl;


  int keyframe_idx_ = findCorrespondenceNN_FLANN(seq_keypoints_, seq_descriptors_, corr_/*corr1_*/, keyframe_keypoints_2d_[num].size(),num);
  std::cout<<"size of pt pairs with corr"<<corr_.size()<<std::endl;

  vector<CvPoint2D32f> outliers_obj_2d_ ; //objOutliers;
  vector<CvPoint2D32f> outliers_img_2d_ ; //imgOutliers;
  vector<CvPoint2D32f> inliers_obj_2d_ ; //objInliers;
  vector<CvPoint2D32f> inliers_img_2d_ ; //imgInliers;
  vector<CvPoint3D32f> outliers_obj_3d_  ; //objOutliers3D;
  vector<CvPoint3D32f>  inliers_obj_3d_ ; //imgOutliers3D;

  //outliers_obj_2d_.resize(0);



  SE3 pose;
  num_of_corr = refineCorrespondenceEpnpRANSAC(corr_/*corr1_[i]*/, frame_ref,outliers_obj_2d_, outliers_obj_3d_, outliers_img_2d_, inliers_obj_2d_, inliers_obj_3d_, inliers_img_2d_, pose,num);


  std::cout<<"Number of corr"<<num_of_corr<<std::endl;

  //SE3 pose;
  //refineCorrespondenceEpnpRANSAC(corr_,frame_ref,pose,num);
  //num_of_corr = corr_.size();

  cvClearSeq(seq_keypoints_);
  cvClearSeq(seq_descriptors_);
  cvClearMemStorage(ms_);

 // return pose;

  // Refine the correspondences with RANSAC and estimate pose
 // for(int i=0;i<1;i++)
//  {
//  num_of_corr = refineCorrespondenceEpnpRANSAC(corr_/*corr1_[i]*/, outliers_obj_2d_, outliers_obj_3d_, outliers_img_2d_, inliers_obj_2d_, inliers_obj_3d_, inliers_img_2d_, pose_);

  // Copy image and object images
 //  std::cout<<poset<<"here in the estimation"<<std::endl;


 // } // Display inliers/outliers

  // If the number of correspondence is smaller than 4, finish
  /*if(num_of_corr < 4)
  {
    printf("Insufficient matches...(%d)\n", num_of_corr);
    num_of_corr = 0;
    cvSetIdentity(pose);
    return pose_;
  }*/

 // IplImage* img_result_ = new IplImage(frame_ref->img());
  IplImage* img_object_ = new IplImage(frame_ref->img());
  IplImage* img_result_ = cvCreateImage(cvSize(img_object_->width*2, img_object_->height), 8, 3);


  cvSetImageROI(img_result_, cvRect(0, 0, img_object_->width, img_object_->height));
  cvCvtColor(keyframe_images_[0][keyframe_idx_], img_result_, CV_GRAY2BGR );
  cvSetImageROI(img_result_, cvRect( img_object_->width, 0, img_object_->width, img_object_->height ) );
  cvCvtColor(img_object_, img_result_, CV_GRAY2BGR );
  cvResetImageROI(img_result_);
  //cv::Mat poset(pose_);
//  std::cout<<poset<<"here in the estimation"<<std::endl;//*/

 /* for(int i=0; i<int(inliers_obj_2d_.size()); i++)
    cvLine(img_result_, cvPointFrom32f(inliers_obj_2d_[i]),
    cvPoint(cvRound(inliers_img_2d_[i].x+img_object_->width), cvRound(inliers_img_2d_[i].y)), CV_RGB(0,255,0), 1, CV_AA, 0);

  for(int i=0; i<int(outliers_obj_2d_.size()); i++)
    cvLine(img_result_, cvPointFrom32f(outliers_obj_2d_[i]),
    cvPoint(cvRound(outliers_img_2d_[i].x+img_object_->width), cvRound(outliers_img_2d_[i].y)), CV_RGB(255,100,100), 1, CV_AA, 0); //*/

  // Draw object
 // obj_model_->displayPoseLine(img_result_, pose_, CV_RGB(255, 255, 0), 1, true); // on the right side

  //  cvShowImage("Initilization", img_result_);
//    cvWaitKey(0.1);





  return pose;//*/
}



int KltHomographyInit::findCorrespondenceNN_FLANN(const CvSeq* imageKeypoints, const CvSeq* imageDescriptors, vector<int>& ptpairs/*vector< vector<int> >& ptpairs1*/, int numOfKeyframes, int num)
{

  if(imageDescriptors->total == 0)
  {
    ptpairs.clear();
    return 0;
  }
 // vector<int> ptpairs;
  vector<CvPoint2D32f> input_keypoints_2d;
  CvSeqReader kreader, reader;
  int dims = imageDescriptors->elem_size/sizeof(float);
  cv::Mat id(imageDescriptors->total, dims, CV_32F); // imageDescriptors
  input_keypoints_2d.resize(imageDescriptors->total);
  std::cout<<"in the refinement"<<imageDescriptors->total<<std::endl;

  // save image descriptor into CvMat
  cvStartReadSeq( imageKeypoints, &kreader );
  cvStartReadSeq( imageDescriptors, &reader );
  for(int i = 0; i < imageDescriptors->total; i++ )
  {
    const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
    const float* descriptor = (const float*)reader.ptr;
    CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
    CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
    input_keypoints_2d[i] = (*kp).pt;
    for(int j=0; j<dims; j++)
      id.at<float>(i, j) = descriptor[j];
  }
  input_keypoints_2d_.push_back(input_keypoints_2d);

  std::cout<<"here"<<std::endl;


  double th_ratio = 0.6;      // ratio test threshold
  cv::Mat indices(kfd_[num].rows, 2, CV_32S);
  cv::Mat dists(kfd_[num].rows, 2, CV_32F);

  cv::flann::Index flann_index(id, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
  flann_index.knnSearch(kfd_[num], indices, dists, 2, cv::flann::SearchParams(64)); // maximum number of leafs
  std::cout<<"point pair size"<<indices.rows<<std::endl;
  ptpairs.clear();
  vector<int> idxpairs;

  // check results and save to 'ptpairs'
  int* indices_ptr = indices.ptr<int>(0);
  float* dists_ptr = dists.ptr<float>(0);

  for (int i=0; i<indices.rows; ++i)
  {
    if (dists_ptr[2*i] < th_ratio*dists_ptr[2*i+1])
    {
      ptpairs.push_back(i); // image index
      ptpairs.push_back(indices_ptr[2*i]); // object index
      idxpairs.push_back(keyframe_lut_[num][indices_ptr[2*i]]);
    }
  }

 // std::cout<<"point pair size"<<ptpairs.size()<<std::endl;
  vector<int> cnt_buf;
  cnt_buf.resize(numOfKeyframes);

  for(int i=0; i<idxpairs.size(); i++)
  {
      if(idxpairs[i]<cnt_buf.size() && idxpairs[i] > -1)
    cnt_buf[idxpairs[i]]++;
  }

  int keyframe_idx_ = 0; // default
  int max_cnt = 0;
  for(int i=0; i<numOfKeyframes; i++)
  {
    if(max_cnt < cnt_buf[i])
    {
      max_cnt = cnt_buf[i];
      keyframe_idx_ = i;
    }
  }

  assert(keyframe_idx_ >= 0);
  printf("Matched keyframe: %d\n", keyframe_idx_);
  vector<int> ptpairs_new;
  for(int i=0; i<idxpairs.size(); i++)
  {
    if(idxpairs[i] == keyframe_idx_)
    {
      // check adding correspondence is already saved
      bool added = false;
      for(int j=0; j<int(ptpairs_new.size()/2); j++)
      {
        if(ptpairs_new[2*j] == ptpairs[2*i])
        {
          added = true;
          break;
        }
      }

      if(!added)
      {
        ptpairs_new.push_back(ptpairs[2*i]);
        ptpairs_new.push_back(ptpairs[2*i+1]);
      }
    }
  }

  ptpairs = ptpairs_new;

  return keyframe_idx_;
//  std::cout<<ptpairs.size()<<std::endl;
//  ptpairs1.push_back(ptpairs);
  //}
}





int KltHomographyInit::refineCorrespondenceEpnpRANSAC(const vector<int>& ptpairs, FramePtr frame_ref,vector<CvPoint2D32f> &objOutliers,vector<CvPoint3D32f> &objOutliers3D, vector<CvPoint2D32f> &imgOutliers, vector<CvPoint2D32f> &objInliers, vector<CvPoint3D32f> &objInliers3D, vector<CvPoint2D32f> &imgInliers, SE3 &pose,int num)
{
  const int NOM = 7; // number of model parameters

  int n;
  int iter = 0;
  int k = 100000;
  const int max_k = 50;
  int best_noi = 0;
  const float th = 5;
  const double p = 0.99;
    std::cout<<" in refinement"<<ptpairs.size()/2<<std::endl;
  n = int(ptpairs.size()/2);
  if( n < 8 ) // at least 8 points are needed to estimate fundamental matrix
    return -1;




  objOutliers.resize(n);
  imgOutliers.resize(n);
  objOutliers3D.resize(n);
  for(int i = 0; i < n; i++ )
  {
    objOutliers[i] = keypoints2D[num][ptpairs[i*2]];
    imgOutliers[i] = input_keypoints_2d_[num][ptpairs[i*2+1]];
    objOutliers3D[i] = keypoints3D[num][ptpairs[i*2]];
  }

 // std::cout<<"Outputting the intrinsics 123"<<frame_ref->cam_->errorMultiplier2()<<std::endl;

  epnp ePnP;

  float fu = 541.7084706800928;    //frame_ref->cam_->errorMultiplier2() ; //CV_MAT_ELEM(*intrinsic_, float, 0, 0);
  float fv = 539.9289409636573;  //frame_ref->cam_->errorMultiplier2() ; //CV_MAT_ELEM(*intrinsic_, float, 1, 1);
  float uc = 312.6945803266324; //(frame_ref->cam_->width()/2) ; //CV_MAT_ELEM(*intrinsic_, float, 0, 2);
  float vc = 241.325960768061;  //(frame_ref->cam_->height()/2); //fabs(frame_ref->cam_->cy());  //CV_MAT_ELEM(*intrinsic_, float, 1, 2);

  vector<int> inlier_idx;
  inlier_idx.resize(n);
  vector<int> best_inlier_idx;
  best_inlier_idx.resize(n);

  ePnP.set_internal_parameters(uc, vc, fu, fv);
  ePnP.set_maximum_number_of_correspondences(NOM);
 // std::cout<<"Outputting the intrinsics 456 "<<frame_ref->cam_->errorMultiplier2()<<std::endl;

  CvRNG rng = cvRNG(cvGetTickCount());
  int rand_idx[NOM];
  CvMat* P = cvCreateMat(3,4,CV_32F);
  CvMat* P2 = cvCreateMat(3,4,CV_32F);
  CvMat* x3d_h = cvCreateMat(4, n, CV_32F);
  CvMat* x2d_proj = cvCreateMat(3, n, CV_32F);

 // std::cout<<"Outputting the intrinsics 789"<<frame_ref->cam_->errorMultiplier2()<<std::endl;

  for(int i=0; i<n; i++)
  {
    CV_MAT_ELEM(*x3d_h, float, 0, i) = objOutliers3D[i].x;
    CV_MAT_ELEM(*x3d_h, float, 1, i) = objOutliers3D[i].y;
    CV_MAT_ELEM(*x3d_h, float, 2, i) = objOutliers3D[i].z;
    CV_MAT_ELEM(*x3d_h, float, 3, i) = 1.0;
  }

  double R_est[3][3], T_est[3];

  //std::cout<<"Outputting the intrinsics 546457"<<frame_ref->cam_->errorMultiplier2()<<std::endl;

  /*ePnP.reset_correspondences();
  for(int i=0; i<NOM; i++)
  {
    ePnP.add_correspondence(objOutliers3D[i].x, objOutliers3D[i].y, objOutliers3D[i].z, imgOutliers[i].x, imgOutliers[i].y);
  }
  double err = ePnP.compute_pose(R_est, T_est);

  std::cout<<R_est[0][0]<<T_est[0]<<std::endl;*/



  //return err;


  CvMat* intrinsic_ = cvCreateMat(3,3,CV_32F);

  CV_MAT_ELEM(*intrinsic_, float, 0, 2) = 312.6945803266324;
  CV_MAT_ELEM(*intrinsic_, float, 1, 2) = 241.325960768061;
  CV_MAT_ELEM(*intrinsic_, float, 0, 0) = fu;
  CV_MAT_ELEM(*intrinsic_, float, 1, 1) = fv;
  CV_MAT_ELEM(*intrinsic_, float, 2, 2) = 1.0;
  CV_MAT_ELEM(*intrinsic_, float, 0, 1) = 0.0;
  CV_MAT_ELEM(*intrinsic_, float, 1, 0) = 0.0;
  CV_MAT_ELEM(*intrinsic_, float, 2, 0) = 0.0;
  CV_MAT_ELEM(*intrinsic_, float, 2, 1) = 0.0;


 // std::cout<<"Outputting the intrinsics herer e"<<cv::Mat(intrinsic_)<<std::endl;

  while(iter < k && iter < max_k)
  {
    // sampling
    for(int i=0; i<NOM; i++)
    {
      int temp_idx= 0;
      bool found = true;
      //std::cout<<temp_idx<<"ere"<<std::endl;
      while(found)
      {
        temp_idx = cvRandInt(&rng) % n;
     //   std::cout<<temp_idx<<"ere"<<std::endl;

        found = false;
        for(int j=0; j<i; j++)
        {
          if(rand_idx[j] == temp_idx)
            found = true;
        }
      }
      rand_idx[i] = temp_idx;
    }
    // model parameters fitted to rand_idx

  //  std::cout<<"Outputting the intrinsics fdgdfg"<<frame_ref->cam_->errorMultiplier2()<<std::endl;
    ePnP.reset_correspondences();
    for(int i=0; i<NOM; i++)
    {
      ePnP.add_correspondence(objOutliers3D[rand_idx[i]].x, objOutliers3D[rand_idx[i]].y, objOutliers3D[rand_idx[i]].z, imgOutliers[rand_idx[i]].x, imgOutliers[rand_idx[i]].y);
    }
    double err = ePnP.compute_pose(R_est, T_est);

 //   std::cout<<"number"<<iter<<" "<<R_est<<T_est<<std::endl;

    // project rest points into the image plane
    CV_MAT_ELEM(*P, float, 0, 0) = (float) R_est[0][0];
    CV_MAT_ELEM(*P, float, 0, 1) = (float) R_est[0][1];
    CV_MAT_ELEM(*P, float, 0, 2) = (float) R_est[0][2];

    CV_MAT_ELEM(*P, float, 1, 0) = (float) R_est[1][0];
    CV_MAT_ELEM(*P, float, 1, 1) = (float) R_est[1][1];
    CV_MAT_ELEM(*P, float, 1, 2) = (float) R_est[1][2];

    CV_MAT_ELEM(*P, float, 2, 0) = (float) R_est[2][0];
    CV_MAT_ELEM(*P, float, 2, 1) = (float) R_est[2][1];
    CV_MAT_ELEM(*P, float, 2, 2) = (float) R_est[2][2];

    CV_MAT_ELEM(*P, float, 0, 3) = (float) T_est[0];
    CV_MAT_ELEM(*P, float, 1, 3) = (float) T_est[1];
    CV_MAT_ELEM(*P, float, 2, 3) = (float) T_est[2];

  //   std::cout<<"number"<<iter<<" "<<R_est<<T_est<<std::endl;
//
    cvGEMM(intrinsic_, P, 1, NULL, 0, P2, 0);



    // x2d_proj = P * x3d_h
    cvGEMM(P2, x3d_h, 1, NULL, 0, x2d_proj, 0);

  ////   std::cout<<"number"<<iter<<" "<<R_est<<T_est<<std::endl;


    for(int i=0; i<n; i++)
    {
      float u = CV_MAT_ELEM(*x2d_proj, float, 0, i);
      float v = CV_MAT_ELEM(*x2d_proj, float, 1, i);
      float w = CV_MAT_ELEM(*x2d_proj, float, 2, i);

      CV_MAT_ELEM(*x2d_proj, float, 0, i) = u/w;
      CV_MAT_ELEM(*x2d_proj, float, 1, i) = v/w;
      // save reprojection error to third rows
      CV_MAT_ELEM(*x2d_proj, float, 2, i) = sqrt((u/w - imgOutliers[i].x)*(u/w - imgOutliers[i].x) + (v/w - imgOutliers[i].y)*(v/w - imgOutliers[i].y));
    }


//  std::cout<<"number 234234  "<<iter<<" "<<R_est<<T_est<<std::endl;
    // Count number of inliers
    int noi = 0;
    for(int i=0; i<n; i++)
    {
 //      std::cout<<"number 234234 "<<" "<<CV_MAT_ELEM(*x2d_proj, float, 2, i)<<" "<<i<<std::endl;
        if( CV_MAT_ELEM(*x2d_proj, float, 2, i)  < th*th)
      {
        inlier_idx[i] = 1;
        noi++;
//        std::cout<<"number "<<" "<<CV_MAT_ELEM(*x2d_proj, float, 2, i)<<"  "<<i <<std::endl;
          // Count number of inliers
      }
      else
        inlier_idx[i] = 0;
    }
//   std::cout<<"number 234234"<<iter<<" "<<R_est<<T_est<<std::endl;

    if(noi > best_noi)
    {
      for(int i=0; i<NOM; i++)
        inlier_idx[rand_idx[i]] = 1;
      best_noi = noi;
      best_inlier_idx = inlier_idx;
      // Determine adaptive number of iteration
    //  double e = 1. - (double)best_noi/(double)n;
    //  k = (int)(log(1. - p)/log(1. - pow(1.-e, NOM)));
    }
 //   std::cout<<"number 2334"<<iter<<" "<<R_est<<T_est<<std::endl;
    iter++;
    if(0) printf("(%d/%d) iter: %d/%d\n", iter, k, best_noi, n);
  }

  if(best_noi > 0)
  {
    ePnP.set_maximum_number_of_correspondences(best_noi+NOM);
    ePnP.reset_correspondences();
    for(int i=0; i<n; i++)
    {
      if(best_inlier_idx[i])
        ePnP.add_correspondence(objOutliers3D[i].x, objOutliers3D[i].y, objOutliers3D[i].z, imgOutliers[i].x, imgOutliers[i].y);
    }

    double err = ePnP.compute_pose(R_est, T_est);

   /* CV_MAT_ELEM(*pmPose, float, 0, 0) = R_est[0][0];
    CV_MAT_ELEM(*pmPose, float, 1, 0) = R_est[1][0];
    CV_MAT_ELEM(*pmPose, float, 2, 0) = R_est[2][0];
    CV_MAT_ELEM(*pmPose, float, 3, 0) = 0.0;
    CV_MAT_ELEM(*pmPose, float, 0, 1) = R_est[0][1];
    CV_MAT_ELEM(*pmPose, float, 1, 1) = R_est[1][1];
    CV_MAT_ELEM(*pmPose, float, 2, 1) = R_est[2][1];
    CV_MAT_ELEM(*pmPose, float, 3, 1) = 0.0;
    CV_MAT_ELEM(*pmPose, float, 0, 2) = R_est[0][2];
    CV_MAT_ELEM(*pmPose, float, 1, 2) = R_est[1][2];
    CV_MAT_ELEM(*pmPose, float, 2, 2) = R_est[2][2];
    CV_MAT_ELEM(*pmPose, float, 3, 2) = 0.0;
    CV_MAT_ELEM(*pmPose, float, 0, 3) = T_est[0];
    CV_MAT_ELEM(*pmPose, float, 1, 3) = T_est[1];
    CV_MAT_ELEM(*pmPose, float, 2, 3) = T_est[2];
    CV_MAT_ELEM(*pmPose, float, 3, 3) = 1.0;*/
  }

  // Display estimated pose
#if 0
  cout << "Found pose:" << endl;
  ePnP.print_pose(R_est, T_est);
#endif

  // Refined points
  objInliers.clear();
  imgInliers.clear();
  objInliers3D.clear();
  vector<CvPoint2D32f> pt1_out, pt2_out;
  vector<CvPoint3D32f> pt3_out;
  for(int i=0; i<n; i++)
  {
    if(best_inlier_idx[i] == 1) // inliers only
    {
      objInliers.push_back(objOutliers[i]);
      imgInliers.push_back(imgOutliers[i]);
      objInliers3D.push_back(objOutliers3D[i]);
    }
    else // outliers
    {
      pt1_out.push_back(objOutliers[i]);
      pt2_out.push_back(imgOutliers[i]);
      pt3_out.push_back(objOutliers3D[i]);
    }
  }

  objOutliers = pt1_out;
  imgOutliers = pt2_out;
  objOutliers3D = pt3_out;


  MatrixXd Rotation(3,3);
  Vector3d translation;

  for(int i=0;i<3;i++)
  {
      for(int j=0;j<3;j++)
      {
          Rotation(i,j) = R_est[i][j];
      }
      translation(i) = T_est[i];
  }

  pose = SE3(Rotation,translation);




  cvReleaseMat(&P);
  cvReleaseMat(&P2);
  cvReleaseMat(&x3d_h);
  cvReleaseMat(&x2d_proj);

  return int(objInliers.size());//*/
}












void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec)
{
  Features new_features;
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

  // now for all maximum corners, initialize a new seed
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
    f_vec.push_back(ftr->f);
    delete ftr;
  });
}

void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities)
{
  const double klt_win_size = 30.0;
  const int klt_max_iter = 30;
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

  //<<"Calculating the opical flow"<<std::endl;
  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  f_cur.clear(); f_cur.reserve(px_cur.size());
  disparities.clear(); disparities.reserve(px_cur.size());
  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    ////<<"Status"<<" "<<  status[i];
      if(!status[i])
    {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}


void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref)
{
  //<<"in homography computation"<<std::endl;
  vector<Vector2d, aligned_allocator<Vector2d> > uv_ref(f_ref.size());
  vector<Vector2d, aligned_allocator<Vector2d> > uv_cur(f_cur.size());
  for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
  {
    uv_ref[i] = vk::project2d(f_ref[i]);
    uv_cur[i] = vk::project2d(f_cur[i]);
 //   xyz_in_cur[i] =  vk::triangulateFeatureNonLin(T_cur_from_ref.rotation_matrix(),T_cur_from_ref.translation(),f_cur[i],f_ref[i]);
  }

//  //<<"after triaangualtion"<<T_cur_from_ref.rotation_matrix()<<" "<<T_cur_from_ref.translation()<<std::endl;

  vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  Homography.computeSE3fromMatches();
  vector<int> outliers;

  vk::computeInliers(f_cur, f_ref,
                       T_cur_from_ref.rotation_matrix(), T_cur_from_ref.translation(),
                       reprojection_threshold, focal_length,
                       xyz_in_cur, inliers, outliers);//*/

  /*vk::computeInliers(f_cur, f_ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);
  T_cur_from_ref = Homography.T_c2_from_c1;//*/
}


















} // namespace initialization
} // namespace svo
