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

#ifndef SVO_INITIALIZATION_H
#define SVO_INITIALIZATION_H

#include <svo/global.h>
#include <opencv2/contrib/contrib.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

namespace svo {

class FrameHandlerMono;

/// Bootstrapping the map from the first two views.
namespace initialization {

enum InitResult { FAILURE, NO_KEYFRAME, SUCCESS };

/// Tracks features using Lucas-Kanade tracker and then estimates a homography.
class KltHomographyInit {
  friend class svo::FrameHandlerMono;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FramePtr frame_ref_;
  KltHomographyInit() { cv::initModule_nonfree();}; // for SURF feature};
  ~KltHomographyInit() {};
  InitResult addFirstFrame(FramePtr frame_ref);
  Vector3d triangulateFeatureNonLin1(const Matrix3d& R,  const Vector3d& t,
                                    const Vector3d& feature1, const Vector3d& feature2 );
  double computeInliers1(const vector<Vector3d>& features1, // c1
                 const vector<Vector3d>& features2, // c2
                 const Matrix3d& R,                 // R_c1_c2
                 const Vector3d& t,                 // c1_t
                 const double reproj_thresh,
                 double error_multiplier2,
                 vector<Vector3d>& xyz_vec,         // in frame c1
                 vector<int>& inliers,
                 vector<int>& outliers);
  InitResult addSecondFrame(FramePtr frame_ref);//, SE3 old_pose);
  InitResult addSecondFrame(FramePtr frame_ref, SE3 pose);
  void reset();
  void loadKeyframes(string path, int num);
  void buildKdTree(int num);
  void findCorrespondenceNN_FLANN(const CvSeq* imageKeypoints, const CvSeq* imageDescriptors, vector<int>& ptpairs, int numOfKeyframes, int num);
  SE3 estimatePose(int &num_of_corr, FramePtr frame_ref, int num);
  int refineCorrespondenceEpnpRANSAC(const vector<int>& ptpairs, FramePtr frame_ref, SE3 &pose, int num);

protected:
  vector<cv::Point2f> px_ref_;      //!< keypoints to be tracked in reference frame.
  vector<cv::Point2f> px_cur_;      //!< tracked keypoints in current frame.
  vector<Vector3d> f_ref_;          //!< bearing vectors corresponding to the keypoints in the reference image.
  vector<Vector3d> f_cur_;          //!< bearing vectors corresponding to the keypoints in the current image.
  vector<double> disparities_;      //!< disparity between first and second frame.
  vector<int> inliers_;             //!< inliers after the geometric check (e.g., Homography).
  vector<Vector3d> xyz_in_cur_;     //!< 3D points computed during the geometric check.
  SE3 T_cur_from_ref_;              //!< computed transformation between the first two frames.
  SE3 F_pose;

  std::vector< std::vector<CvMat*> >     keyframe_keypoints_2d_;
  std::vector< std::vector<CvMat*> >     keyframe_keypoints_3d_;

  std::vector< std::vector<CvMat*> >     keyframe_descriptors_;


  vector<vector<CvPoint2D32f>> keypoints2D;
  vector<vector<CvPoint3D32f>> keypoints3D;
  vector<vector<CvPoint2D32f>> input_keypoints_2d_;
  vector<vector<int>> keyframe_lut_;
  vector<cv::Mat> kfd_;



};





/// Detect Fast corners in the image.
void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec);



/// Compute optical flow (Lucas Kanade) for selected keypoints.
void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities);

void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref);

} // namespace initialization
} // namespace nslam

#endif // NSLAM_INITIALIZATION_H
