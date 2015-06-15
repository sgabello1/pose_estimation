// Ceres Pose estimation - direct approach
// 

#include "CeresOptimizer.h"


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp" //TickMeter

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

void printHelp()
{
  std::cout<<" <config_file.yml> <imgRGB0.png> <imgRGB1.png>"<<std::endl;
}

int main (int argc,char ** argv)
{
  if(argc<3){printHelp();return -1;}

  typedef double CoordinateType;
  typedef phovo::Numeric::Matrix33RowMajor< CoordinateType > Matrix33Type;
  typedef phovo::Numeric::Matrix44RowMajor< CoordinateType > Matrix44Type;
  typedef phovo::Numeric::VectorCol8< CoordinateType >       Vector8Type;

  typedef unsigned char PixelType;
  typedef cv::Mat_< PixelType >      IntensityImageType;
  typedef cv::Mat_< CoordinateType > DepthImageType;

  //Set the camera parameters
  Matrix33Type intrinsicMatrix;
  intrinsicMatrix << 525., 0., 319.5,
                     0., 525., 239.5,
                     0., 0., 1.;


  //Load two RGB frames (RGB and depth images)
  IntensityImageType imgGray0 = cv::imread(argv[2],0);

  IntensityImageType imgGray1 = cv::imread(argv[3],0);
  

  //Define the photoconsistency odometry object and set the input parameters

  phovo::Ceres::CPhotoconsistencyOdometryCeres< PixelType, CoordinateType > photoconsistencyOdometry;


  Vector8Type stateVector;
  stateVector << 1., 0., 0.,
                 0., 1., 0.,
                 0., 0.; //M transformation matrix start as identity

  photoconsistencyOdometry.ReadConfigurationFile( std::string( argv[1] ) );
  photoconsistencyOdometry.SetIntrinsicMatrix( intrinsicMatrix );
  photoconsistencyOdometry.SetSourceFrame( imgGray0 );
  photoconsistencyOdometry.SetTargetFrame( imgGray1);
  photoconsistencyOdometry.SetInitialStateVector( stateVector );
 

  //Optimize the problem to estimate the rigid transformation
  cv::TickMeter tm;tm.start();
  photoconsistencyOdometry.Optimize();
  tm.stop();
  std::cout << "Time = " << tm.getTimeSec() << " sec." << std::endl;

  //Show results
  Matrix33Type M = photoconsistencyOdometry.GetOptimalRigidTransformationMatrix();
  std::cout << " Transformation matrix " << std::endl << M << std::endl;
  IntensityImageType warpedImage;
  phovo::warpImage< PixelType, CoordinateType >( imgGray0, imgDepth0, warpedImage, Rt, intrinsicMatrix );
  IntensityImageType imgDiff;
  cv::absdiff( imgGray1, warpedImage, imgDiff );
  cv::imshow( "main::imgDiff", imgDiff );
  cv::waitKey( 0 );

    return 0;
}

