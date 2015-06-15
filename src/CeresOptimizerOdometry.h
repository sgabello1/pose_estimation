// Ceres Optimizer

#define ENABLE_OPENMP_MULTITHREADING_WARP_IMAGE 0

#include "opencv2/imgproc/imgproc.hpp"
#include "eigen3/Eigen/Dense"

#include "Matrix.h"

namespace phovo
{

/*!This abstract class defines the mandatory methods that any derived class must implement to compute the rigid (6DoF) transformation that best aligns a pair of RGBD frames using a photoconsistency maximization approach.*/
template< class TPixel, class TCoordinate >
class CPhotoconsistencyOdometry
{
public:
  typedef TPixel                PixelType;
  typedef cv::Mat_< PixelType > IntensityImageType;

  typedef TCoordinate                CoordinateType;
  typedef cv::Mat_< CoordinateType > DepthImageType;

  typedef Numeric::Matrix33RowMajor< CoordinateType > Matrix33Type;
  typedef Numeric::VectorCol8< CoordinateType >       Vector8Type;
  typedef Numeric::VectorCol4< CoordinateType >       Vector4Type;

  /*!Sets the 3x3 intrinsic pinhole matrix.*/
  virtual void SetIntrinsicMatrix( const Matrix33Type & intrinsicMatrix ) = 0;

  /*!Sets the source (Intensity+Depth) frame.*/
  virtual void SetSourceFrame( const IntensityImageType & intensityImage ) = 0;

  /*!Sets the source (Intensity+Depth) frame.*/
  virtual void SetTargetFrame( const IntensityImageType & intensityImage) = 0;

  /*!Initializes the state vector to a certain value. The optimization process uses
   *the initial state vector as the initial estimate.*/
  virtual void SetInitialStateVector( const Vector8Type & initialStateVector ) = 0;

  /*!Launches the least-squares optimization process to find the configuration of the
   *state vector parameters that maximizes the photoconsistency between the source and
   *target frame.*/
  virtual void Optimize() = 0;

  /*!Returns the optimal state vector. This method has to be called after calling the
   *Optimize() method.*/
  virtual Vector8Type GetOptimalStateVector() const = 0;

  /*!Returns the optimal 4x4 rigid transformation matrix between the source and target frame.
   *This method has to be called after calling the Optimize() method.*/
  virtual Matrix33Type GetOptimalRigidTransformationMatrix() const = 0;
};

} //end namespace phovo

