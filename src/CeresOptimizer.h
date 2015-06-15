//  Direct Pose Estimation for Homography

#include "config.h"
//#ifdef PHOVO_WITH_CERES // Check for Ceres-solver

#ifndef _CPHOTOCONSISTENCY_ODOMETRY_CERES_
#define _CPHOTOCONSISTENCY_ODOMETRY_CERES_

#define ENABLE_GAUSSIAN_BLUR 1
#define ENABLE_BOX_FILTER_BLUR 0
#define ENABLE_OPENMP_MULTITHREADING_CERES 0 // Enables OpenMP for CPhotoconsistencyOdometryCeres

#include "CeresOptimizerOdometry.h"

#include "sample.h"
#include "jet_extras.h"
#include "ceres/ceres.h"
#include "opencv2/highgui/highgui.hpp" //visualize iterations

namespace phovo
{

namespace Ceres
{

/*!This class computes the rigid (6DoF) transformation that best aligns a pair of RGBD frames using a photoconsistency maximization approach.
To estimate the rigid transformation, this class implements a coarse to fine approach. Thus, the algorithm starts finding a first pose approximation at
a low resolution level and uses the estimate to initialize the optimization at greater image scales. This class uses Ceres autodifferentiation to compute the derivatives of the cost function.*/
template< class TPixel, class TCoordinate >
class CPhotoconsistencyOdometryCeres :
    public CPhotoconsistencyOdometry< TPixel, TCoordinate >
{
public:
  typedef CPhotoconsistencyOdometry< TPixel, TCoordinate > Superclass;

  typedef typename Superclass::PixelType          PixelType;
  typedef typename Superclass::CoordinateType     CoordinateType;
  typedef typename Superclass::IntensityImageType IntensityImageType;
  typedef typename Superclass::DepthImageType     DepthImageType;
  typedef typename Superclass::Matrix33Type       Matrix33Type;
  typedef typename Superclass::Vector8Type        Vector8Type;
  typedef typename Superclass::Vector4Type        Vector4Type;

private:
  typedef DepthImageType                            InternalIntensityImageType;
  typedef std::vector< InternalIntensityImageType > InternalIntensityImageContainerType;
  typedef std::vector< DepthImageType >             DepthImageContainerType;
  typedef std::vector< CoordinateType >             CoordinateContainerType;
  typedef std::vector< int >                        IntegerContainerType;

  /*!Intensity (gray), depth and gradient image pyramids. Each pyramid has 'numOptimizationLevels' levels.*/
  InternalIntensityImageContainerType m_IntensityPyramid0;
  InternalIntensityImageContainerType m_IntensityPyramid1;
  DepthImageContainerType             m_DepthPyramid0;
  DepthImageContainerType             m_DepthPyramid1;
  InternalIntensityImageContainerType m_IntensityGradientXPyramid1;
  InternalIntensityImageContainerType m_IntensityGradientYPyramid1;
  /*!Camera matrix (intrinsic parameters).*/
  Matrix33Type m_IntrinsicMatrix;
  /*!Current optimization level. Level 0 corresponds to the higher image resolution.*/
  int m_OptimizationLevel;
  /*!Number of optimization levels.*/
  int m_NumOptimizationLevels;
  /*!Size (in pixels) of the blur filter (at each level).*/
  IntegerContainerType m_BlurFilterSizes;
  /*!Scaling factor applied to the image gradients (at each level).*/
  CoordinateContainerType m_ImageGradientsScalingFactors;
  /*!Maximum number of iterations for the optimization algorithm (at each level).*/
  IntegerContainerType m_MaxNumIterations;
  /*!Enable the visualization of the optimization process (only for debug).*/
  bool m_VisualizeIterations;
  /*!State vector.*/
  CoordinateType m_StateVector[ 8 ]; //Parameter vector (x y z yaw pitch roll)
  /*!Current iteration at the current optimization level.*/
  int m_Iteration;
  CoordinateContainerType m_FunctionTolerances;
  CoordinateContainerType m_GradientTolerances;
  CoordinateContainerType m_ParameterTolerances;
  CoordinateContainerType m_InitialTrustRegionRadiuses;
  CoordinateContainerType m_MaxTrustRegionRadiuses;
  CoordinateContainerType m_MinTrustRegionRadiuses;
  CoordinateContainerType m_MinRelativeDecreases;
  int m_NumLinearSolverThreads;
  int m_NumThreads;
  bool m_MinimizerProgressToStdout;
  /*!Minimum allowed depth to consider a depth pixel valid.*/
  CoordinateType m_MinDepth;
  /*!Maximum allowed depth to consider a depth pixel valid.*/
  CoordinateType m_MaxDepth;

  class ResidualRGBDPhotoconsistency
  {
  private:
    Matrix33Type m_IntrinsicMatrix;
    int m_OptimizationLevel;
    InternalIntensityImageType m_SourceIntensityImage;
    DepthImageType m_SourceDepthImage;
    InternalIntensityImageType m_TargetIntensityImage;
    InternalIntensityImageType m_TargetGradXImage;
    InternalIntensityImageType m_TargetGradYImage;
    //CoordinateType m_MaxDepth;
    //CoordinateType m_MinDepth;

  public:
    ResidualRGBDPhotoconsistency( const Matrix33Type & intrinsicMatrix,
                                  const int optimizationLevel,
                                  const InternalIntensityImageType & sourceIntensityImage,
                                  const InternalIntensityImageType & targetIntensityImage,
                                  const InternalIntensityImageType & targetGradXImage,
                                  const InternalIntensityImageType & targetGradYImage ) :

      m_IntrinsicMatrix( intrinsicMatrix ), m_OptimizationLevel( optimizationLevel ),
      m_SourceIntensityImage( sourceIntensityImage ),
      m_TargetIntensityImage( targetIntensityImage ),
      m_TargetGradXImage( targetGradXImage ),
      m_TargetGradYImage( targetGradYImage )

    {}

    template <typename T>
    bool operator()( const T* const stateVector,
                     T* residuals ) const
    {
      int nRows = m_SourceIntensityImage.rows;
      int nCols = m_SourceIntensityImage.cols;
      T m[3][3];

      m[0][0] = stateVector[0];
      m[0][1] = stateVector[1];
      m[0][2] = stateVector[2];

      m[1][0] = stateVector[3];
      m[1][1] = stateVector[4];      
      m[1][2] = stateVector[5];      

      m[2][0] = stateVector[6];
      m[2][1] = stateVector[7];      
      m[2][2] = T(1.);      

      // for(int i = 0; i < 8; i++)
      //   std::cout << stateVector[i] << std::endl;


      //Initialize the error function (residuals) with an initial value
      #if ENABLE_OPENMP_MULTITHREADING_CERES
      #pragma omp parallel for
      #endif
      for( int r=0; r<nRows; r++ )
      {
        for( int c=0; c<nCols; c++ )
        {
          residuals[ nCols*r+c ] = T( 0. );
        }
      }

      T residualScalingFactor = T( 1. );

      #if ENABLE_OPENMP_MULTITHREADING_CERES
      #pragma omp parallel for
      #endif
      for( int r=0; r<nRows; r++ )
      {

        T transformed_r,transformed_c; // 2D coordinates of the transformed pixel(r,c) of frame 1
        T pixel1; //Intensity value of the pixel(r,c) of the warped frame 1
        T pixel2; //Intensity value of the pixel(r,c) of frame 2

        for( int c=0; c<nCols; c++ )
        {
         
            // Now it s an homography
            transformed_c = (m[0][0]*T(c) +  m[0][1]*T(r) + m[0][1])/(m[2][0]*T(c) +  m[2][1]*T(r) + m[2][2]); //transformed x (2D)
            transformed_r = (m[1][0]*T(c) +  m[1][1]*T(r) + m[1][1])/(m[2][0]*T(c) +  m[2][1]*T(r) + m[2][2]); //transformed y (2D)
            // std::cout << " c,r  "<< c << " , "<<  r << "transformed_c, transformed_r " << transformed_c << " , " << transformed_r << std::endl;

            //Asign the intensity value to the warped image and compute the difference between the transformed
            //pixel of frame 1 and the corresponding pixel of frame 2. Compute the error function
            if( transformed_r >= T(0.) && transformed_r < T(nRows) &&
                transformed_c >= T(0.) && transformed_c < T(nCols) )
            {
              //Compute the proyected coordinates of the transformed 3D point
              int transformed_r_scalar = static_cast<int>(ceres::JetOps<T>::GetScalar(transformed_r));
              int transformed_c_scalar = static_cast<int>(ceres::JetOps<T>::GetScalar(transformed_c));

              //Compute the pixel residual
              pixel1 = T( m_SourceIntensityImage(r,c) );
              pixel2 = SampleWithDerivative< T, InternalIntensityImageType >( m_TargetIntensityImage,
                                             m_TargetGradXImage,
                                             m_TargetGradYImage, transformed_c, transformed_r );
              residuals[ nCols * transformed_r_scalar + transformed_c_scalar ] =
                residualScalingFactor * ( pixel2 - pixel1 );
                // std::cout << "residuals " << residuals[ nCols * transformed_r_scalar + transformed_c_scalar ] << std::endl;
            }
          
        }
      }

      return true;
    }
  };


template< class TImage >
void BuildPyramid( const TImage & img,
                   std::vector< TImage > & pyramid,
                   const int levels, const bool applyBlur )
{
  typedef TImage ImageType;

  //Create space for all the images
  pyramid.resize( levels );

  double factor = 1.;
  for( int level=0; level<levels; level++ )
  {
    //Create an auxiliar image of factor times the size of the original image
    ImageType imgAux;
    if( level!=0 )
    {
      cv::resize( img, imgAux, cv::Size(0,0), factor, factor );
    }
    else
    {
      imgAux = img;
    }

    //Blur the resized image with different filter size depending on the current pyramid level
    if( applyBlur )
    {
      int blurFilterSize = m_BlurFilterSizes[level];
      #if ENABLE_GAUSSIAN_BLUR
      if( blurFilterSize>0 )
      {
        cv::GaussianBlur( imgAux, imgAux, cv::Size( blurFilterSize, blurFilterSize ), 3 );
        cv::GaussianBlur( imgAux, imgAux, cv::Size( blurFilterSize, blurFilterSize ), 3 );
      }
      #elif ENABLE_BOX_FILTER_BLUR
      if( blurFilterSize>0 )
      {
        cv::blur( imgAux, imgAux, cv::Size( blurFilterSize, blurFilterSize ) );
        cv::blur( imgAux, imgAux, cv::Size( blurFilterSize, blurFilterSize ) );
      }
      #endif
    }

    //Assign the resized image to the current level of the pyramid
    pyramid[level] = imgAux;

    factor = factor/2;
  }
}

void BuildDerivativesPyramids( InternalIntensityImageContainerType & imagePyramid,
                               InternalIntensityImageContainerType & derXPyramid,
                               InternalIntensityImageContainerType & derYPyramid)
{
  //Compute image gradients
  double delta = 0.0;
  int ddepth = m_IntensityPyramid0[0].type();

  //Create space for all the derivatives images
  derXPyramid.resize(imagePyramid.size());
  derYPyramid.resize(imagePyramid.size());

  for( size_t level=0; level<imagePyramid.size(); level++ )
  {
    // Compute the gradient in x
    InternalIntensityImageType imgGray1_grad_x;
    cv::Scharr( imagePyramid[level], derXPyramid[level], ddepth, 1, 0,
                m_ImageGradientsScalingFactors[level], delta, cv::BORDER_DEFAULT );

    // Compute the gradient in y
    InternalIntensityImageType imgGray1_grad_y;
    cv::Scharr( imagePyramid[level], derYPyramid[level],ddepth, 0, 1,
                m_ImageGradientsScalingFactors[level], delta, cv::BORDER_DEFAULT );
  }
}

public:

CPhotoconsistencyOdometryCeres() : m_MinDepth( 0.3 ), m_MaxDepth( 5.0 ){}

~CPhotoconsistencyOdometryCeres(){}

/*!Sets the 3x3 intrinsic camera matrix*/
void SetIntrinsicMatrix( const Matrix33Type & intrinsicMatrix )
{
  m_IntrinsicMatrix = intrinsicMatrix;
}

/*!Sets the source (Intensity+Depth) frame.*/
void SetSourceFrame( const IntensityImageType & intensityImage)
{

  //Create an auxialiary image from the imput image
  InternalIntensityImageType intensityImageAux;

  intensityImage.convertTo( intensityImageAux, -1, 1./255 );

  //Compute image pyramids for the grayscale and depth images
  BuildPyramid( intensityImageAux, m_IntensityPyramid0, m_NumOptimizationLevels, true );
}

/*!Sets the source (Intensity+Depth) frame. Depth image is ignored*/
void SetTargetFrame( const IntensityImageType & intensityImage )
{
   //Create an auxialiary image from the imput image
  InternalIntensityImageType intensityImageAux;
  intensityImage.convertTo( intensityImageAux, -1, 1./255 );

  //Compute image pyramids for the grayscale and depth images
  BuildPyramid( intensityImageAux, m_IntensityPyramid1, m_NumOptimizationLevels, true );

  //Compute image pyramids for the gradients images
  BuildDerivativesPyramids( m_IntensityPyramid1, m_IntensityGradientXPyramid1, m_IntensityGradientYPyramid1 );
}

/*!Initializes the state vector to a certain value. The optimization process uses the initial state vector as the initial estimate.*/
void SetInitialStateVector( const Vector8Type & initialStateVector )
{
  m_StateVector[0] = initialStateVector( 0 );
  m_StateVector[1] = initialStateVector( 1 );
  m_StateVector[2] = initialStateVector( 2 );
  m_StateVector[3] = initialStateVector( 3 );
  m_StateVector[4] = initialStateVector( 4 );
  m_StateVector[5] = initialStateVector( 5 );
  m_StateVector[6] = initialStateVector( 6 );
  m_StateVector[7] = initialStateVector( 7 );
}

/*!Launches the least-squares optimization process to find the configuration of the state vector parameters that maximizes the photoconsistency between the source and target frame.*/
void Optimize()
{
  for( m_OptimizationLevel = m_NumOptimizationLevels-1;
       m_OptimizationLevel >= 0; m_OptimizationLevel-- )
  {
    if( m_MaxNumIterations[ m_OptimizationLevel] > 0 ) //compute only if the number of maximum iterations are greater than 0
    {
      int nRows = m_IntensityPyramid0[ m_OptimizationLevel ].rows;
      int nCols = m_IntensityPyramid0[ m_OptimizationLevel ].cols;
      int nPoints = nRows * nCols;

      // Build the problem.
      ceres::Problem problem;

      // Set up the only cost function (also known as residual). This uses
      // auto-differentiation to obtain the derivative (jacobian).
      problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ResidualRGBDPhotoconsistency,ceres::DYNAMIC,6>(
            new ResidualRGBDPhotoconsistency( m_IntrinsicMatrix, m_OptimizationLevel,
                                              m_IntensityPyramid0[ m_OptimizationLevel ],
                                              m_IntensityPyramid1[ m_OptimizationLevel ],
                                              m_IntensityGradientXPyramid1[ m_OptimizationLevel ],
                                              m_IntensityGradientYPyramid1[ m_OptimizationLevel ]
                                              ),
            nPoints /*dynamic size*/),
            NULL,
            m_StateVector );

      // Run the solver!
      ceres::Solver::Options options;
      options.max_num_iterations = m_MaxNumIterations[ m_OptimizationLevel ];
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;//ceres::DENSE_QR;
      options.minimizer_progress_to_stdout = m_MinimizerProgressToStdout;
      options.function_tolerance = m_FunctionTolerances[ m_OptimizationLevel ];
      options.gradient_tolerance = m_GradientTolerances[ m_OptimizationLevel ];
      options.parameter_tolerance = m_ParameterTolerances[ m_OptimizationLevel ];
      options.initial_trust_region_radius = m_InitialTrustRegionRadiuses[ m_OptimizationLevel ];
      options.max_trust_region_radius = m_MaxTrustRegionRadiuses[ m_OptimizationLevel ];
      options.min_trust_region_radius = m_MinTrustRegionRadiuses[ m_OptimizationLevel ];
      options.min_relative_decrease = m_MinRelativeDecreases[ m_OptimizationLevel ];
      options.num_linear_solver_threads = m_NumLinearSolverThreads;
      options.num_threads = m_NumThreads;
      options.max_num_consecutive_invalid_steps = 0;

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      std::cout << summary.FullReport() << std::endl;
    }
  }

  //After all the optimization process the optimization level is 0
  m_OptimizationLevel = 0;
}

/*!Returns the optimal state vector. This method has to be called after calling the Optimize() method.*/
Vector8Type GetOptimalStateVector() const
{
  Vector8Type statevector;
  statevector(0) = m_StateVector[0];
  statevector(1) = m_StateVector[1];
  statevector(2) = m_StateVector[2];
  statevector(3) = m_StateVector[3];
  statevector(4) = m_StateVector[4];
  statevector(5) = m_StateVector[5];
  statevector(6) = m_StateVector[6];
  statevector(7) = m_StateVector[7];


  return statevector;
}

/*!Returns the optimal 4x4 rigid transformation matrix between the source and target frame. This method has to be called after calling the Optimize() method.*/
Matrix33Type GetOptimalRigidTransformationMatrix() const
{
  Matrix33Type m;

      m(0,0) = m_StateVector[0];
      m(0,1) = m_StateVector[1];
      m(0,2) = m_StateVector[2];

      m(1,0) = m_StateVector[3];
      m(1,1) = m_StateVector[4];      
      m(1,2) = m_StateVector[5];      

      m(2,0) = m_StateVector[6];
      m(2,1) = m_StateVector[7];      
      m(2,2) = 1;      
  
  return m;
}

/*!Reads the configuration parameters from a .yml file.*/
void ReadConfigurationFile( const std::string & fileName )
{
  cv::FileStorage fs( fileName, cv::FileStorage::READ );

  //Read the number of optimization levels
  fs["numOptimizationLevels"] >> m_NumOptimizationLevels;

  #if ENABLE_GAUSSIAN_BLUR || ENABLE_BOX_FILTER_BLUR
  //Read the blur filter size at every pyramid level
  fs["blurFilterSize (at each level)"] >> m_BlurFilterSizes;
  #endif

  //Read the scaling factor for each gradient image at each level
  fs["imageGradientsScalingFactor (at each level)"] >> m_ImageGradientsScalingFactors;

  //Read the number of Levenberg-Marquardt iterations at each optimization level
  fs["max_num_iterations (at each level)"] >> m_MaxNumIterations;

  //Read optimizer function tolerance at each level
  fs["function_tolerance (at each level)"] >> m_FunctionTolerances;

  //Read optimizer gradient tolerance at each level
  fs["gradient_tolerance (at each level)"] >> m_GradientTolerances;

  //Read optimizer parameter tolerance at each level
  fs["parameter_tolerance (at each level)"] >> m_ParameterTolerances;

  //Read optimizer initial trust region at each level
  fs["initial_trust_region_radius (at each level)"] >> m_InitialTrustRegionRadiuses;

  //Read optimizer max trust region radius at each level
  fs["max_trust_region_radius (at each level)"] >> m_MaxTrustRegionRadiuses;

  //Read optimizer min trust region radius at each level
  fs["min_trust_region_radius (at each level)"] >> m_MinTrustRegionRadiuses;

  //Read optimizer min LM relative decrease at each level
  fs["min_relative_decrease (at each level)"] >> m_MinRelativeDecreases;

  //Read the number of threads for the linear solver
  fs["num_linear_solver_threads"] >> m_NumLinearSolverThreads;

  //Read the number of threads for the jacobian computation
  fs["num_threads"] >> m_NumThreads;

  //Read the boolean value to determine if print the minimization progress or not
  fs["minimizer_progress_to_stdout"] >> m_MinimizerProgressToStdout;

  //Read the boolean value to determine if visualize the progress images or not
  fs["visualizeIterations"] >> m_VisualizeIterations;
}
};

} //end namespace Ceres

} //end namespace phovo

#endif

//#endif  // Check for Ceres-solver
