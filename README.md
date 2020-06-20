**Introduction**



- Camera includes single camera calibration and undistorting of images.
- StereoCalibration includes stereo camera calibration and distortion of images.
- StereoDisparity computes the disparity map of image, which uses the result of stereo calibration.



**Camera functions**



- findChessboardCorners function attempts to determine whether the input image is a view of the chessboard pattern and locate the internal chessboard corners

- cornerSubPix function iterates to find the sub-pixel accurate location of corners or radial saddle points
- drawChessboardCorners function draws individual chessboard corners detected either as red circles if the board was not found, or as colored corners connected with lines if the board was found
- calibrateCamera function estimates the intrinsic camera parameters and extrinsic parameters for each of the views
- initUndistortRectifyMap function computes the joint undistortion and rectification transformation and represents the result in the form of maps for remap



**StereoCalibration function**



- functions, including findChessboardCorners, cornerSubPix, initUndistortRectifyMap and drawChessboardCorners, have the same use with above functions.
- stereoCalibrate function estimates transformation between two cameras making a stereo pair.
- undistortPoints function computes the ideal point coordinates from the observed point coordinates.
- computeCorrespondEpilines function finds the equation of the corresponding epipolar line in the other image.
- stereoRectify function computes the rotation matrices for each camera that (virtually) make both camera image planes the same plane.



**StereoDisparity functions**



- compute function computes disparity map for the specified stereo pair
- functions, including stereoRectify and initUndistortRectifyMap, have the same use with above functions.