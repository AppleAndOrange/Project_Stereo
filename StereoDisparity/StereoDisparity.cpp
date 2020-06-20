
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <string>

using namespace cv;
using namespace std;

int main()
{
    Mat imgLeft = imread("left01.jpg", IMREAD_GRAYSCALE);
    Mat imgRight = imread("right01.jpg", IMREAD_GRAYSCALE);
    if(imgLeft.empty()) {
        printf("Command-line parameter error: could not load the first input image file\n");
        return 0;
    }
    if (imgRight.empty()) {
        printf("Command-line parameter error: could not load the second input image file\n");
        return 0;
    }
    Rect roi1, roi2;
    string intrinsicFilename = "intrinsic.yml";
    string extrinsicFilename = "extrinsics.yml";
    FileStorage fs(intrinsicFilename, FileStorage::READ);
    if (!fs.isOpened())
    {
        printf("Failed to open file %s\n", intrinsicFilename.c_str());
        return 0;
    }
    Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;
    M1 *= 1.0;
    M2 *= 1.0;
    fs.open(extrinsicFilename, FileStorage::READ);
    if (!fs.isOpened())
    {
        printf("Failed to open file %s\n", extrinsicFilename.c_str());
        return -1;
    }
    Mat R, T, R1, P1, R2, P2, Q;
    fs["R"] >> R;
    fs["T"] >> T;
    Size imgSize = imgLeft.size();
    stereoRectify(M1, D1, M2, D2, imgSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imgSize, &roi1, &roi2);
    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(M1, D1, R1, P1, imgSize, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, imgSize, CV_16SC2, map21, map22);
    Mat img1r, img2r;
    remap(imgLeft, img1r, map11, map12, INTER_LINEAR);
    remap(imgRight, img2r, map21, map22, INTER_LINEAR);

    imgLeft = img1r;
    imgRight = img2r;


    Mat disp, disp8;
    int nDisparities = 16 * 5;
    int SADWindowsSize = 7;//it must be odd
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
    sgbm->setPreFilterCap(63);
    sgbm->setBlockSize(SADWindowsSize);
    int cn = imgLeft.channels();
    sgbm->setP1(8 * cn * SADWindowsSize * SADWindowsSize);
    sgbm->setP2(32 * cn * SADWindowsSize * SADWindowsSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(nDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);
    sgbm->compute(imgLeft, imgRight, disp);
    disp.convertTo(disp8, CV_8U, 255 / (nDisparities * 16.));
    imshow("disparity", disp8);
    waitKey();
    return 0;
}
