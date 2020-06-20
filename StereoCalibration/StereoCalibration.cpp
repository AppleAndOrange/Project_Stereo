
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace std;
using namespace cv;

static void StereoCalib(const vector<string>& imagelist, Size boardSize, float squareSize, bool displayCorners, bool useCalibrated, bool showRectified) {
    if (imagelist.size() % 2 != 0) {
        cout<< "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }
    const int maxScale = 2;
    vector<vector<Point2f>> imagePoints[2];//2D point
    vector<vector<Point3f>> objectPoints;//3D point
    Size imageSize;//图像大小

    int i, j, k;
    int nimages = imagelist.size() / 2;
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImagelist;
    for (i = 0, j = 0; i < nimages; i++) {
        for (k = 0; k < 2; k++) {
            string filename = imagelist[i * 2 + k];
            Mat img = imread(filename, 0);
            if (img.empty()) {
                break;
            }
            if (imageSize == Size()) {
                imageSize = img.size();                
            }
            else if(imageSize != img.size())
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for (int scale = 1; scale <= maxScale; scale++) {
                Mat timg;
                if (scale == 1) {
                    timg = img;
                }
                else
                {
                    resize(img, timg, Size(), scale, scale);
                    
                }
                //Finds the positions of internal corners of the chessboard
                //The function returns a non-zero value if all of the corners
                //are foundand they are placed in a certain order(row by row, left to right in every row).
                //Otherwise, if the function fails to find all the corners or reorder them, it returns 0.
                found = findChessboardCorners(timg, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                if (found) {
                    if (scale > 1) {
                        Mat cornersMat(corners);
                        cornersMat *= 1. / scale;
                    }
                    break;
                }
                else
                {
                    break;
                }
            }
            cornerSubPix(img, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
            if (displayCorners) {
                //cout << filename << endl;
                Mat cimg;
                drawChessboardCorners(img, boardSize, corners, found);
                double sf = 640. / MAX(img.rows, img.cols);
                resize(img, cimg, Size(), sf, sf);
                //imshow("corners", cimg);
                //waitKey(500);
            }
        }
        if (k == 2) {
            goodImagelist.push_back(imagelist[i * 2]);
            goodImagelist.push_back(imagelist[i * 2 + 1]);
            j++;
        }

    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if (nimages < 2) {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }
    
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    //保存三维坐标
    for (i = 0; i < nimages; i++) {
        for (j = 0; j < boardSize.height; j++) {
            for (k = 0; k < boardSize.width; k++) {
                objectPoints[i].push_back(Point3f(k * squareSize, j * squareSize, 0));
            }
        }
    }
    cout << "Running stereo calibration ...\n";
    Mat cameraMatrix[2], distCoeffs[2];
    //The function estimates and returns an initial camera matrix for the camera calibration process.
    //cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints[0], imageSize, 0);
    //cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints[1], imageSize, 0);
    Mat R, T, E, F;
    //The function estimates transformation between two cameras making a stereo pair.
    //Similar to calibrate Camera , the function minimizes the total re-projection error for all the
    //points in all the available views from both cameras.The function returns the final value of the
    //re-projection error.
    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
        cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        imageSize, R, T, E, F,
        CALIB_FIX_ASPECT_RATIO +
        CALIB_ZERO_TANGENT_DIST +
        CALIB_USE_INTRINSIC_GUESS +
        CALIB_SAME_FOCAL_LENGTH +
        CALIB_RATIONAL_MODEL +
        CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
    cout << "done with RMS error=" << rms << endl;
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for (i = 0; i < nimages; i++) {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        //对每张图片进行处理
        for (k = 0; k < 2; k++) {
            imgpt[k] = Mat(imagePoints[k][i]);
            //Computes the ideal point coordinates from the observed point coordinates.
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], Mat(), cameraMatrix[k]);
            //For every point in one of the two images of a stereo pair, the function finds the equation of the
            //corresponding epipolar line in the other image.
            computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
        }
        for (j = 0; j < npt; j++) {
            double errij= fabs(imagePoints[0][i][j].x * lines[1][j][0] +
                imagePoints[0][i][j].y * lines[1][j][1] + lines[1][j][2]) +
                fabs(imagePoints[1][i][j].x * lines[0][j][0] +
                    imagePoints[1][i][j].y * lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average epipolar err = " << err / npoints << endl;
    //save intrinsic parameters
    FileStorage fs("intrinsic.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
    {
        cout << "Error: can not save the intrinsic parameters\n";
    }
    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    /*
    The function computes the rotation matrices for each camera that (virtually) make both camera image planes the same plane
    @param R Rotation matrix between the coordinate systems of the first and the second cameras.
    @param T Translation vector between coordinate systems of the cameras.
    @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
    @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
    @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera.
    @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera.
    @param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see reprojectImageTo3D ).
    */
    stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q,
        CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else {
        cout << "Error: can not save the extrinsic parameters\n";
    }
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    vector<Point2f> allimgpt[2];
    for (k = 0; k < 2; k++) {
        for (i = 0; i < nimages; i++) {
            copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
    }
    //得到基础矩阵
    //F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
    //E=findEssentialMat
    //Mat H1, H2;
    //The function computes the rectification transformations without knowing intrinsic parameters of the camerasand their relative position in the space
    //@param H1 Output rectification homography matrix for the first image.
    //@param H2 Output rectification homography matrix for the second image.
    //stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
    Mat rmap[2][2], canvas;
    //The function computes the joint undistortion and rectification transformation and represents the result in the form of maps for remap
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R1, P1, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    double sf;
    int w, h;
    if (!isVerticalStereo)
    {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h, w * 2, CV_8UC3);
    }
    else
    {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width * sf);
        h = cvRound(imageSize.height * sf);
        canvas.create(h * 2, w, CV_8UC3);
    }
    for (i = 0; i < nimages; i++) {
        for (k = 0; k < 2; k++) {
            Mat img = imread(goodImagelist[2 * i + k], 0);
            Mat rimg, cimg;
            //The function remap transforms the source image using the specified map:
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w * k, 0, w, h)) : canvas(Rect(0, h * k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            /*if (useCalibrated)
            {
                Rect vroi(cvRound(validRoi[k].x * sf), cvRound(validRoi[k].y * sf),
                    cvRound(validRoi[k].width * sf), cvRound(validRoi[k].height * sf));
                rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
            }*/
        }
        if (!isVerticalStereo)
            for (j = 0; j < canvas.rows; j += 16)
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for (j = 0; j < canvas.cols; j += 16)
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        waitKey(500);
    }


}

int main(int argc, char** argv)
{
    Size boardSize = Size(9, 6);//标定板上每行每列的角点数
    string imagelistfn;//图片列表文件名称
    bool showRectified;//显示被修改的
    vector<string> imagelist;//图片列表
    fstream fin("images.txt");
    string filename;
    int imgCount = 0;
    float squareSize = 1.0;
    while (getline(fin, filename)) {
        imgCount++;
        cout << "------imageCount------" << imgCount << "    " << filename << endl;
        imagelist.push_back(filename);
    }
    if (imagelist.empty()) {
        cout << "the string list is empty." << endl;
        return 0;
    }
    StereoCalib(imagelist, boardSize, squareSize, true, true, true);//相机标定
    return 0;
}