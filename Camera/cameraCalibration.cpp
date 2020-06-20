#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <iostream>  
#include <fstream>
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;

int main() {
	ifstream fin("images.txt");
	ofstream fout("calibration_result.txt");
	cout << "---------开始提取角点---------" << endl;  // 读取每一幅图片，从中提取出角点，然后对角点进行亚像素精确化
	int imageCount = 0;//图像数量
	Size imageSize;//图像尺寸
	Size boardSize = Size(9, 6);//标定板上每行每列的角点数
	vector<Point2f> imagePointsBuf; //缓存每幅图像上检测到的角点数
	vector<vector<Point2f>> imagePointSeq;//保存检测到的所有角点数
	string fileName;
	int count = 0;//用于存储角点个数
	while (getline(fin,fileName))
	{
		imageCount++;
		cout << "------imageCount------" << imageCount<<"    "<<fileName << endl;
		Mat imageInput = imread(fileName);
		if (imageInput.empty()) {
			cout << "empty" << endl;
			continue;
		}
		imshow("Test", imageInput);
		if (imageCount == 1) {  //读取第一张图片时获取图像宽高信息
			imageSize.width = imageInput.cols;
			imageSize.height = imageInput.rows;
			cout << "imageSize.width = " << imageSize.width << endl;
			cout << "imageSize.height = " << imageSize.height << endl;
		}
		//发现棋盘角点
		if (0 == findChessboardCorners(imageInput, boardSize, imagePointsBuf)) {
			cout << "can not find chessboard corners! " << endl;
			return 0;
		}
		else
		{
			Mat viewGray;
			cvtColor(imageInput, viewGray, CV_BGR2GRAY);
			//亚像素精确化,Size(5,5)是搜索窗口的大小，Size(-1,-1)表示没有死区，第四个参数定义求角点的迭代过程的终止条件
			cornerSubPix(viewGray, imagePointsBuf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			count += imagePointsBuf.size();
			imagePointSeq.push_back(imagePointsBuf);
			//绘制被成功标定的角点
			drawChessboardCorners(viewGray, boardSize, imagePointsBuf, true);//false表示有未被探测到的内角点，这时候函数会以圆圈标记出检测到的内角点
			namedWindow("Camera Calibration", WINDOW_NORMAL);
			imshow("Camera Calibration", viewGray);
			waitKey(500);
		}
	}
	int total = imagePointSeq.size();//图片数量
	cout << "total = " << total << endl;
	int cornerNum = boardSize.height * boardSize.width; // 每张图片上的总角点数
	cout << "----图片的数据----" << endl;
	for (int i = 0; i < total; i++) {
		//输出第i张图片的第一个角点的坐标
		cout << "---" << imagePointSeq[i][0].x;
		cout << "---" << imagePointSeq[i][0].y << endl;
	}
	cout << endl << "角点提取完成" << endl;
	cout << "开始标定。。。" << endl;
	Size squareSize = Size(10, 10);//实际测量得到的标定板上每个棋盘格的大小
	vector<vector<Point3f>> objectPoints;//保存标定板上角点的三维坐标
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));//摄像机内参数矩阵
	vector<int> pointCount;//每幅图像中角点的数量
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));//摄像机的5个畸变系数
	vector<Mat> rvecs;//每幅图像的旋转向量
	vector<Mat> tvecs;//每幅图像的平移向量
	//初始化标定板上角点的三维坐标
	int i, j, t;
	for (t = 0; t < imageCount; t++) {
		vector<Point3f> tempPointSet;
		for (i = 0; i < boardSize.height; i++) {
			for (j = 0; j < boardSize.width; j++) {
				Point3f realPoint;
				//假设标定板放在世界坐标中z=0的平面上
				realPoint.x = i * squareSize.width;
				realPoint.y = j * squareSize.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		objectPoints.push_back(tempPointSet);
	}
	//初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板
	for (i = 0; i < imageCount; i++) {
		pointCount.push_back(boardSize.width * boardSize.height);
	}
	calibrateCamera(objectPoints, imagePointSeq, imageSize, cameraMatrix, distCoeffs, tvecs, rvecs);
	cout << "----标定结束，对标定结果进行评价----" << endl;
	double totalErr = 0.0;//所有图像的平均误差总和
	double err = 0.0;//每幅图像的平均误差
	vector<Point2f> imagePoints;//保存重新计算得到的投影点
	cout << "\t每幅图像的标定误差：\n";
	fout << "每幅图像的标定误差：\n";
	for (i = 0; i < imageCount; i++) {
		vector<Point3f> tempPointSet = objectPoints[i];
		//通过得到的内外参数，对空间的三维点进行重新计算，得到新的投影点
		projectPoints(tempPointSet, tvecs[i], rvecs[i], cameraMatrix, distCoeffs, imagePoints);
		vector<Point2f> tempImagePoint = imagePointSeq[i];//第i张图像的角点
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);//
		Mat imagePointsMat = Mat(1, imagePoints.size(), CV_32FC2);
		for (j = 0; j < tempImagePoint.size(); j++) {
			imagePointsMat.at<Vec2f>(0, j) = Vec2f(imagePoints[j].x, imagePoints[j].y);//重新投影后的
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);//之前投影的
		}
		err = norm(imagePointsMat, tempImagePoint, NORM_L2);
		err /= pointCount[i];
		totalErr += err;
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << totalErr / imageCount << "像素" << endl;
	fout << "总体平均误差：" << totalErr / imageCount << "像素" << endl << endl;
	
	cout << "开始保存标定结果。。。。。" << endl;
	Mat rotationMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));//保存每幅图像的旋转矩阵
	fout << "相机内参数矩阵：" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs << endl << endl;
	for (int i = 0; i < imageCount; i++) {
		fout << "第" << i + 1 << "幅图像的旋转向量" << endl;
		fout << rvecs[i] << endl;
		//将旋转向量转换为相应的旋转矩阵
		Rodrigues(rvecs[i], rotationMatrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotationMatrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << tvecs[i] << endl;
	}
	cout << "完成保存" << endl;
	fout << endl;
	// 显示标定结果
	Mat mapx = Mat(imageSize, CV_32FC1);
	Mat mapy = Mat(imageSize, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "保存矫正图像" << endl;
	string imageFileName;
	stringstream ss;
	for (int i = 1; i <= imageCount; i++) {
		cout << "Image " << i << " ......" << endl;
		initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, imageSize, CV_32FC1, mapx, mapy);//计算畸变映射
		ss.clear();
		imageFileName.clear();
		string filePath = "left";
		ss << setw(2) << setfill('0') << i;
		ss >> imageFileName;
		filePath += imageFileName;
		filePath += ".jpg";
		Mat imageSource = imread(filePath);
		Mat newImage = imageSource.clone();
		remap(imageSource, newImage, mapx, mapy, INTER_LINEAR);//把求得的映射应用到图像上
		imageFileName += "_d.jpg";
		imwrite(imageFileName, newImage);//保存矫正后的图片
		imshow("Original Image", imageSource);
		waitKey(500);
		imshow("Undistorted Image", newImage);
		waitKey(500);
	}
	fin.close();
	fout.close();

	return 0;
}
