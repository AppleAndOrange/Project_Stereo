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
	cout << "---------��ʼ��ȡ�ǵ�---------" << endl;  // ��ȡÿһ��ͼƬ��������ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��
	int imageCount = 0;//ͼ������
	Size imageSize;//ͼ��ߴ�
	Size boardSize = Size(9, 6);//�궨����ÿ��ÿ�еĽǵ���
	vector<Point2f> imagePointsBuf; //����ÿ��ͼ���ϼ�⵽�Ľǵ���
	vector<vector<Point2f>> imagePointSeq;//�����⵽�����нǵ���
	string fileName;
	int count = 0;//���ڴ洢�ǵ����
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
		if (imageCount == 1) {  //��ȡ��һ��ͼƬʱ��ȡͼ������Ϣ
			imageSize.width = imageInput.cols;
			imageSize.height = imageInput.rows;
			cout << "imageSize.width = " << imageSize.width << endl;
			cout << "imageSize.height = " << imageSize.height << endl;
		}
		//�������̽ǵ�
		if (0 == findChessboardCorners(imageInput, boardSize, imagePointsBuf)) {
			cout << "can not find chessboard corners! " << endl;
			return 0;
		}
		else
		{
			Mat viewGray;
			cvtColor(imageInput, viewGray, CV_BGR2GRAY);
			//�����ؾ�ȷ��,Size(5,5)���������ڵĴ�С��Size(-1,-1)��ʾû�����������ĸ�����������ǵ�ĵ������̵���ֹ����
			cornerSubPix(viewGray, imagePointsBuf, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			count += imagePointsBuf.size();
			imagePointSeq.push_back(imagePointsBuf);
			//���Ʊ��ɹ��궨�Ľǵ�
			drawChessboardCorners(viewGray, boardSize, imagePointsBuf, false);//false��ʾ��δ��̽�⵽���ڽǵ㣬��ʱ��������ԲȦ��ǳ���⵽���ڽǵ�
			namedWindow("Camera Calibration", WINDOW_NORMAL);
			imshow("Camera Calibration", viewGray);
			waitKey(500);
		}
	}
	int total = imagePointSeq.size();//ͼƬ����
	cout << "total = " << total << endl;
	int cornerNum = boardSize.height * boardSize.width; // ÿ��ͼƬ�ϵ��ܽǵ���
	cout << "----ͼƬ������----" << endl;
	for (int i = 0; i < total; i++) {
		//�����i��ͼƬ�ĵ�һ���ǵ������
		cout << "---" << imagePointSeq[i][0].x;
		cout << "---" << imagePointSeq[i][0].y << endl;
	}
	cout << endl << "�ǵ���ȡ���" << endl;
	cout << "��ʼ�궨������" << endl;
	Size squareSize = Size(10, 10);//ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С
	vector<vector<Point3f>> objectPoints;//����궨���Ͻǵ����ά����
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));//������ڲ�������
	vector<int> pointCount;//ÿ��ͼ���нǵ������
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));//�������5������ϵ��
	vector<Mat> rvecs;//ÿ��ͼ�����ת����
	vector<Mat> tvecs;//ÿ��ͼ���ƽ������
	//��ʼ���궨���Ͻǵ����ά����
	int i, j, t;
	for (t = 0; t < imageCount; t++) {
		vector<Point3f> tempPointSet;
		for (i = 0; i < boardSize.height; i++) {
			for (j = 0; j < boardSize.width; j++) {
				Point3f realPoint;
				//����궨���������������z=0��ƽ����
				realPoint.x = i * squareSize.width;
				realPoint.y = j * squareSize.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		objectPoints.push_back(tempPointSet);
	}
	//��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨��
	for (i = 0; i < imageCount; i++) {
		pointCount.push_back(boardSize.width * boardSize.height);
	}
	calibrateCamera(objectPoints, imagePointSeq, imageSize, cameraMatrix, distCoeffs, tvecs, rvecs);
	cout << "----�궨�������Ա궨�����������----" << endl;
	double totalErr = 0.0;//����ͼ���ƽ������ܺ�
	double err = 0.0;//ÿ��ͼ���ƽ�����
	vector<Point2f> imagePoints;//�������¼���õ���ͶӰ��
	cout << "\tÿ��ͼ��ı궨��\n";
	fout << "ÿ��ͼ��ı궨��\n";
	for (i = 0; i < imageCount; i++) {
		vector<Point3f> tempPointSet = objectPoints[i];
		//ͨ���õ�������������Կռ����ά��������¼��㣬�õ��µ�ͶӰ��
		projectPoints(tempPointSet, tvecs[i], rvecs[i], cameraMatrix, distCoeffs, imagePoints);
		vector<Point2f> tempImagePoint = imagePointSeq[i];//��i��ͼ��Ľǵ�
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);//
		Mat imagePointsMat = Mat(1, imagePoints.size(), CV_32FC2);
		for (j = 0; j < tempImagePoint.size(); j++) {
			imagePointsMat.at<Vec2f>(0, j) = Vec2f(imagePoints[j].x, imagePoints[j].y);//����ͶӰ���
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);//֮ǰͶӰ��
		}
		err = norm(imagePointsMat, tempImagePoint, NORM_L2);
		err /= pointCount[i];
		totalErr += err;
		cout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
	}
	cout << "����ƽ����" << totalErr / imageCount << "����" << endl;
	fout << "����ƽ����" << totalErr / imageCount << "����" << endl << endl;
	
	cout << "��ʼ����궨�������������" << endl;
	Mat rotationMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));//����ÿ��ͼ�����ת����
	fout << "����ڲ�������" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "����ϵ����\n";
	fout << distCoeffs << endl << endl;
	for (int i = 0; i < imageCount; i++) {
		fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		fout << rvecs[i] << endl;
		//����ת����ת��Ϊ��Ӧ����ת����
		Rodrigues(rvecs[i], rotationMatrix);
		fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		fout << rotationMatrix << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		fout << tvecs[i] << endl;
	}
	cout << "��ɱ���" << endl;
	fout << endl;
	// ��ʾ�궨���
	Mat mapx = Mat(imageSize, CV_32FC1);
	Mat mapy = Mat(imageSize, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "�������ͼ��" << endl;
	string imageFileName;
	stringstream ss;
	for (int i = 1; i <= imageCount; i++) {
		cout << "Image " << i << " ......" << endl;
		initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, imageSize, CV_32FC1, mapx, mapy);//�������ӳ��
		ss.clear();
		imageFileName.clear();
		string filePath = "left";
		ss << setw(2) << setfill('0') << i;
		ss >> imageFileName;
		filePath += imageFileName;
		filePath += ".jpg";
		Mat imageSource = imread(filePath);
		Mat newImage = imageSource.clone();
		remap(imageSource, newImage, mapx, mapy, INTER_LINEAR);//����õ�ӳ��Ӧ�õ�ͼ����
		imageFileName += "_d.jpg";
		imwrite(imageFileName, newImage);//����������ͼƬ
		imshow("Original Image", imageSource);
		waitKey(500);
		imshow("Undistorted Image", newImage);
		waitKey(500);
	}
	fin.close();
	fout.close();

	return 0;
}