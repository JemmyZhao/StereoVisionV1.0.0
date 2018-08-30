#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

#define IMG_NUM 5

static void calObjectPosition(Size boardSize, float squareSize, vector<Point3f>& corners);
double ch_cameraCalibration(String* imageName, int imageNum, Size boardSize, float squareSize, 
	                        Mat& cameraMatrix, Mat& distCoeffs, Mat& map1, Mat& map2);
void ch_undistort(InputArray src, OutputArray dst, InputArray map1, InputArray map2);

int getPics();

void testCalibration()
{
	Size boardSize = Size(9, 6);
	float squareSize = 50.0f;

	String img_names[IMG_NUM] = { "F:/WorkSpace_OpenCV/data/left01.jpg" ,
								  "F:/WorkSpace_OpenCV/data/left02.jpg" ,
								  "F:/WorkSpace_OpenCV/data/left03.jpg" ,
								  "F:/WorkSpace_OpenCV/data/left04.jpg" ,
								  "F:/WorkSpace_OpenCV/data/left05.jpg" };


	Mat cameraMatrix;
	Mat distCoeffs;
	Mat map1, map2;
	ch_cameraCalibration(img_names, 4, boardSize, squareSize, cameraMatrix, distCoeffs, map1, map2);

	Mat v = imread(img_names[3]);
	Mat v1;
	ch_undistort(v, v1, map1, map2);
	imshow("distored", v1);
	waitKey(100);

	//PnP
	vector<Point3f> objPoints;
	calObjectPosition(boardSize, squareSize, objPoints);
	vector<Point2f> imgPoints;
	v = imread(img_names[0]);
	findChessboardCorners(v, boardSize, imgPoints, 0);
	Mat rvec, tvec;

	int t0, t1;
	double t;
	//Pnp
	t0 = getTickCount();
	solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, 0, CV_ITERATIVE);
	t1 = getTickCount();
	t = ((double)(t1 - t0)) / getTickFrequency();
	cout << "iterative time: "<< t <<endl << "rvec\n" << rvec <<endl<< "tvec\n" << tvec << endl<<endl;

	t0 = getTickCount();
	solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, 0, CV_EPNP);
	t1 = getTickCount();
	t = ((double)(t1 - t0)) / getTickFrequency();
	cout << "epnp time: " << t << endl << "rvec\n" << rvec << endl << "tvec\n" << tvec << endl<<endl;

	t0 = getTickCount();
	solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec, 0, CV_DLS);
	t1 = getTickCount();
	t = ((double)(t1 - t0)) / getTickFrequency();
	cout << "dls time: " << t << endl << "rvec\n" << rvec << endl << "tvec\n" << tvec << endl<<endl;

	waitKey();


}

int main()
{
	getPics();
}

int getPics()
{
	VideoCapture capture(0); // open the first camera
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	//capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720.0);

	if (!capture.isOpened())
	{
		cerr << "ERROR: Can't initialize camera capture" << endl;
		return 1;
	}

	/*capture.set(CV_CAP_PROP_FORMAT, 980);*/
	
	cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << endl;
	cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << endl;
	cout << "Format: " << capture.get(CAP_PROP_FORMAT) << endl;

	cout << endl << "Press 'ESC' to quit, 'space' to toggle frame processing" << endl;
	cout << endl << "Start grabbing..." << endl;
	Mat v;

	int img_cnt = 0;
	String dir_left = "F:/WorkSpace_OpenCV/data/stereo/left/";
	String dir_right = "F:/WorkSpace_OpenCV/data/stereo/right/";

	while (true)
	{
		capture.read(v);
		imshow("v", v);
		Mat v1(v, Rect(0, 0, 640, 480));
		Mat v2(v, Rect(640, 0, 640, 480));

		char k = waitKey(20);
		if (k == 'q')
		{
			break;
		}
		else if (k == 'p')
		{
			imwrite(dir_left + "left" + to_string(img_cnt) + ".jpg", v1);
			imwrite(dir_right + "right" + to_string(img_cnt) + ".jpg", v2);
			cout << "Write Images done" << endl;
			img_cnt++;
		}
	}
	capture.release();

	return 0;

}

double ch_cameraCalibration(String* imageName, int imageNum, Size boardSize, float squareSize, Mat& cameraMatrix, Mat& distCoeffs, Mat& map1, Mat& map2)
{
	vector<vector<Point2f>> imagePoints;
	vector<Mat> views;

	//Collecting Points
	for (int i = 0; i < imageNum; i++)
	{
		vector<Point2f> pointBuf;
		Mat v;

		v = imread(imageName[i]);
		views.push_back(v);
		//Finding corners
		bool found;
		found = findChessboardCorners(v, boardSize, pointBuf, 0);

		//Draw corners
		if (found)
		{
			Mat viewGray;
			cvtColor(v, viewGray, COLOR_BGR2GRAY);
			cornerSubPix(viewGray, pointBuf, Size(11, 11),
				Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
			imagePoints.push_back(pointBuf);
			drawChessboardCorners(v, boardSize, Mat(pointBuf), found);
		}
		imshow("org", v);
		waitKey(50);
	}
	Size imageSize = views[0].size();

	vector<vector<Point3f>> objectPoints(1);
	calObjectPosition(boardSize, squareSize, objectPoints[0]);
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	//Call opencv function
	double rms;
	vector<Mat> rvecs, tvecs;
	rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

	cout << "Camera Matrix:\n" << cameraMatrix << endl;

	Mat v2;

	//Undistored
	initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize,
		CV_16SC2, map1, map2);

	for (int i = 0; i < imageNum; i++)
	{
		remap(views[i], v2, map1, map2, INTER_LINEAR);
		imshow("New View", v2);
		waitKey(50);
	}

	return rms;
}

void ch_undistort(InputArray src, OutputArray dst, InputArray map1, InputArray map2)
{
	remap(src, dst, map1, map2, INTER_LINEAR);
}

static void calObjectPosition(Size boardSize, float squareSize, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(j*squareSize, i*squareSize, 0));
}