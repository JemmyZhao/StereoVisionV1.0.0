#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main()
{

	VideoCapture c0,c1;
	c0.open(0, CAP_DSHOW);
	c1.open(1, CAP_DSHOW);
	waitKey(2000);
	if (!c0.isOpened())
	{
		cout << "Can not open Camera 0" << endl;
		return 1;
	}
	if (!c1.isOpened())
	{
		cout << "Can not open Camera 1" << endl;
		return 1;
	}


	Mat f0, f1;

	for (int i = 0; i < 1000; i++)
	{
		c0.read(f0);
		c1.read(f1);

		imshow("f0", f0);
		waitKey(100);
		imshow("f1", f1);
		waitKey(10);
	}
	
	c0.release();
	c1.release();

	return 0;
}