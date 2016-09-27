#include<opencv2/opencv.hpp>
#include<opencv2/gpu/gpu.hpp>
#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>

#define FALSE -1
#define TRUE 0

using namespace cv;
using namespace std;

#define T_ANGLE_THRE 10
#define T_SIZE_THRE 5

#define D_BRIGHT -120
#define N_THRE 25
enum ch{ BLUE, GREEN, RED };
struct Buffer{
	gpu::GpuMat gI;
	gpu::GpuMat subMat, tmp;
	vector<gpu::GpuMat> gchannels;
};

vector<RotatedRect> ArmorDetect(vector<RotatedRect> &vEllipse)
{
	vector<RotatedRect> vRlt;
	RotatedRect Armor;
	int nL, nW;
	double dAngle;
	vRlt.clear();
	if (vEllipse.size() < 2)
		return vRlt;

	for (unsigned int nI = 0; nI < vEllipse.size() - 1; ++nI)
	{
		for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); ++nJ)
		{
			dAngle = abs(vEllipse[nI].angle - vEllipse[nJ].angle);
			while (dAngle > 180)
				dAngle -= 180;
			if ((dAngle < T_ANGLE_THRE || 180 - dAngle < T_ANGLE_THRE) &&
				abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) < (vEllipse[nI].size.height + vEllipse[nJ].size.height) / T_SIZE_THRE&&
				abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) < (vEllipse[nI].size.width + vEllipse[nJ].size.width) / T_SIZE_THRE)
			{
				Armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2;
				Armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2;
				Armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;
				if (180 - dAngle < T_ANGLE_THRE)
					Armor.angle += 90;
				nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2;
				nW = sqrt((vEllipse[nI].center.x - vEllipse[nJ].center.x)*(vEllipse[nI].center.x - vEllipse[nJ].center.x) +
					(vEllipse[nI].center.y - vEllipse[nJ].center.y)*(vEllipse[nI].center.y - vEllipse[nJ].center.y));
				if (nL < nW)
				{
					Armor.size.height = nL;
					Armor.size.width = nW;
				}
				else {
					Armor.size.height = nW;
					Armor.size.width = nL;
				}
				vRlt.push_back(Armor);
			}
		}
	}
	return vRlt;
}


void DrawBox(RotatedRect box, Mat &img,Scalar s)
{
	Point2f vertex[4];
	for (int i = 0; i < 4; ++i)
	{
		vertex[i].x = 0;
		vertex[i].y = 0;
	}
	box.points(vertex);
	for (int i = 0; i < 4; ++i)
	{
		line(img, vertex[i], vertex[(i + 1) % 4], s, 2);
	}
}
Point FindCenter(vector<RotatedRect> &vEllipse);
bool isHere(Point &pt, Rect &rt);
int main()
{
//	VideoCapture cap("/home/ubuntu/Videos/RedCar.MOV");
    VideoCapture cap(1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 960);
	if (!cap.isOpened()) {
		cout << "Cannot open the camera!!!" << endl;
		return -1;
	}
	Mat frame;
	Buffer bf;
	double time;
	gpu::Stream stream;

	Mat element = getStructuringElement(MORPH_RECT, Size(1, 1), Point(-1, -1));
	vector<vector<Point> > contours;
	Mat rlt;
	bool bFlag = false;
	RotatedRect s;
	Vec3b s1;
	vector<RotatedRect> vEllipse;

	//last count
	int count = 0;
	//last state
	unsigned int lastState;

	bool yFlag = false;

	Point center;
	char buf[6] = "0LURD";

	bool isInit = false;
	Rect centerArea;
	int chs;
	int rows;

	/**/
	vector<RotatedRect> vRlt;


	while (true) {
		time = (double)getTickCount();

		bool isOK = cap.read(frame);

		cout << frame.cols << " " << frame.rows << endl;
		if (!isOK) {
			cout << "Cannot read a frame from video stream!!!" << endl;
			break;
		}
		if (!isInit){
			chs = frame.channels();
			rows = frame.rows;
			centerArea = Rect(Point(2 * frame.cols / 5, 2 * frame.rows / 5), Point(3 * frame.cols / 5, 3 * frame.rows / 5));
		}
		bf.gI.upload(frame);
		bf.gI = bf.gI.reshape(1);
		gpu::multiply(bf.gI, 1, bf.gI, 1.0, -1, stream);
		gpu::add(bf.gI, D_BRIGHT, bf.gI, gpu::GpuMat(), -1, stream);
		bf.gI = bf.gI.reshape(chs, rows);
		gpu::split(bf.gI, bf.gchannels, stream);
		gpu::subtract(bf.gchannels[RED], bf.gchannels[GREEN], bf.subMat, gpu::GpuMat(), -1, stream);
		gpu::threshold(bf.subMat, bf.subMat, N_THRE, 255, CV_THRESH_BINARY, stream);
		gpu::dilate(bf.subMat, bf.tmp, element, Point(-1, -1), 3);
		gpu::erode(bf.tmp, bf.subMat, element, Point(-1, -1), 1);
		bf.subMat.download(rlt);
		imshow("RLT", rlt);
		findContours(rlt, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		for (unsigned int i = 0; i < contours.size(); ++i)
		{
			if (contours[i].size()>10) {
				bFlag = true;
				s = fitEllipse(contours[i]);
				for (int nI = 0; nI < 5; ++nI)
					for (int nJ = 0; nJ < 5; ++nJ)
					{
						if (s.center.y - 2 + nJ>0 && s.center.y - 2 + nJ < frame.rows && s.center.x - 2 + nI>0 && s.center.x - 2 + nI < frame.cols)
						{
							s1 = frame.at<Vec3b>((int)(s.center.y - 2 + nJ), (int)(s.center.x - 2 + nI));
							if (s1.val[0] < 200 || s1.val[1] < 200 || s1.val[2] < 200)
								bFlag = false;
						}
					}
				if (bFlag)
					vEllipse.push_back(s);
			}
		}
		for (unsigned int nI = 0; nI < vEllipse.size(); ++nI)
			DrawBox(vEllipse[nI], frame,Scalar(0,0,255));

		/**/
		vRlt = ArmorDetect(vEllipse);

		for (unsigned int nI = 0; nI < vRlt.size(); ++nI)
			DrawBox(vRlt[nI], frame,Scalar(0,255,0));

		if (vRlt.empty()){
			if (yFlag){
				cout << buf[lastState] << endl;
				++count;
				if (count == 5)
					yFlag = false;
			}
			else{
				cout << "1" << endl;
			}
			contours.clear();
			vEllipse.clear();
			vRlt.clear();

			imshow("Raw", frame);
			time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
			cout << "CPU runs:" << time << endl;

			if (waitKey(1) == 27)
				break;
			continue;
		}
		center = FindCenter(vRlt);

		line(frame, Point(0, center.y), Point(frame.cols, center.y), Scalar(0), 3);
		line(frame, Point(center.x, 0), Point(center.x, frame.rows), Scalar(0), 3);

		if (isHere(center, centerArea)) {
			cout << "0" << endl;
			lastState = 0;
		}
		else {
			if (center.x > centerArea.x + centerArea.width){
				cout << "R" << endl;
				lastState = 2;
			}
			if (center.x < centerArea.x){
				cout << "L" << endl;
				lastState = 1;
			}
			if (center.y > centerArea.y + centerArea.height){
				cout << "D" << endl;
				lastState = 4;
			}
			if (center.y < centerArea.y){
				cout << "U" << endl;
				lastState = 3;
			}
		}
		yFlag = true;
		count = 0;
		contours.clear();
		vEllipse.clear();
		vRlt.clear();

		imshow("Raw", frame);
		time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
		cout << "CPU runs:" << time << endl;

		//cir once,finish init
		isInit = true;
		if (waitKey(1) == 27)
			break;
	}
	return 0;
}
Point FindCenter(vector<RotatedRect> &vEllipse)
{
	Size2f rs;
	int s = 0;
	int j = 0;
	for (unsigned int i = 0; i < vEllipse.size(); ++i)
	{
		if (s < vEllipse[i].size.area()){
			s = vEllipse[i].size.area();
			j = i;
		}
	}
	return vEllipse[j].center;
}
bool isHere(Point &pt, Rect &rt)
{
	if (pt.x > rt.x&&pt.x<rt.x + rt.width&&pt.y>rt.y&&pt.y < rt.y + rt.height)
		return true;
	else
		return false;
}