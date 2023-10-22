#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string.h>
#include <fstream>

using namespace std;
using namespace cv;
using namespace dnn;
Mat getSkin(Mat &ImageIn)
{
	Mat Image = ImageIn.clone();
	Mat ycrcb_Image;
	cvtColor(Image, ycrcb_Image, COLOR_BGR2YCrCb); //转换色彩空间

	vector<Mat> y_cr_cb;
	split(ycrcb_Image, y_cr_cb); //分离YCrCb

	Mat CR = y_cr_cb[1]; //图片的CR分量
	Mat CR1;

	Mat Binary = Mat::zeros(Image.size(), CV_8UC1);
	GaussianBlur(CR, CR1, Size(3, 3), 0, 0);	 //对CR分量进行高斯滤波，得到CR1（注意这里一定要新建一张图片存放结果）
	threshold(CR1, Binary, 0, 255, THRESH_OTSU); //用系统自带的threshold函数，对CR分量进行二值化，算法为自适应阈值的OTSU算法

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
	Mat out, out2;
	erode(Binary, out, element);
	dilate(out, out2, element);
	return out2;
}
Mat getSkin2(Mat &ImageIn)
{
	Mat Image = ImageIn.clone(); //复制输入的图片

	//利用OPENCV自带的ellipse函数生成一个椭圆的模型
	Mat skinCrCbHist = Mat::zeros(Size(256, 256), CV_8UC1);
	ellipse(skinCrCbHist, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);

	Mat ycrcb_Image;
	cvtColor(Image, ycrcb_Image, COLOR_BGR2YCrCb); //用cvtColor函数将图片转换为YCrCb色彩空间的图片

	Mat Binary = Mat::zeros(Image.size(), CV_8UC1); //输出的二值化图片
	vector<Mat> y_cr_cb;							//用于存放分离开的YCrCb分量
	split(ycrcb_Image, y_cr_cb);					//分离YCrCb分量，顺序是Y、Cr、Cb

	Mat CR = y_cr_cb[1];
	Mat CB = y_cr_cb[2];

	for (int i = 0; i < Image.rows; i++)
	{
		for (int j = 0; j < Image.cols; j++)
		{
			if (skinCrCbHist.at<uchar>(CR.at<uchar>(i, j), CB.at<uchar>(i, j)) > 0) //在椭圆内的点置为255
			{
				Binary.at<uchar>(i, j) = 255;
			}
		}
	}
	return Binary;
}

vector<float> GetFourier(Mat &ImageIn)
{
	Mat ImageBinary = ImageIn;		//二值化的图片（传入时应该就已经是二值化的图片了）
	vector<vector<Point>> contours; //定义轮廓向量

	findContours(ImageBinary, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE); //寻找轮廓

	int max_size = 0;						  //最大的轮廓的size（通常来说就是我们的手势轮廓）
	int contour_num = 0;					  //最大的轮廓是图片中的第contour_num个轮廓
	for (int i = 0; i < contours.size(); i++) //找到最大的轮廓，记录在contour_num中
	{
		if (contours[i].size() > max_size)
		{
			max_size = contours[i].size();
			contour_num = i;
		}
	}
	/***计算图像的傅里叶描绘子***/
	/***傅里叶变换后的系数储存在f[d]中***/
	vector<float> f;
	vector<float> fd; //最终傅里叶描绘子前14位
	Point p;
	for (int i = 0; i < max_size; i++) //主要的计算部分
	{
		float x, y, sumx = 0, sumy = 0;
		for (int j = 0; j < max_size; j++)
		{
			p = contours[contour_num].at(j);
			x = p.x;
			y = p.y;
			sumx += (float)(x * cos(2 * CV_PI * i * j / max_size) + y * sin(2 * CV_PI * i * j / max_size));
			sumy += (float)(y * cos(2 * CV_PI * i * j / max_size) - x * sin(2 * CV_PI * i * j / max_size));
		}
		f.push_back(sqrt((sumx * sumx) + (sumy * sumy)));
	}

	fd.push_back(0); //放入了标志位‘0’，并不影响最终结果

	for (int k = 2; k < 14; k++) //进行归一化，然后放入最终结果中
	{
		f[k] = f[k] / f[1];
		fd.push_back(f[k]);
	}

	// Mat out = Mat::zeros(1, fd.size(), CV_32F);//out是用于输出的手势特征
	vector<float> out;
	for (int i = 0; i < fd.size(); i++)
	{
		out.push_back(fd[i]);
	}
	return out;
}
int main(int argc, char **argv)
{
	namedWindow("camera", WINDOW_NORMAL);
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		cout << "Camera not opened!\n";
	}
	Mat frame;
	int count = 0;
	FILE *fp = fopen(argv[1], "wb");
	char *data = (char *)malloc(1024 * 1024);
	long cur = 0;
	bool start = false;
	while (1)
	{
		cap >> frame;
		if (frame.empty())
			break;
		Mat out = getSkin(frame);
		imshow("video", out);

		vector<float> fft = GetFourier(out);

		for (int i = 1; i < 13; i++)
		{
			cout << fft[i] << " ";
			float value = fft[i];
			if (start)
			{
				memcpy(data + cur, &value, 4);
				cout << "Start recording count=" << count << "\n";
			}

			cur += 4;
		}
		if (start)
			count++;
		cout << endl;
		if (waitKey(20) == 32)
			start = true;
		if (waitKey(20) == 27 || count == 400)
		{
			fwrite(data, cur, 1, fp);
			fclose(fp);
			free(data);
			break;
		}
	}
	cap.release();
	return 0;
}