#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;
using namespace dnn;
CascadeClassifier face_cascade;
CascadeClassifier body_cascade;
void detectFace(Mat &frame, Mat &grayFrame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);

	for (size_t i = 0; i < faces.size(); i++)
	{
		rectangle(frame, faces[i], cv::Scalar(255, 0, 0), -1);
	}
	grayFrame = frame_gray;
}
void detectUpperbody(Mat &frame, Mat &grayFrame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	std::vector<Rect> bodies;
	body_cascade.detectMultiScale(frame_gray, bodies);

	for (size_t i = 0; i < bodies.size(); i++)
	{
		rectangle(frame_gray, bodies[i], cv::Scalar(0, 255, 0));
		// Mat selectedBody = frame_gray(bodies[i]);
		// frame_gray = selectedBody;
	}
	grayFrame = frame_gray;
}
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
	Mat eroded, out;
	erode(Binary, eroded, element);
	dilate(eroded, out, element);
	// return out;
	return out;
}
Mat getSkin3(Mat &ImageIn)
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
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
	Mat eroded, out;
	// erode(Binary, eroded, element);
	// dilate(eroded, out, element);
	// return out;
	return Binary;
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
	Mat out, out2;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
	erode(Binary, out, element);
	dilate(out, out2, element);
	return out2;
	// return Binary;
}

vector<float> GetFourier(Mat &ImageIn, Point &pLeft, Point &pRight)
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
	float xmax = INT_MIN, xmin = INT_MAX, ymax = INT_MIN, ymin = INT_MAX;
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
			if (xmax < x)
				xmax = x;
			if (xmin > x)
				xmin = x;
			if (ymax < y)
				ymax = y;
			if (ymin > y)
				ymin = y;
		}
		f.push_back(sqrt((sumx * sumx) + (sumy * sumy)));
	}
	Point p1(xmin, ymin);
	Point p2(xmax, ymax);
	pLeft = p1;
	pRight = p2;

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

void TrainKNNModel()
{
	int K = 4;
	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
	knn->setDefaultK(K);
	knn->setIsClassifier(true);
	knn->setAlgorithmType(cv::ml::KNearest::KDTREE);

	vector<string> featureList = {"One", "Two", "Three", "Fist"};
	// vector<vector<float>> trainData;
	Mat trainData(4 * 300, 14, CV_32FC1);
	// vector<float> trainLabels;
	Mat trainLabels(4 * 300, 1, CV_32FC1);
	for (size_t i = 0; i < featureList.size(); i++)
	{
		FILE *fp = fopen(const_cast<char *>((featureList[i] + ".fea").c_str()), "rb");
		fseek(fp, 0, SEEK_END);
		long len = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		char buff[len];
		fread(buff, len, 1, fp);
		int l = 0;
		while (l < len)
		{
			// vector<float> vec;
			for (int j = 0; j < 14; j++)
			{
				char buf[4];
				for (int k = 0; k < 4; k++)
				{
					buf[k] = buff[l + k];
				}
				float floatVariable;
				void *pf;
				pf = &floatVariable;
				for (char i = 0; i < 4; i++)
				{
					*((unsigned char *)pf + i) = buf[i];
				}
				cout << floatVariable << " ";
				// vec.push_back(floatVariable);
				trainData.at<float>(i * 300 + l / 56, (l % 56) / 4) = floatVariable;

				l += 4;
			}
			cout << "\n";
			trainLabels.at<float>(i * 300 + l / 56, 0) = i;
		}
	}
	cout << "Size of trainData:" << trainData.size() << endl;
	cout << "Size of trainLabels:" << trainLabels.size() << endl;

	knn->train(trainData, ml::ROW_SAMPLE, trainLabels);
	knn->save("trainedKNN");
}
void getAnnXML()
{
	FileStorage fs("ann_xml.xml", FileStorage::WRITE);
	if (!fs.isOpened())
	{
		cout << "failed to open "
			 << "/n";
	}
	Mat trainData(3 * 300, 12, CV_32F);
	// Mat trainData;							   //用于存放样本的特征数据
	Mat classes(3 * 300, 1, CV_32F); //用于标记是第几类手势
	char path[60];					 //样本路径
	Mat Image_read;					 //读入的样本
	vector<string> featureList = {"One", "Two", "Three"};
	for (int i = 0; i < 3; i++) //第i类手势 比如手势1、手势2
	{
		FILE *fp = fopen(const_cast<char *>(("/Users/harrywu/Documents/gesture/" + featureList[i] + ".fea").c_str()), "rb");
		fseek(fp, 0, SEEK_END);
		long len = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		char buff[len];
		fread(buff, len, 1, fp);
		for (int j = 1; j < 301; j++) //每个手势设置50个样本
		{
			int l = 0;
			while (l < len)
			{
				// vector<float> vec;
				for (int j = 0; j < 12; j++)
				{
					char buf[4];
					for (int k = 0; k < 4; k++)
					{
						buf[k] = buff[l + k];
					}
					float floatVariable;
					void *pf;
					pf = &floatVariable;
					for (char i = 0; i < 4; i++)
					{
						*((unsigned char *)pf + i) = buf[i];
					}
					trainData.at<float>(i * 300 + l / 56, (l % 56) / 4) = floatVariable;

					l += 4;
				}
				classes.at<float>(i * 300 + l / 56, 0) = i;
			}

			classes.at<uchar>(i * 50 + j - 1) = i;
		}
		fclose(fp);
	}
	fs << "TrainingData" << trainData;
	fs << "classes" << classes;
	fs.release();

	cout << "训练矩阵和标签矩阵搞定了！" << endl;
}
void ann_train(Ptr<ml::ANN_MLP> &ann, int numCharacters, int nlayers) //神经网络训练函数，numCharacters设置为4，nlayers设置为24
{
	Mat trainData, classes;
	FileStorage fs;
	fs.open("ann_xml.xml", FileStorage::READ);

	fs["TrainingData"] >> trainData;
	fs["classes"] >> classes;

	Mat layerSizes(1, 3, CV_32SC1);			// 3层神经网络
	layerSizes.at<int>(0) = trainData.cols; //输入层的神经元结点数，设置为15
	layerSizes.at<int>(1) = nlayers;		// 1个隐藏层的神经元结点数，设置为24
	layerSizes.at<int>(2) = numCharacters;	//输出层的神经元结点数为:4

	ann->setLayerSizes(layerSizes);
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.1, 0.1); //后两个参数： 权梯度项的强度（一般设置为0.1） 动量项的强度（一般设置为0.1）
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5000, 0.01)); //后两项参数为迭代次数和误差最小值

	Mat trainClasses; //用于告诉神经网络该特征对应的是什么手势
	trainClasses.create(trainData.rows, numCharacters, CV_32FC1);
	for (int i = 0; i < trainData.rows; i++)
	{
		for (int k = 0; k < trainClasses.cols; k++)
		{
			if (k == (int)classes.at<uchar>(i))
			{
				trainClasses.at<float>(i, k) = 1;
			}
			else
				trainClasses.at<float>(i, k) = 0;
		}
	}
	// Mat weights(1 , trainData.rows , CV_32FC1 ,Scalar::all(1) );
	ann->train(trainData, ml::ROW_SAMPLE, trainClasses);

	cout << " 训练完了！ " << endl;
}

int classify(Ptr<ml::ANN_MLP> &ann, Mat &feature) //预测函数，找到最符合的一个手势(输入的图片是二值化的图片)
{
	int result = -1;

	Mat output(1, 4, CV_32FC1); // 1*4矩阵

	ann->predict(feature, output);

	Point maxLoc;
	double maxVal;
	minMaxLoc(output, 0, &maxVal, 0, &maxLoc); //对比值，看哪个是最符合的。

	result = maxLoc.x;

	return result;
}

int main()
{
	// Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();
	// ann_train(ann, 3, 24);
	// ann->save("bp_gesture");
	// return 0;
	int K = 20;
	// Ptr<ml::KNearest> knn = ml::KNearest::load("trainedKNN.yaml");
	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
	knn->setDefaultK(K);
	knn->setIsClassifier(true);
	knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);

	vector<string> featureList = {"One", "Two", "Three", "Four", "Five"};
	Mat trainData(5 * 400, 12, CV_32F);
	Mat trainLabels(5 * 400, 1, CV_32F);
	for (size_t i = 0; i < featureList.size(); i++)
	{
		FILE *fp = fopen(const_cast<char *>(("/Users/harrywu/Documents/gesture/" + featureList[i] + ".fea").c_str()), "rb");
		fseek(fp, 0, SEEK_END);
		long len = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		char buff[len];
		fread(buff, len, 1, fp);
		int l = 0;
		while (l < len)
		{
			for (int j = 0; j < 12; j++)
			{
				char buf[4];
				for (int k = 0; k < 4; k++)
				{
					buf[k] = buff[l + k];
				}
				float floatVariable;
				void *pf;
				pf = &floatVariable;
				for (char i = 0; i < 4; i++)
				{
					*((unsigned char *)pf + i) = buf[i];
				}
				// cout << floatVariable << " ";
				//  vec.push_back(floatVariable);
				trainData.at<float>(i * 400 + l / 48, (l % 48) / 4) = floatVariable;

				l += 4;
			}
			// cout << "\n";
			trainLabels.at<float>(i * 400 + l / 48, 0) = i;
		}
		fclose(fp);
	}
	cout << "Size of trainData:" << trainData.size() << endl;
	cout << "Size of trainLabels:" << trainLabels.size() << endl;
	knn->train(trainData, ml::ROW_SAMPLE, trainLabels);
	// knn->save("trainedKNN");

	// Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::load("bp_gesture");
	String face_cascade_name = samples::findFile("/Users/harrywu/Documents/gesture/data/haarcascades/haarcascade_frontalface_alt.xml");

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	vector<vector<float>> features;
	vector<float> labels;

	namedWindow("camera", WINDOW_NORMAL);
	namedWindow("binary", WINDOW_NORMAL);
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		cout << "Camera not opened!\n";
	}
	Mat frame, gray;
	int count = 0;
	while (1)
	{
		cap >> frame;
		if (frame.empty())
			break;
		Mat frameWithoutFace = frame.clone();
		detectFace(frameWithoutFace, gray);
		Mat out = getSkin(frameWithoutFace);
		imshow("binary", out);

		Point p1, p2;
		vector<float> fft = GetFourier(out, p1, p2);
		rectangle(frame, p1, p2, Scalar(0, 0, 255), 2);

		Mat input(1, 12, CV_32F);
		for (size_t j = 1; j < 13; j++)
		{
			input.at<float>(j - 1) = fft[j];
		}
		// int res = classify(ann, input);
		Mat output, neighbourRes, dist;
		knn->findNearest(input, K, output, neighbourRes, dist);
		std::cout << "Predict:" << output.at<float>(0) << endl;
		Point recp1 = p1, recp2 = p1;
		recp1.y -= 30;
		recp2.x += 80;

		rectangle(frame, recp1, recp2, Scalar(0, 0, 255), -1);
		putText(frame, to_string((int)output.at<float>(0) + 1), p1, cv::FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255), 2);
		imshow("camera", frame);
		if (waitKey(20) > 0)
		{
			break;
		}
	}
	cap.release();
	return 0;
	// for (size_t i = 0; i < featureList.size(); i++)
	// {
	// 	FILE *fpr = fopen(const_cast<char *>((featureList[i] + ".fea").c_str()), "rb");

	// 	fseek(fpr, 0, SEEK_END);
	// 	long len = ftell(fpr);
	// 	fseek(fpr, 0, SEEK_SET);
	// 	char buff[len];
	// 	fread(buff, len, 1, fpr);
	// 	int l = 0;
	// 	while (l < len)
	// 	{
	// 		vector<float> vec;
	// 		for (int i = 0; i < 14; i++)
	// 		{
	// 			char buf[4];
	// 			for (int j = 0; j < 4; j++)
	// 			{
	// 				buf[j] = buff[l + j];
	// 			}
	// 			float floatVariable;
	// 			void *pf;
	// 			pf = &floatVariable;
	// 			for (char i = 0; i < 4; i++)
	// 			{
	// 				*((unsigned char *)pf + i) = buf[i];
	// 			}
	// 			// std::cout << floatVariable << " ";
	// 			vec.push_back(floatVariable);
	// 			l += 4;
	// 		}
	// 		// std::cout << "\n";
	// 		features.push_back(vec);
	// 		labels.push_back(i);
	// 	}
	// 	fclose(fpr);
	// }
	// std::cout << "Size of features:" << features.size() << endl;
	// std::cout << "Size of labels:" << labels.size() << endl;
	// vector<pair<vector<float>, float>> testSet;
	// // time_t t;
	// // localtime(&t);
	// // srand(t);

	// for (size_t i = 0; i < 100; i++)
	// {
	// 	int num = rand() % 1200;
	// 	testSet.push_back(make_pair(features[num], labels[num]));
	// }
	// int correct = 0;
	// for (size_t i = 0; i < testSet.size(); i++)
	// {
	// 	Mat input(1, 14, CV_32F);
	// 	for (size_t j = 0; j < 14; j++)
	// 	{
	// 		input.at<float>(j) = testSet[i].first[j];
	// 	}

	// 	Mat output, neighbourRes, dist;
	// 	knn->findNearest(input, K, output, neighbourRes, dist);
	// 	// std::cout<<input.size()<<" "<<output.size()<<" "<<neighbourRes.size()<<" "<<dist.size()<<endl;

	// 	std::cout << "Predict:" << output.at<float>(0) << "  Actual:" << testSet[i].second << "\n";
	// 	if (output.at<float>(0) == testSet[i].second)
	// 		correct++;
	// }
	// std::cout << "Correct:" << correct << endl;
	// std::cout << "Accuracy:" << (float)correct / (float)100 << endl;
	// return 0;

	/*
	FILE *fpr = fopen("Features.fea", "rb");
	vector<vector<float>> features;
	fseek(fpr, 0, SEEK_END);
	long len = ftell(fpr);
	fseek(fpr, 0, SEEK_SET);
	char buff[len];
	fread(buff, len, 1, fpr);
	int l = 0;
	while (l < len)
	{
		vector<float> vec;
		for (int i = 0; i < 14; i++)
		{
			char buf[4];
			// fread(buf,4,1,fp);
			for (int j = 0; j < 4; j++)
			{
				buf[j] = buff[l + j];
			}
			// float dat;
			// dat = *((float *)buf);
			float floatVariable;
			void *pf;
			pf = &floatVariable;
			for (char i = 0; i < 4; i++)
			{
				*((unsigned char *)pf + i) = buf[i];
			}
			cout << floatVariable << " ";
			vec.push_back(floatVariable);
			l += 4;
		}
		cout << "\n";
		features.push_back(vec);
	}
	*/
	//转换字节数组到float数据
	// float floatVariable;
	// void *pf;
	// pf = &floatVariable;
	// for (char i = 0; i < 4; i++)
	// {
	//     *((unsigned char *)pf + i) = str[i];
	// }
	return 0;
}