#include <opencv2/highgui.hpp>

#include "Yolo.h"
#include "common.h"

using namespace cv;

void runningYoloV3();


int main(int argc, char** argv)
{
	runningYoloV3();

	return 0;
}


void runningYoloV3() {
	String modelPath = "./cfg/hsh_cloth.weights";
	String configPath = "./cfg/hsh_cloth.cfg";
	String classesFile = "./cfg/hsh_cloth.names";


	Yolo yolov3 = Yolo(modelPath, configPath, classesFile, true);
	yolov3.loadModel();


	VideoCapture cap;
	cap.open("007.avi");
	Mat frame;
	while (waitKey(1) < 0) {
		cap >> frame;
		if (frame.empty()) {
			waitKey();
			break;
		}

		double start_time = (double)cv::getTickCount();

		std::vector<YoloDetSt> yoloRet;
		std::vector<RegularDetect> regularDetect;
		yolov3.runningYolo(frame, yoloRet, regularDetect);
		//yolov3.drowBoxes(frame, yoloRet);
		//showDetectRet(frame, yoloRet);
		showMatching(frame, regularDetect);

		double end_time = (double)cv::getTickCount();
		double fps = cv::getTickFrequency() / (end_time - start_time);
		double spend_time = (end_time - start_time) / cv::getTickFrequency();
		std::string FPS = "FPS:" + cv::format("%.2f", fps) + "  spend time:" + cv::format("%.2f", spend_time * 1000) + "ms";
		putText(frame, FPS, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("YoloV3 detect results", frame);

		//yolov3.saveVider(frame, yoloRet);
		//yolov3.saveVider(frame);
	}

	cv::destroyAllWindows();
}