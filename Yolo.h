#pragma once
#include <iostream>
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


typedef struct YoloDetSt {
	std::string label;
	float confidences;
	cv::Rect rect;
}YoloDetSt;

typedef struct RegularDetect {
	cv::Rect personRect;
	cv::Rect hatRect;
	cv::Rect bootsRect;
	cv::Rect workClothesRect;
	int flag = 0;   // 三位数，从左到右依次代表：帽子，靴子，衣服；1表示穿戴，0表示未穿戴
}RegularDetect;

class Yolo {
public:
	Yolo(std::string& modelPath, std::string& configPath, std::string& classesFile, bool isGpu);
	~Yolo();
	int loadModel();
	int runningYolo(cv::Mat& img, std::vector<YoloDetSt>& yoloRet, std::vector<RegularDetect>& regularDetect);
	void drowBoxes(cv::Mat& img, std::vector<YoloDetSt>& yoloRet);
	void saveVider(cv::Mat img, std::vector<YoloDetSt>& yoloRet);    // 将结果yoloRet绘制到img上并录制视频
	void saveVider(cv::Mat img);                                     // 将图片img录制成视频
private:
	std::string m_modelPath;
	std::string m_configPath;
	std::string m_classesFile;
	std::vector<std::string> m_outNames;
	bool m_isGpu = false;
	std::vector<std::string> m_classes;
	cv::dnn::Net m_net;

	// Yolo参数设置
	float m_confThreshold = 0.5;
	float m_nmsThreshold = 0.4;
	float m_scale = 0.00392;
	cv::Scalar m_mean = { 0,0,0 };
	bool m_swapRB = true;
	int m_inpWidth = 416;
	int m_inpHeight = 416;

	float m_matchingThreshold = 0.6;

	// 检测的图片保存成视频的参数
	int m_saveH = 0;
	int m_saveW = 0;
	cv::VideoWriter m_viderWriter;
	std::string m_viderName;
	int m_frames = 0;

	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, std::vector<YoloDetSt>& yoloRet, std::vector<RegularDetect>& regularDetect);
	std::string getLocNameTime();                    // 返回格式化时间：20200426_150925
	void setViderWriterPara(const cv::Mat& img);
	void matching(cv::Rect& personRect, std::vector<cv::Rect>& boxes_hat, std::vector<cv::Rect>& boxes_boots, std::vector<cv::Rect>& boxes_workClothes,
		std::vector<RegularDetect>& regularDetect, std::vector<int>& indices_hat, std::vector<int>& indices_boots, std::vector<int>& indices_workClothes);
	double maxIOU(cv::Rect& personRect, std::vector<cv::Rect>& boxes, std::vector<int>& indices, int& ind, std::string flag);
};