#pragma once

#include <opencv2/highgui.hpp>
#include "Yolo.h"

// deep learn
void showDetectRet(cv::Mat& frame, std::vector<YoloDetSt>& yoloRet);
void drawRet(cv::Mat& frame, std::vector<YoloDetSt>& yoloRet, int i);
void showMatching(cv::Mat& frame, std::vector<RegularDetect>& regularDetect);

// color