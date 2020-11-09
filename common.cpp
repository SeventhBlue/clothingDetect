#include "common.h"

void drawRet(cv::Mat& frame, std::vector<YoloDetSt>& yoloRet, int i) {
	cv::rectangle(frame, yoloRet[i].rect, cv::Scalar(0, 255, 0));
	std::string label = cv::format("%.2f", yoloRet[i].confidences);
	label = yoloRet[i].label + ": " + label;
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

	int top = cv::max(yoloRet[i].rect.y, labelSize.height);
	rectangle(frame, cv::Point(yoloRet[i].rect.x, top - labelSize.height),
		cv::Point(yoloRet[i].rect.x + labelSize.width, top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
	putText(frame, label, cv::Point(yoloRet[i].rect.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}

void showMatching(cv::Mat& frame, std::vector<RegularDetect>& regularDetect) {
	for (int i = 0; i < regularDetect.size(); i++) {
		if (regularDetect[i].flag != 7) {
			cv::rectangle(frame, regularDetect[i].personRect, cv::Scalar(0, 0, 255));
			cv::rectangle(frame, regularDetect[i].hatRect, cv::Scalar(0, 0, 255));
			cv::rectangle(frame, regularDetect[i].bootsRect, cv::Scalar(0, 0, 255));
			cv::rectangle(frame, regularDetect[i].workClothesRect, cv::Scalar(0, 0, 255));
		}
		else {
			cv::rectangle(frame, regularDetect[i].personRect, cv::Scalar(0, 255, 0));
			cv::rectangle(frame, regularDetect[i].hatRect, cv::Scalar(0, 255, 0));
			cv::rectangle(frame, regularDetect[i].bootsRect, cv::Scalar(0, 255, 0));
			cv::rectangle(frame, regularDetect[i].workClothesRect, cv::Scalar(0, 255, 0));
		}
	}
}

void showDetectRet(cv::Mat& frame, std::vector<YoloDetSt>& yoloRet) {
	bool isDrowPerson = true;
	bool isDrowHat = true;
	bool isDrowWorkClothes = true;
	bool isDrowBoots = true;

	int personNum = 0;
	int hatNum = 0;
	int workClothesNum = 0;
	int bootsNum = 0;

	cv::namedWindow("检测结果统计", cv::WINDOW_AUTOSIZE);
	cv::resizeWindow("检测结果统计", 800, 600);
	cv::Mat retMat = cv::Mat(600, 800, CV_8UC3, cv::Scalar::all(255));

	for (int i = 0; i < yoloRet.size(); i++) {
		if (yoloRet[i].label == "person") {
			++personNum;
			if (isDrowPerson) {
				drawRet(frame, yoloRet, i);
			}
		}

		if (yoloRet[i].label == "hat") {
			++hatNum;
			if (isDrowHat) {
				drawRet(frame, yoloRet, i);
			}
		}

		if (yoloRet[i].label == "workClothes") {
			++workClothesNum;
			if (isDrowWorkClothes) {
				drawRet(frame, yoloRet, i);
			}
		}

		if (yoloRet[i].label == "boots") {
			++bootsNum;
			if (isDrowBoots) {
				drawRet(frame, yoloRet, i);
			}
		}
	}
	putText(retMat, "person numbers:", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0));
	putText(retMat, cv::format("%d", personNum), cv::Point(110, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
	putText(retMat, "hat numbers:", cv::Point(500, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar());
	putText(retMat, cv::format("%d", hatNum), cv::Point(580, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
	putText(retMat, "workClothes numbers:", cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar());
	putText(retMat, cv::format("%d", workClothesNum), cv::Point(110, 230), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
	putText(retMat, "boots numbers:", cv::Point(500, 200), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar());
	putText(retMat, cv::format("%d", bootsNum), cv::Point(580, 230), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));

	cv::imshow("检测结果统计", retMat);
}