#include <fstream>
#include "Yolo.h"

Yolo::Yolo(std::string& modelPath, std::string& configPath, std::string& classesFile, bool isGpu) {
	m_modelPath = modelPath;
	m_configPath = configPath;
	m_classesFile = classesFile;
	m_isGpu = isGpu;
}

Yolo::~Yolo() {
	m_viderWriter.release();
}

int Yolo::loadModel() {
	int backendId;
	int targetId;
	// cpu or gpu
	if (m_isGpu) {
		backendId = cv::dnn::DNN_BACKEND_CUDA;
		targetId = cv::dnn::DNN_TARGET_CUDA;
	}
	else {
		backendId = cv::dnn::DNN_BACKEND_OPENCV;
		targetId = cv::dnn::DNN_TARGET_CPU;
	}

	// Open file with classes names.
	if (!m_classesFile.empty()) {
		std::ifstream ifs(m_classesFile.c_str());
		if (!ifs.is_open()) {
			std::string error = "File " + m_classesFile + " not found";
			std::cout << error << std::endl;
			return -1;
		}
		std::string line;
		while (std::getline(ifs, line)) {
			m_classes.push_back(line);
		}
	}

	// Load a model.
	m_net = cv::dnn::readNet(m_modelPath, m_configPath);
	m_net.setPreferableBackend(backendId);
	m_net.setPreferableTarget(targetId);

	m_outNames = m_net.getUnconnectedOutLayersNames();

	return 0;
}

int Yolo::runningYolo(cv::Mat& img, std::vector<YoloDetSt>& yoloRet, std::vector<RegularDetect>& regularDetect) {
	// Create a 4D blob from a frame.
	cv::Mat blob;
	cv::Mat frame;
	cv::Size inpSize(m_inpWidth > 0 ? m_inpWidth : img.cols,
		m_inpHeight > 0 ? m_inpHeight : img.rows);
	cv::dnn::blobFromImage(img, blob, m_scale, inpSize, m_mean, m_swapRB, false);

	// Run a model.
	m_net.setInput(blob);
	if (m_net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		cv::resize(img, img, inpSize);
		cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
		m_net.setInput(imInfo, "im_info");
	}
	std::vector<cv::Mat> outs;
	m_net.forward(outs, m_outNames);
	postprocess(img, outs, m_net, yoloRet, regularDetect);
	return 0;
}

void Yolo::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, std::vector<YoloDetSt>& yoloRet, std::vector<RegularDetect>& regularDetect) {
	static std::vector<int> outLayers = net.getUnconnectedOutLayers();
	static std::string outLayerType = net.getLayer(outLayers[0])->type;

	std::vector<int> classIds_person;
	std::vector<float> confidences_person;
	std::vector<cv::Rect> boxes_person;

	std::vector<int> classIds_hat;
	std::vector<float> confidences_hat;
	std::vector<cv::Rect> boxes_hat;

	std::vector<int> classIds_boots;
	std::vector<float> confidences_boots;
	std::vector<cv::Rect> boxes_boots;

	std::vector<int> classIds_workClothes;
	std::vector<float> confidences_workClothes;
	std::vector<cv::Rect> boxes_workClothes;

	if (outLayerType == "Region") {
		for (size_t i = 0; i < outs.size(); ++i) {
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > m_confThreshold) {
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					std::string label;
					if (!m_classes.empty()) {
						CV_Assert(classIdPoint.x < (int)m_classes.size());
					}

					if (m_classes[classIdPoint.x] == "person") {
						classIds_person.push_back(classIdPoint.x);
						confidences_person.push_back((float)confidence);
						boxes_person.push_back(cv::Rect(left, top, width, height));
					}else if (m_classes[classIdPoint.x] == "hat") {
						classIds_hat.push_back(classIdPoint.x);
						confidences_hat.push_back((float)confidence);
						boxes_hat.push_back(cv::Rect(left, top, width, height));
					}else if (m_classes[classIdPoint.x] == "boots") {
						classIds_boots.push_back(classIdPoint.x);
						confidences_boots.push_back((float)confidence);
						boxes_boots.push_back(cv::Rect(left, top, width, height));
					}else if (m_classes[classIdPoint.x] == "workClothes") {
						classIds_workClothes.push_back(classIdPoint.x);
						confidences_workClothes.push_back((float)confidence);
						boxes_workClothes.push_back(cv::Rect(left, top, width, height));
					}
					else {
						std::cout << "There was an unexpected result!" << std::endl;
					}
				}
			}
		}
	}
	else {
		std::cout << "Unknown output layer type: " + outLayerType << std::endl;
	}

	std::vector<int> indices_hat;
	cv::dnn::NMSBoxes(boxes_hat, confidences_hat, m_confThreshold, m_nmsThreshold, indices_hat);
	std::vector<int> indices_boots;
	cv::dnn::NMSBoxes(boxes_boots, confidences_boots, m_confThreshold, m_nmsThreshold, indices_boots);
	std::vector<int> indices_workClothes;
	cv::dnn::NMSBoxes(boxes_workClothes, confidences_workClothes, m_confThreshold, m_nmsThreshold, indices_workClothes);

	std::vector<int> indices_person;
	cv::dnn::NMSBoxes(boxes_person, confidences_person, m_confThreshold, m_nmsThreshold, indices_person);
	for (size_t i = 0; i < indices_person.size(); ++i) {
		int idx = indices_person[i];
		yoloRet.push_back(YoloDetSt{ m_classes[classIds_person[idx]], confidences_person[idx], boxes_person[idx] });
		matching(boxes_person[idx], boxes_hat, boxes_boots, boxes_workClothes, regularDetect, indices_hat, indices_boots, indices_workClothes);
	}

	// 以下都是测试使用
	/*for (size_t i = 0; i < indices_hat.size(); ++i) {
		int idx = indices_hat[i];
		yoloRet.push_back(YoloDetSt{ m_classes[classIds_hat[idx]], confidences_hat[idx], boxes_hat[idx] });
	}
	
	for (size_t i = 0; i < indices_boots.size(); ++i) {
		int idx = indices_boots[i];
		yoloRet.push_back(YoloDetSt{ m_classes[classIds_boots[idx]], confidences_boots[idx], boxes_boots[idx] });
	}
	
	for (size_t i = 0; i < indices_workClothes.size(); ++i) {
		int idx = indices_workClothes[i];
		yoloRet.push_back(YoloDetSt{ m_classes[classIds_workClothes[idx]], confidences_workClothes[idx], boxes_workClothes[idx] });
	}*/
}

void Yolo::matching(cv::Rect& personRect, std::vector<cv::Rect>& boxes_hat, std::vector<cv::Rect>& boxes_boots, std::vector<cv::Rect>& boxes_workClothes,
	std::vector<RegularDetect>& regularDetect, std::vector<int>& indices_hat, std::vector<int>& indices_boots, std::vector<int>& indices_workClothes) {
	int ind_hat = -1;
	int ind_boots = -1;
	int ind_workClothes = -1;
	double maxIOU_hat = maxIOU(personRect, boxes_hat, indices_hat, ind_hat, "hat");
	double maxIOU_boots = maxIOU(personRect, boxes_boots, indices_boots, ind_boots, "boots");
	double maxIOU_workClothes = maxIOU(personRect, boxes_workClothes, indices_workClothes, ind_workClothes, "workClothes");
	RegularDetect tmp_regularDetect;
	tmp_regularDetect.personRect = personRect;
	cv::Rect zeroRect(0, 0, 0, 0);
	if (maxIOU_hat >= m_matchingThreshold) {
		tmp_regularDetect.hatRect = boxes_hat[ind_hat];
		//indices_hat.erase(indices_hat.begin() + ind_hat);
		tmp_regularDetect.flag += 4;
	}
	else {
		tmp_regularDetect.hatRect = zeroRect;
	}

	if (maxIOU_boots >= m_matchingThreshold) {
		tmp_regularDetect.bootsRect = boxes_boots[ind_boots];
		//indices_boots.erase(indices_boots.begin() + ind_boots);
		tmp_regularDetect.flag += 2;
	}
	else {
		tmp_regularDetect.bootsRect = zeroRect;
	}

	if (maxIOU_workClothes >= m_matchingThreshold) {
		tmp_regularDetect.workClothesRect = boxes_workClothes[ind_workClothes];
		//indices_workClothes.erase(indices_workClothes.begin() + ind_workClothes);
		tmp_regularDetect.flag += 1;
	}
	else {
		tmp_regularDetect.workClothesRect = zeroRect;
	}

	regularDetect.push_back(tmp_regularDetect);
}

double Yolo::maxIOU(cv::Rect& personRect, std::vector<cv::Rect>& boxes, std::vector<int>& indices, int& ind, std::string flag) {
	double maxIOU = -1;
	cv::Rect tmp_rect;
	if (flag == "hat") {
		int x = personRect.x - 20;
		if (x < 0) {
			x = 0;
		}
		int y = personRect.y - 0.25 * personRect.height;
		if (y < 0) {
			y = 0;
		}
		int w = personRect.width + 40;
		int h = 0.5 * personRect.height;
		tmp_rect = cv::Rect(x, y, w, h);
	}
	else if (flag == "boots") {
		int x = personRect.x - 20;
		if (x < 0) {
			x = 0;
		}
		int y = personRect.y + 0.75 * personRect.height;
		int w = personRect.width + 40;
		int h = 0.5 * personRect.height;
		tmp_rect = cv::Rect(x, y, w, h);
	}
	else {
		tmp_rect = personRect;
	}

	for (int i = 0; i < indices.size(); i++) {
		int tmp_ind = indices[i];
		cv::Rect un_rect = tmp_rect & boxes[tmp_ind];
		double IOU = un_rect.area()*1.0 / boxes[tmp_ind].area();
		if (IOU > maxIOU) {
			maxIOU = IOU;
			ind = tmp_ind;
		}
	}
	return maxIOU;
}

void Yolo::drowBoxes(cv::Mat& img, std::vector<YoloDetSt>& yoloRet) {
	for (int i = 0; i < yoloRet.size(); i++) {
		cv::rectangle(img, yoloRet[i].rect, cv::Scalar(0, 0, 255));
		std::string label = cv::format("%.2f", yoloRet[i].confidences);
		label = yoloRet[i].label + ": " + label;
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int top = cv::max(yoloRet[i].rect.y, labelSize.height);
		rectangle(img, cv::Point(yoloRet[i].rect.x, top - labelSize.height),
			cv::Point(yoloRet[i].rect.x + labelSize.width, top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
		putText(img, label, cv::Point(yoloRet[i].rect.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
	}
}

// 返回格式化时间：20200426_150925
std::string Yolo::getLocNameTime() {
	struct tm t;              //tm结构指针
	time_t now;               //声明time_t类型变量
	time(&now);               //获取系统日期和时间
	localtime_s(&t, &now);    //获取当地日期和时间

	std::string time_name = cv::format("%d", t.tm_year + 1900) + cv::format("%.2d", t.tm_mon + 1) + cv::format("%.2d", t.tm_mday) + "_" +
		cv::format("%.2d", t.tm_hour) + cv::format("%.2d", t.tm_min) + cv::format("%.2d", t.tm_sec);
	return time_name;
}

void Yolo::setViderWriterPara(const cv::Mat& img) {
	m_saveH = img.size().height;
	m_saveW = img.size().width;
	m_viderName = "./data/" + getLocNameTime() + ".avi";
	m_viderWriter = cv::VideoWriter(m_viderName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, cv::Size(m_saveW, m_saveH));
	m_frames = 0;
}

void Yolo::saveVider(cv::Mat img) {
	if ((m_saveH == 0) && (m_saveW == 0)) {
		setViderWriterPara(img);
		m_viderWriter << img;
	}
	else {
		if ((m_saveH != img.size().height) || (m_saveW != img.size().width)) {
			cv::resize(img, img, cv::Size(m_saveW, m_saveH));
			m_viderWriter << img;
		}
		else {
			m_viderWriter << img;
		}
	}

	++m_frames;
	if (m_frames == 25 * 60 * 10) {   // 每十分钟从新录制新视频
		m_saveH = 0;
		m_saveW = 0;
		m_viderWriter.release();
	}
}

void Yolo::saveVider(cv::Mat img, std::vector<YoloDetSt>& yoloRet) {
	drowBoxes(img, yoloRet);
	if ((m_saveH == 0) && (m_saveW == 0)) {
		setViderWriterPara(img);
		m_viderWriter << img;
	}
	else {
		if ((m_saveH != img.size().height) || (m_saveW != img.size().width)) {
			cv::resize(img, img, cv::Size(m_saveW, m_saveH));
			m_viderWriter << img;
		}
		else {
			m_viderWriter << img;
		}
	}

	++m_frames;
	if (m_frames == 25 * 60 * 10) {   // 每十分钟从新录制新视频
		m_saveH = 0;
		m_saveW = 0;
		m_viderWriter.release();
	}
}