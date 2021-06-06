#include <iostream>
#include "opencv2/opencv.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "FreeImage.h"

using namespace cv;

int main(int argc, char** argv) {
	const char* filename =
		"..\\..\\PortraitNet.mnn";
	MNN::Interpreter* interpreter = MNN::Interpreter::createFromFile(filename);
	
	MNN::ScheduleConfig config;
	MNN::Session* session = interpreter->createSession(config);
	
	auto input  = interpreter->getSessionInput(session, NULL);
	auto output = interpreter->getSessionOutput(session, NULL);

	auto dims  = input->shape();
	int size_h = dims[2];
	int size_w = dims[3];

	auto inputPatch = argv[1];
	FREE_IMAGE_FORMAT f = FreeImage_GetFileType(inputPatch);
	FIBITMAP* bitmap = FreeImage_Load(f, inputPatch);
	auto newBitmap = FreeImage_ConvertTo32Bits(bitmap);
	auto width = FreeImage_GetWidth(newBitmap);
	auto height = FreeImage_GetHeight(newBitmap);
	FreeImage_Unload(bitmap);
	
	MNN::CV::Matrix trans;
	// Dst -> [0, 1]
	trans.postScale(1.0/size_w, 1.0/size_h);
	// Flip Y
	trans.postScale(1.0,-1.0, 0.0, 0.5);
	// [0, 1] -> Src
	trans.postScale(width, height);
	
	MNN::CV::ImageProcess::Config img_config;
	img_config.filterType = MNN::CV::NEAREST;
	float mean[3]     = {0.0f, 0.0f, 0.0f};
	float normals[3]  = {1.0f, 1.0f, 1.0f};
	::memcpy(img_config.mean, mean, sizeof(mean));
	::memcpy(img_config.normal, normals, sizeof(normals));
	img_config.sourceFormat = MNN::CV::RGBA;
	img_config.destFormat = MNN::CV::BGR;
	std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(img_config));
	pretreat->setMatrix(trans);
	
	pretreat->convert((uint8_t*)FreeImage_GetScanLine(newBitmap, 0), width, height, 0, input);
	
	MNN::ErrorCode error = interpreter->runSession(session);
	
	std::cout << output->batch() << std::endl;
	std::cout << output->channel() << std::endl;
	std::cout << output->height() << std::endl;
	std::cout << output->width() << std::endl;
	
	float* seg_out = output->host<float>();
	cv::Mat out(size_h, size_w, CV_32FC1, seg_out);
	
	cv::Mat out_img(out.size(), CV_8UC1);
	out.convertTo(out_img, CV_8UC1, 255.0f);
	cv::Mat resized_img;
	cv::resize(out_img, resized_img, cv::Size(width, height));
	cv::imwrite("out.png", resized_img);
	
	delete interpreter;
	return 0;
}