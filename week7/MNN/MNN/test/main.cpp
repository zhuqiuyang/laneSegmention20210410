#include <iostream>
#include "opencv2/opencv.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"

#define my_min(a, b) (((a) < (b)) ? (a) : (b))

using namespace cv;

// copy 1 channel mat to 3 channels (used on mask)
Mat convertTo3Channels(const Mat& img, int type)
{
    cv::Mat three_channel = cv::Mat::zeros(img.rows, img.cols, type);
    std::vector<cv::Mat> channels;
    for (int i = 0; i < 3; i++)
    {
        channels.push_back(img);
    }
    cv::merge(channels, three_channel);
    return three_channel;
}

int main(int argc, char** argv) {
	
	// input background image path
	if (argc < 2) {
		std::cout << "Please specify the background image" << std::endl;
		return 0;
	}
	
	// specify model path and create interpreter
	const char* filename = "..\\..\\PortraitNet.mnn";
	MNN::Interpreter* interpreter = MNN::Interpreter::createFromFile(filename);
	
	// create inference config
	MNN::ScheduleConfig config;
	
	// you can change backend type (CPU/AUTO/OPENCL/... , but now only CPU success)
	// config.type = MNN_FORWARD_CPU;
	// config.type = MNN_FORWARD_AUTO;
	config.type = MNN_FORWARD_OPENCL;
	
	// create inference session
	MNN::Session* session = interpreter->createSession(config);
	
	// get input and output tensor
	auto input  = interpreter->getSessionInput(session, NULL);
	auto output = interpreter->getSessionOutput(session, NULL);

	/************  Tail of model (softmax & slice)  ***************/
	// specify model path and create interpreter
	const char* filename_tail = "..\\..\\PortraitNet_tail.mnn";
	MNN::Interpreter* interpreter_tail = MNN::Interpreter::createFromFile(filename_tail);
	
	// create inference config
	MNN::ScheduleConfig config_tail;
	
	// for now, softmax & slice can only run on CPU
	config_tail.type = MNN_FORWARD_CPU;
	
	// create inference session
	MNN::Session* session_tail = interpreter_tail->createSession(config_tail);
	
	// get input and output tensor
	auto input_tail  = interpreter_tail->getSessionInput(session_tail, NULL);
	auto output_tail = interpreter_tail->getSessionOutput(session_tail, NULL);

	/*************************************************************/

	// get input shape (224 x 224)
	auto dims  = input->shape();
	int size_h = dims[2];
	int size_w = dims[3];
	
	// create a video capture to read camera image
	cv::VideoCapture cap(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
	
	// read background image and convert to float (0.0 - 1.0) for float blending
	const char* background_path = argv[1];
	cv::Mat background = cv::imread(background_path);
	cv::resize(background, background, cv::Size(640, 360));
	background.convertTo(background, CV_32FC3, 1.0/255);
	
	// firstly read a frame and ensure the camera is ok
	cv::Mat img_bgr;
	bool ret = cap.read(img_bgr);
	if (!ret) return -1;
	
	// create some intermediate variables
	cv::Mat resized_img_bgr, padding_img_bgr;
	cv::Mat img_bgra(img_bgr.size(), CV_8UC4);
	int width = img_bgr.cols;
	int height = img_bgr.rows;
	
	// do some image transformations (e.g. resize) using MNN functions
	// you can use MNN or use opencv (this code use opencv)
	//
	// MNN::CV::Matrix trans;
	// // Dst -> [0, 1]
	// trans.postScale(1.0/size_w, 1.0/size_h);
	// // [0, 1] -> Src
	// trans.postScale(width, height);
	
	// define MNN image preprocessor
	MNN::CV::ImageProcess::Config img_config;
	img_config.filterType = MNN::CV::NEAREST;
	// set mean and std
	float mean[3]     = {103.94f, 116.78f, 123.68f};
	float normals[3]  = {0.017f, 0.017f, 0.017f};
	::memcpy(img_config.mean, mean, sizeof(mean));
	::memcpy(img_config.normal, normals, sizeof(normals));
	// set image type
	img_config.sourceFormat = MNN::CV::BGRA;
	img_config.destFormat = MNN::CV::BGR;
	// create preprocessor
	std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(img_config));
	// set transformation
	// pretreat->setMatrix(trans);
	
	// count time and fps
	int frame_cnt = 0;
	double start = static_cast<double>(cv::getTickCount());
	
	while (ret) {
		// read image
		ret = cap.read(img_bgr);
		
		// resize and pad image to square (224 x 224)
		// resize
		width = img_bgr.cols;
		height = img_bgr.rows;
		float ratio = 1.0f * width / height;
		int dst_w = int(my_min(size_h * ratio, size_w));
		int dst_h = int(my_min(size_w / ratio, size_h));
		int origin_x = (size_w - dst_w) / 2;
		int origin_y = (size_h - dst_h) / 2;
		cv::resize(img_bgr, resized_img_bgr, cv::Size(dst_w, dst_h));
		// pad
		int top = origin_y;
		int bottom = size_h - dst_h - top;
		int left = origin_x;
		int right = size_w - dst_w - left;
		cv::copyMakeBorder(resized_img_bgr, padding_img_bgr, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
		
		// image preprocessing
		cv::cvtColor(padding_img_bgr, img_bgra, COLOR_BGR2BGRA, 4);
		pretreat->convert((uint8_t*)img_bgra.data, size_w, size_h, 0, input);
		
		// run inference and count time
		// double run_start = static_cast<double>(cv::getTickCount());
		MNN::ErrorCode error = interpreter->runSession(session);
		// double run_end = static_cast<double>(cv::getTickCount());
		// std::cout << (run_end - run_start) / (double)cv::getTickFrequency() << std::endl;
		
		// output = interpreter->getSessionOutput(session, NULL);
		MNN::Tensor host_output(output, output->getDimensionType());
		output->copyToHostTensor(&host_output);
		
		input_tail->copyFromHostTensor(&host_output);
		interpreter_tail->runSession(session_tail);
		
		// get mask output
		float* mask_data = output_tail->host<float>();
		cv::Mat ori_mask(size_h, size_w, CV_32FC1, mask_data);
		
		// crop and resize back to original size, and convert to 3 channels
		cv::Rect rect(origin_x, origin_y, dst_w, dst_h);
		cv::Mat crop_mask(ori_mask(rect));
		cv::Mat resized_mask;
		cv::resize(crop_mask, resized_mask, cv::Size(width, height));
		cv::Mat mask = convertTo3Channels(resized_mask, CV_32FC3);
		
		// blend foreground and background using float type (0.0 - 1.0)
		img_bgr.convertTo(img_bgr, CV_32FC3, 1.0/255);
		cv::Mat blend_background;
		cv::multiply(mask, background, blend_background);
		cv::multiply(cv::Scalar::all(1.0f) - mask, img_bgr, img_bgr);
		cv::add(blend_background, img_bgr, img_bgr);
		
		// show results, the value of img_bgr is (0.0 - 1.0)
		// you can also convert to 0-255 and convert to uint8
		cv::imshow("show", img_bgr);
		int key = cv::waitKey(1);
		// press ESC to quit
		if (key == 27) break;
		
		// count time and fps
		frame_cnt++;
		double end = static_cast<double>(cv::getTickCount());
		double time = (end - start) / (double)cv::getTickFrequency();
		std::cout << (frame_cnt / time) << std::endl;
	}
	
	// release resources
	delete interpreter;
	cap.release();
	return 0;
}