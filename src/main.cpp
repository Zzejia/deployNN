//#include <opencv2/opencv.hpp>
#include "../include/use_onnx.hpp"

int main(int argc, char** argv)
{
    std::string videoPath = "../video/fan2.avi";
    std::string xpath = "../weights/fan.onnx";

    cv::VideoCapture video(videoPath);
    if (!video.isOpened()) {
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(xpath);
    USE_ONNX dealer;

    while (1) {
        cv::Mat raw = cv::imread(videoPath);
        video.read(raw);
        if (raw.empty()) {
            break;
        }
        cv::Point lt = cv::Point(320, 0);
        cv::Point rb = cv::Point(960, 640);
        cv::Rect roi = cv::Rect(lt, rb);
        raw = raw(roi);
        
        // 输入图像，得到预测结果
        cv::Mat blob;
        blob = dealer.getBlob(raw);
        net.setInput(blob);
        cv::Mat predict = net.forward();

        // 绘制框
        for (int i = 0; i < predict.size[1]; i++) {
            float* data = predict.ptr<float>(0, i);
            if (data[4] > 0.5) {
                std::cout << data[4] << std::endl;
                int width = static_cast<int>(data[2]);
                int height = static_cast<int>(data[3]);
                int left = static_cast<int>(data[0] - 0.5 * width);
                int top = static_cast<int>(data[1] - 0.5 * height);
                cv::Rect target = cv::Rect(left, top, width, height);

                cv::rectangle(raw, target, cv::Scalar(0, 255, 0), 2);
                std::string label = "fan";
                cv::putText(raw, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("yolo", raw);
        cv::waitKey(0);
    }
    video.release();
    cv::destroyAllWindows();

    return 0;
}