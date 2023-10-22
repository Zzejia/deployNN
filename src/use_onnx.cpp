#include "../include/use_onnx.hpp"

cv::Mat USE_ONNX::getBlob(cv::Mat & raw) 
{
    double ipwidth = 640;
    double ipheight = 640;
    // cv::Mat ResizedImg;
    // cv::resize(raw, raw, cv::Size(ipwidth, ipheight), 0, 0, cv::INTER_CUBIC);

    cv::Scalar mean;
    cv::Scalar stdDev;
    cv::meanStdDev(raw, mean, stdDev);

    cv::Mat blob;
    cv::dnn::blobFromImage(raw, blob, 1./255, cv::Size(ipwidth, ipheight), cv::Scalar(), true, false);
    if (stdDev.val[0] != 0.0 && stdDev.val[1] != 0.0 && stdDev.val[2] != 0.0) {
        cv::divide(blob, stdDev, blob);
    }

    return blob;
}
