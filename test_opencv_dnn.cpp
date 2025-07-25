#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "DNN module available: " << (cv::dnn::getAvailableBackends().size() > 0 ? "Yes" : "No") << std::endl;
    
    auto backends = cv::dnn::getAvailableBackends();
    std::cout << "Available DNN backends: ";
    for (auto& backend : backends) {
        std::cout << backend.first << " ";
    }
    std::cout << std::endl;
    
    auto targets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_OPENCV);
    std::cout << "Available DNN targets: ";
    for (auto& target : targets) {
        std::cout << target << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
