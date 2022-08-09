//
// Created by xmcchv on 2022/8/2.
//
#include <dirent.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <argparse.hpp>
#include "include/detector.h"

#ifndef LOSIAMMASK_UTILS_H
#define LOSIAMMASK_UTILS_H

bool dirExists(const std::string& path)
{
    struct stat info{};
    if (stat(path.c_str(), &info) != 0)
        return false;
    return info.st_mode & S_IFDIR;
}

std::vector<std::string> listDir(const std::string& path, const std::vector<std::string>& match_ending)
{
    static const auto ends_with = [](std::string const & value, std::string const & ending) -> bool
    {
        if (ending.size() > value.size()) return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    };

    if(!dirExists(path)) {
        throw std::runtime_error(std::string("Directory not found: ") + path);
    }

    std::vector<std::string> files;
    DIR *dir = opendir(path.c_str());

    if(dir == nullptr)
        return files;

    struct dirent *pdirent;
    while ((pdirent = readdir(dir)) != nullptr) {
        std::string name(pdirent->d_name);
        for(const auto& ending : match_ending){
            if(ends_with(name, ending)) {
                files.push_back(path + "/" + name);
                break;
            }
        }
    }
    closedir(dir);

    return files;
}

void overlayMask(const cv::Mat& src, const cv::Mat& mask, cv::Mat& dst) {
    std::vector<cv::Mat> chans;
    cv::split(src, chans);
    cv::max(chans[2], mask, chans[2]);
    cv::merge(chans, dst);
}

void drawBox(
        cv::Mat& img, const cv::RotatedRect& box, const cv::Scalar& color,
        int thickness = 1, int lineType = cv::LINE_8, int shift = 0
) {
    cv::Point2f corners[4];
    box.points(corners);
    for(int i = 0; i < 4; ++i) {
        cv::line(img, corners[i], corners[(i + 1) % 4], color, thickness, lineType, shift);
    }
}

// load yolo classes name
std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline (infile,line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

  return class_names;
}


int mouseclickx,mouseclicky;

void onMouse(int event,int x,int y,int flags,void* param)
{
//    cv::Mat* im = reinterpret_cast<cv::Mat*>(param);
    switch(event)
    {
        case cv::EVENT_LBUTTONDBLCLK:{
            mouseclickx = x;
            mouseclicky = y;
            std::cout << "clicking at (" << x << "," << y << ")" << std::endl;
        }
            //左键按下显示像素值

    }
}

cv::Point showDetectResult(cv::Mat& img,
          const std::vector<std::vector<Detection>>& detections,
          const std::vector<std::string>& class_names,
          bool label = true) {

    if (!detections.empty()) {
        for (const auto& detection : detections[0]) {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

            if (label) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = class_names[class_idx] + " " + ss.str();

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline=0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(img,
                              cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                              cv::Point(box.tl().x + s_size.width, box.tl().y),
                              cv::Scalar(0, 0, 255), -1);
                cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                            font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
            }
        }
    }

    cv::namedWindow("Result and select ROI", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Result and select ROI",onMouse,reinterpret_cast<void*>(&img));//鼠标响应函数

    while(1){
        cv::imshow("Result and select ROI", img);
        if((cv::waitKey(1) & 0xFF) == 13)
            return cv::Point(mouseclickx,mouseclicky);
//            break;
    }


}

#endif //LOSIAMMASK_UTILS_H
