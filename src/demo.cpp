#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <argparse.hpp>
#include <SiamMask/siammask.h>
#include <include/detector.h>
#include <SiamMask/utils.h>

// global define
float conf_thres = 0.4;         // yolo confidence threshold
float iou_thres = 0.5;          // yolo iou threshold
float input_image_size = 640;   // yolo input image size


int main(int argc, const char* argv[]) try {
    argparse::ArgumentParser parser;
    parser.appName("Video Object Tracking with yolov5 and SiamMask");

//    --source ../images/bus.jpg --weights ../weights/yolov5s.torchscript.pt --gpu --view-img
//    ./LOSiamMask -c config_vot.json -m ../models/SiamMask_VOT -w ../weights/yolov5s.torchscript.pt ../images/tennis

//  yolov5 .torchscript.pt models
    parser.addArgument("-w","--weights",1,false);
//    yolo classes name
    parser.addArgument("-n","--names",1,true);
//  siammask model
    parser.addArgument("-m", "--modeldir", 1, false);
//  siammask config  config_vot.json
    parser.addArgument("-c", "--config", 1, false);
//  first image
    // parser.addArgument("-s","--source",1,true);
//  iou conf
    parser.addArgument("--iou",1,true);
    parser.addArgument("--conf",1,true);
//    image dir
    parser.addFinalArgument("target");
//    parse the args
    parser.parse(argc, argv);
//  use CUDA
    torch::Device device(torch::kCUDA);
//    std::cout << device.is_cuda() << std::endl;
//    torch::DeviceType device_type = torch::kCUDA;

// load class names from dataset for visualization
    std::string classesdir = "../weights/coco.names";
    if (parser.gotArgument("names")){
        classesdir = parser.retrieve<std::string>("names");
    }
    std::vector<std::string> class_names = LoadNames(classesdir);
    std::cout << "classes load in " << classesdir << std::endl;
    if (class_names.empty()) {
        std::cout << "classes load error!" << std::endl;
        return -1;
    }

//  loalov5 weights .torchscript.pt
//    if(parser.exists("weights"))
    std::string weights = parser.retrieve<std::string>("weights");
    if(!parser.gotArgument("weights")){
        std::cout << "weights load error!" << std::endl;
        return -1;
    }
    // set up conf and iou threshold if input otherwise use default
    if (parser.gotArgument("conf")){
        conf_thres = parser.retrieve<float>("conf");
    }
    if(parser.gotArgument("iou")){
        iou_thres = parser.retrieve<float>("iou");
    }

//    load siammask model and config
    std::string modeldir =  parser.retrieve<std::string>("modeldir");
    SiamMask siammask(modeldir, device);
    std::cout << "model load in " << modeldir << std::endl;
    State state;
    std::string configloc = parser.retrieve<std::string>("config");
    state.load_config(configloc);
    std::cout << "config load in " << configloc << std::endl;


//    load images from target
    const std::string target_dir = parser.retrieve<std::string>("target");
    std::vector<std::string> image_files = listDir(target_dir, {"jpg", "png", "bmp"});
    std::sort(image_files.begin(), image_files.end());
    std::cout << image_files.size() << " images load in " << target_dir << std::endl;
//    put all images to vector
    std::vector<cv::Mat> images;
    for(const auto& image_file : image_files) {
        images.push_back(cv::imread(image_file));
    }

//    start LOSiamMask
//    cv::namedWindow("LOSiamMask");
    int64 toc = 0;

    for(unsigned long i = 0; i < images.size(); ++i) {
        int64 tic = cv::getTickCount();

        cv::Mat& src = images[i];

        if (i == 0) {
            cv::Mat orignimg = src;
            // start detector
            auto detector = Detector(weights, device.type());
            // inference
            auto result = detector.Run(images[0], conf_thres, iou_thres);
            cv::Point cpoint = showDetectResult(src, result, class_names);
            double minddis = 1e+9;
            Detection selectROI;
            for (Detection detection : result[0]) {
                std::vector<cv::Point> pointlist;
                // left top point
                pointlist.push_back(cv::Point(detection.bbox.tl()));
                pointlist.push_back(cv::Point(detection.bbox.x+detection.bbox.width,detection.bbox.y));
                pointlist.push_back(cv::Point(detection.bbox.x,detection.bbox.y+detection.bbox.height));
                // right down point
                pointlist.push_back(cv::Point(detection.bbox.br()));
                if(cv::pointPolygonTest(pointlist,cpoint,true) < minddis){
                    selectROI = detection;
                }
            }
            cv::Rect roi = selectROI.bbox;
            std::cout << roi.tl() << std::endl;
            std::cout << "SiamMask Initializing..." << std::endl;
            siameseInit(state, siammask, src, roi, device);
            cv::rectangle(src, roi, cv::Scalar(0, 255, 0));
        } else {
            siameseTrack(state, siammask, src, device);
            overlayMask(src, state.mask, src);
            drawBox(src, state.rotated_rect, cv::Scalar(0, 255, 0));
        }

        cv::imshow("SiamMask", src);
        toc += cv::getTickCount() - tic;
        cv::waitKey(1);
    }

    double total_time = toc / cv::getTickFrequency();
    double fps = image_files.size() / total_time;
    printf("SiamMask Time: %.1fs Speed: %.1ffps (with visulization!)\n", total_time, fps);

    return EXIT_SUCCESS;
} catch (std::exception& e) {
    std::cout << "Exception thrown!\n" << e.what() << std::endl;
    return EXIT_FAILURE;
}

