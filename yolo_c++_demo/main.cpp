#define OPENCV
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>         // std::mutex, std::unique_lock
#include <cmath>
#include <unistd.h>
#include "yolo_v2_class.hpp"    // imported functions from DLL
#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    for (auto &i : result_vec) 
    {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            std::string coords_3d;
            if (!std::isnan(i.z_3d)) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
                coords_3d = ss.str();
                cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
                int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
                if (max_width_3d > max_width) max_width = max_width_3d;
            }
            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
            if(!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y-1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}

void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
                  << ", w = " << i.w << ", h = " << i.h
                  << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

template<typename T>
class send_one_replaceable_object_t {
    const bool sync;
    std::atomic<T *> a_ptr;
public:

    void send(T const& _obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        if (sync) {
            while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while(!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    bool is_object_present() {
        return (a_ptr.load() != NULL);
    }

    send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
    {}
};

int main(int argc, char *argv[])
{
    std::string  names_file = "coco.names";
    std::string  cfg_file = "cfg/yolov4-tiny.cfg";
    std::string  weights_file = "yolov4-tiny.weights";
    std::string filename;

    
    float const thresh =  0.45;
    Detector detector(cfg_file, weights_file);
    auto obj_names = objects_names_from_file(names_file);

    preview_boxes_t large_preview(100, 150, false), small_preview(50, 50, true);
    bool show_small_boxes = false;
    cv::VideoCapture cap;
	cap.open(2);
    cv::Mat mat_img;
    namedWindow("detect",cv::WINDOW_AUTOSIZE);
    while (char(cv::waitKey(1))!='q')
    { 
        cap>>mat_img;
        auto det_image = detector.mat_to_image_resize(mat_img);
        // auto start = std::chrono::steady_clock::now();
        std::vector<bbox_t> result_vec = detector.detect_resized(*det_image, mat_img.size().width, mat_img.size().height);
        draw_boxes(mat_img, result_vec, obj_names);
        // auto end = std::chrono::steady_clock::now();
        // std::chrono::duration<double> spent = end - start;
        // show_console_result(result_vec, obj_names);
        // std::cout << " Time: " << spent.count() << " sec \n";
        cv::imshow("detect",mat_img);
    }
    cap.release();
	cv::destroyAllWindows();
    return 0;
}
