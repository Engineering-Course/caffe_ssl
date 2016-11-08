#include <sstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "caffe/layers/heatmap_draw_layer.hpp"
#include "caffe/util/matio_io.hpp"

namespace caffe {

template <typename Dtype>
void HeatmapDrawLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				      const vector<Blob<Dtype>*>& top) {
  iter_ = 0;
  prefix_ = this->layer_param_.heatmap_draw_param().prefix();

  if (this->layer_param_.heatmap_draw_param().has_source()) {
    std::ifstream infile(this->layer_param_.heatmap_draw_param().source().c_str());
    CHECK(infile.good()) << "Failed to open source file "
			 << this->layer_param_.heatmap_draw_param().source();
    string linestr;
    while (std::getline(infile, linestr)) {
      std::istringstream iss(linestr);
      string filename;
      iss >> filename;
      fnames_.push_back(filename.substr(0, filename.size()));
    }
    LOG(INFO) << "HeatmapDraw will save a maximum of " << fnames_.size() << " files.";
  }
}

template <typename Dtype>
void HeatmapDrawLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void HeatmapDrawLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // const cv::Scalar color1(216, 16, 216);
  // const cv::Scalar color2(16, 196, 21);
  // const cv::Scalar gt_color(255, 16, 21);
  // const cv::Scalar pred_color(21, 16, 255);
  // std::vector<cv::Scalar> colors;
  // colors.push_back(color1);
  // colors.push_back(color2);

  const int num_ = bottom[0]->num();
  const int channels_ = bottom[0]->channels();
  const int height_ = bottom[0]->height();
  const int width_ = bottom[0]->width();
  const int heat_num_ = height_ * width_;
  const int b_size = bottom.size();
  // const int thickness = 0.2;

  for (int item_id = 0; item_id < num_; ++item_id) {
    const std::string image_path = prefix_ + fnames_[iter_] + ".png";
    LOG(INFO) << image_path;
    // std::ostringstream oss;
    // oss << prefix_;
    // if (this->layer_param_.heatmap_draw_param().has_source()) {
    //   CHECK_LT(iter_, fnames_.size()) << "Test has run for more iterations than it was supposed to";
    //   oss << fnames_[iter_] << ".png";
    // }
    cv::Mat heat_map = cv::Mat::zeros(
        b_size * height_, channels_ * width_, CV_8UC3);

    for(int bs = 0; bs < b_size; bs++) {
      for(int c = 0; c < channels_; c++) {
        // convert each heat map into color image
        cv::Mat img = cv::Mat::zeros(height_, width_, CV_8UC3);
        for(int h = 0; h < height_; h++) {
          for(int w = 0; w < width_; w++) { 
            const Dtype v = bottom[bs]->data_at(item_id, c, h, w);
            uchar v1 = 0;
            uchar v2 = 0;
            uchar v3 = v * 255;
            // v3 = v * 255;
            img.at<cv::Vec3b>(h, w) = cv::Vec3b(v1, v2, v3);

            
            // if (v !=0){
              // for(int ww=-6; ww < 6; ww++){
                // for(int hh =-6; hh< 6; hh++){

                            // set pixel
                  // if(h+hh < height_ && h+hh > 0 && w+ww < width_ && w+ww >0){
                    // v1 = 0;
                    // v2 = 0;
                    // v3 = v * 255;
                    // img.at<cv::Vec3b>(h, w) = cv::Vec3b(v1, v2, v3);
                  // }
                  
                // }
              // }
            // }
           
            //v2 = 255;
            

          }
        }  
        // copy img to heat_map
        // top_left.x, top_left.y, width, height
        cv::Rect rect(c * width_, bs * height_, width_, height_);
        cv::Mat rect_img = heat_map(rect) ;
        img.copyTo(rect_img);
      }
    }
    // draw line to distinguish each heat map for each joint/part 
    // and between predicted and ground truth
    // for(int bs = 0; bs < b_size; bs++) {
    //   for(int idx = 1 ; idx < channels_; idx++) {
    //     const cv::Point p1(idx * width_, bs * height_ - 1);
    //     const cv::Point p2(idx * width_, (bs + 1) * height_ - 1);
    //     // img, p1, p2, color, thickness, lineType, shift
    //     cv::line(heat_map, p1, p2, colors[bs], thickness);
    //   }
    // }
    // save
    cv::imwrite(image_path, heat_map);
  }
  ++iter_;
}

template <typename Dtype>
void HeatmapDrawLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  return;
}


INSTANTIATE_CLASS(HeatmapDrawLayer);
REGISTER_LAYER_CLASS(HeatmapDraw);

}  // namespace caffe
