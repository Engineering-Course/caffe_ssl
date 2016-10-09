#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pose_create_layer.hpp"

namespace caffe {

template <typename Dtype>
void PoseCreateLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  	num_joint_ = this->layer_param_.pose_create_param().num_joint();
    label_value_ = this->layer_param_.pose_create_param().label_value();
}

template <typename Dtype>
void PoseCreateLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), num_joint_)
		<< "The bottom channels and num of classes should have the same number.";
  top[0]->Reshape(bottom[0]->num(), 1, 1, num_joint_ * 2);
  top[1]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void PoseCreateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_label = top[1]->mutable_cpu_data();

  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int data_index;

  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < num_joint_; ++c) {
      double max_index = bottom_data[(c * height) * width];
      int max_x = 0, max_y = 0;
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          data_index = (c * height + h) * width + w;
          if (bottom_data[data_index] > max_index) {
            max_index = bottom_data[data_index];
            max_x = w;
            max_y = h;
          }
        }
      }
      // LOG(INFO) << "point1: " << c << "  w: " << max_x << "  y: " << max_y;
      top_data[c*2] = max_x;
      top_data[c*2+1] = max_y;
    }
    if (label_value_ == 1) {
      top_label[0] = 1;
    } else if (label_value_ == 0){
      top_label[0] = 0;
    } else {
      LOG(FATAL) << "Unexpected label_value: " << label_value_;
    }
    bottom_data += bottom[0]->offset(1);
    top_data += top[0]->offset(1);
    top_label += top[1]->offset(1);
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoseCreateLayer);
#endif

INSTANTIATE_CLASS(PoseCreateLayer);
REGISTER_LAYER_CLASS(PoseCreate);

}  // namespace caffe
