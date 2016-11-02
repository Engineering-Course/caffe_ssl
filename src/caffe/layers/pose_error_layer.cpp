#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <numeric>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pose_error_layer.hpp"

namespace caffe {
template <typename Dtype>
void PoseErrorLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    num_joint_ = this->layer_param_.pose_error_param().num_joint();
    error_order_ = this->layer_param_.pose_error_param().error_order();
    CHECK_LE(error_order_, 3) << "The error_order should lower equals 3.";
}
template <typename Dtype>
void PoseErrorLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The bottom data should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The bottom data should have the same channel.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The bottom data should have the same height.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The bottom data should have the same width.";
  CHECK_EQ(bottom[0]->width(), num_joint_ * 2)
    << "The bottom data should have the same width as double num_joint_.";
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}
template <typename Dtype>
void PoseErrorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data_one = bottom[0]->cpu_data();
  const Dtype* bottom_data_two = bottom[1]->cpu_data();
  const Dtype* bottom_data_three = NULL;
  const Dtype* bottom_data_four = NULL;
  if (error_order_ == 2) {
    bottom_data_three = bottom[2]->cpu_data();
    bottom_data_four = bottom[3]->cpu_data();
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int x1, x2, y1, y2;
  int left_arm = 3, right_arm = 4;
  //int left_leg = 5, right_leg = 6, left_shoe = 7, right_shoe = 8;
  for (int i = 0; i < num; ++i) {
    double total_distance = 0;
    for (int j = 0; j < num_joint_; ++j) {
      x1 = bottom_data_one[j*2];
      x2 = bottom_data_two[j*2];
      y1 = bottom_data_one[j*2+1];
      y2 = bottom_data_two[j*2+1];
      total_distance += sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
    }
    //LOG(INFO) << "dis of 2: " << total_distance;
    if (error_order_ == 2) {
      x1 = bottom_data_three[left_arm*2];
      x2 = bottom_data_four[left_arm*2];
      y1 = bottom_data_three[left_arm*2+1];
      y2 = bottom_data_four[left_arm*2+1];
      total_distance += sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
      x1 = bottom_data_three[right_arm*2];
      x2 = bottom_data_four[right_arm*2];
      y1 = bottom_data_three[right_arm*2+1];
      y2 = bottom_data_four[right_arm*2+1];
      total_distance += sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
    }
    //LOG(INFO) << "dis plus 1: " << total_distance;
    if (error_order_ == 1) {
      total_distance /= 10;
    } else if (error_order_ == 2) {
      total_distance /= 8;
    } else if (error_order_ == 3) {
      total_distance /= 5;
    } else {
      LOG(FATAL) << "Unexpected error_order: " << error_order_;
    }
    // if (total_distance > 10) {
    //   total_distance = 10;
    // }
    top_data[0] = total_distance;
    //LOG(INFO) << "total_distance: " << total_distance;
    bottom_data_one += bottom[0]->offset(1);
    bottom_data_two += bottom[1]->offset(1);
    top_data += top[0]->offset(1);
    if (error_order_ == 2) {
      bottom_data_three += bottom[2]->offset(1);
      bottom_data_four += bottom[3]->offset(1);
    }
  }
}

INSTANTIATE_CLASS(PoseErrorLayer);
REGISTER_LAYER_CLASS(PoseError);

}  // namespace caffe
