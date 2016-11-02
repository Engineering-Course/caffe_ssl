#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <numeric>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/heatmap_error_layer.hpp"

namespace caffe {
template <typename Dtype>
void HeatmapErrorLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    scale_ = this->layer_param_.heatmap_error_param().scale();
    CHECK_GT(scale_, 0) << "The scale should larger than 0.";
}
template <typename Dtype>
void HeatmapErrorLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
  diff_.ReshapeLike(*bottom[0]);
}
template <typename Dtype>
void HeatmapErrorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num(); 
  int count = bottom[0]->count();
  int size = count / num;
  for (int i = 0; i < num; ++i) {
    caffe_sub(
      size,
      bottom_data,
      bottom_label,
      diff_.mutable_cpu_data());
    Dtype dot = caffe_cpu_dot(size, diff_.cpu_data(), diff_.cpu_data());
    Dtype loss = dot / Dtype(2);
    top_data[0] = loss * scale_;

    bottom_data += bottom[0]->offset(1);
    bottom_label += bottom[1]->offset(1);
    top_data += top[0]->offset(1);
  }
}

INSTANTIATE_CLASS(HeatmapErrorLayer);
REGISTER_LAYER_CLASS(HeatmapError);

}  // namespace caffe
