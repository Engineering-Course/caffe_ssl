#include <cfloat>
#include <vector>

#include "caffe/layers/reward_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RewardLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void RewardLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int num = bottom[0]->num();
  top[0]->Reshape(num, 1, 1, 1);
}

template <typename Dtype>
void RewardLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    top_data[i] = bottom_data[i * 2] * 0.000001;
  }
}

template <typename Dtype>
void RewardLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  const Dtype* top_data = top[0]->cpu_data();
  if (propagate_down[1]) {
    Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
    const int num = bottom[1]->num();
    const int height = bottom[1]->height();
    const int width = bottom[1]->width();
    for (int i = 0; i < num; ++i) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int index = i * height * width + h * width + w;
          bottom_diff[index] = top_data[i];
        }
      }
    }
  }
  
}

#ifdef CPU_ONLY
STUB_GPU(RewardLossLayer);
#endif

INSTANTIATE_CLASS(RewardLossLayer);
REGISTER_LAYER_CLASS(RewardLoss);

}  // namespace caffe
