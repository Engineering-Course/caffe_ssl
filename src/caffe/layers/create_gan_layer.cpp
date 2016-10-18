#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/create_gan_layer.hpp"


namespace caffe {

template <typename Dtype>
void CreateGanLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  	gan_label_ = this->layer_param_.create_gan_param().gan_label();
}

template <typename Dtype>
void CreateGanLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void CreateGanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();

  for (int i = 0; i < num; ++i) {
  	if (gan_label_ == 0) {
  		top_data[i] = 0;
  	} else if (gan_label_ == 1) {
  		top_data[i] = 1;
  	} else {
  		LOG(FATAL) << "Unexpected gan label: " << gan_label_;
  	}
  }
}


#ifdef CPU_ONLY
STUB_GPU(CreateGanLayer);
#endif

INSTANTIATE_CLASS(CreateGanLayer);
REGISTER_LAYER_CLASS(CreateGan);

}  // namespace caffe
