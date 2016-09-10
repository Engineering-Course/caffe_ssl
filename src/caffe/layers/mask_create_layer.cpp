#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/mask_create_layer.hpp"

namespace caffe {

template <typename Dtype>
void MaskCreateLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  	num_cls_ = this->layer_param_.mask_create_param().num_cls();
}

template <typename Dtype>
void MaskCreateLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), num_cls_)
		<< "The bottom channels and num of classes should have the same number.";
  top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void MaskCreateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int data_index;
  int top_k = 1;  // only support for top_k = 1

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
	    // Top-k accuracy
	    std::vector<std::pair<Dtype, int> > bottom_data_vector;

	    for (int c = 0; c < channels; ++c) {
	      data_index = (c * height + h) * width + w;
	      bottom_data_vector.push_back(std::make_pair(bottom_data[data_index], c));
	    }
	    std::partial_sort(
	      bottom_data_vector.begin(), bottom_data_vector.begin() + top_k,
	      bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

	    top_data[h * width + w] = bottom_data_vector[0].second;
      }
    }
    bottom_data += bottom[0]->offset(1);
    top_data += top[0]->offset(1);
  }
}

INSTANTIATE_CLASS(MaskCreateLayer);
REGISTER_LAYER_CLASS(MaskCreate);

}  // namespace caffe
