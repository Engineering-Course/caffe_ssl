#include <cfloat>
#include <vector>

#include "caffe/layers/merge_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MergeLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  order_ = this->layer_param_.merge_label_param().order();
}
template <typename Dtype>
void MergeLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  CHECK_EQ(channels, 1)
    << "The merge label layer output should be 1.";
  top[0]->Reshape(num, channels, height, width);
}
template <typename Dtype>
void MergeLabelLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int data_index;
  if (order_ == 2) {
    for (int i = 0; i < num; ++i) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int c = 0;
          data_index = (c * height + h) * width + w;
          Dtype in_cls = bottom_data[data_index];
          int out_cls;
          switch(int(in_cls)) {
            case 0:  out_cls = 0; break;
            case 1:  out_cls = 4; break;
            case 2:  out_cls = 4; break;
            case 3:  out_cls = 1; break;
            case 4:  out_cls = 4; break;
            case 5:  out_cls = 3; break;
            case 6:  out_cls = 5; break;
            case 7:  out_cls = 3; break;
            case 8:  out_cls = 2; break;
            case 9:  out_cls = 2; break;
            case 10: out_cls = 5; break;
            case 11: out_cls = 3; break;
            case 12: out_cls = 2; break;
            case 13: out_cls = 4; break;
            case 14: out_cls = 1; break;
            case 15: out_cls = 1; break;
            case 16: out_cls = 2; break;
            case 17: out_cls = 2; break;
            case 18: out_cls = 2; break;
            case 19: out_cls = 2; break;
            case 255: out_cls = 255; break;
            default: LOG(FATAL) << "Unexpected cls :" << in_cls; break; 
          }
          top_data[data_index] = static_cast<Dtype>(out_cls);
        }
      } 
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
    }
  } else if (order_ == 3) {
    for (int i = 0; i < num; ++i) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int c = 0;
          data_index = (c * height + h) * width + w;
          Dtype in_cls = bottom_data[data_index];
          int out_cls;
          switch(int(in_cls)) {
            case 0:  out_cls = 0; break;
            case 1:  out_cls = 1; break;
            case 2:  out_cls = 2; break;
            case 3:  out_cls = 1; break;
            case 4:  out_cls = 1; break;
            case 5:  out_cls = 3; break;
            case 255: out_cls = 255; break;
            default: LOG(FATAL) << "Unexpected cls :" << in_cls; break;
          }
          top_data[data_index] = static_cast<Dtype>(out_cls);
        }
      } 
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
    }
  } else {
    LOG(FATAL) << "Unexpected order :" << order_;
  }  
}
template <typename Dtype>
void MergeLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(MergeLabelLayer);
#endif

INSTANTIATE_CLASS(MergeLabelLayer);
REGISTER_LAYER_CLASS(MergeLabel);
}  // namespace caffe
