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
}

template <typename Dtype>
void PoseCreateLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->width(), num_joint_ * 2)
		<< "The bottom width and num of joint should have the same number.";
  top[0]->Reshape(bottom[1]->num(), num_joint_, bottom[1]->height(), bottom[1]->width());
}

template <typename Dtype>
void PoseCreateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int num = bottom[1]->num();
  int height = bottom[1]->height();
  int width = bottom[1]->width();
  float sigma = 1.0;  //////////// 1.0;

  for (int i = 0; i < num; ++i) {
    for (int n = 0; n < num_joint_; ++n) {
      int center_x = int(bottom_data[n * 2]);
      int center_y = int(bottom_data[n * 2 + 1]);
      for (int yy = 0; yy < height; yy++) {
        for (int xx = 0; xx < width; xx++) {
          int index = (n * height + yy) * width + xx;
          if (center_x == 0 && center_y == 0) {
            top_data[index] = 0;
          } else {
            float gaussian = (1 / (sigma * sqrt(2 * M_PI))) *
                  exp(-0.5 * (pow(yy-center_y, 2.0) + pow(xx-center_x, 2.0)) *
                  pow(1/sigma, 2.0));
            gaussian = 4 * gaussian;     ///4
            top_data[index] = gaussian;
          }
        }
      }
    }
    bottom_data += bottom[0]->offset(1);
    top_data += top[0]->offset(1);
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoseCreateLayer);
#endif

INSTANTIATE_CLASS(PoseCreateLayer);
REGISTER_LAYER_CLASS(PoseCreate);

}  // namespace caffe
