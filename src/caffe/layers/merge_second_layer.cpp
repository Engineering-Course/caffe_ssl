#include <cfloat>
#include <vector>

#include "caffe/layers/merge_second_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MergeSecondLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_labels_ = this->layer_param_.merge_second_param().num_labels();
  CHECK_EQ(num_labels_, 4)
    << "The merge first layer output should be 4.";
  LOG(INFO) << "Merge output size: " << num_labels_;
}

template <typename Dtype>
void MergeSecondLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(num_labels_, 4)
    << "The merge first layer output should be 4.";
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  top[0]->Reshape(num, num_labels_, height, width);
}

template <typename Dtype>
void MergeSecondLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(num_labels_, 4)
    << "The merge first layer output should be 4.";

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
          int c = 0;
          data_index = (c * height + h) * width + w;
          top_data[data_index] = bottom_data[data_index];
      }
    }
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        // Top-k accuracy
        std::vector<std::pair<Dtype, int> > bottom_data_vector;

        for (int c = 1; c < channels; ++c) {
          data_index = (c * height + h) * width + w;
          bottom_data_vector.push_back(std::make_pair(bottom_data[data_index], c));
        }
        std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

        int max_cls = bottom_data_vector[0].second;
        int second_cls;
        if (max_cls == 1 || max_cls == 3 || max_cls == 4) {
          second_cls = 1;
        } else if (max_cls == 2) {
          second_cls = 2;
        } else if (max_cls == 5) {
          second_cls = 3;
        } else {
          LOG(FATAL) << "Unexpected cls " << max_cls;
        }
        for (int c = 1; c < num_labels_; ++c) {
          data_index = (c * height + h) * width + w;
          if (c == second_cls) {
            top_data[data_index] = bottom_data_vector[0].first;
          } else {
            top_data[data_index] = bottom_data_vector.back().first;
          }
        }
      }
    } 
    bottom_data += bottom[0]->offset(1);
    top_data += top[0]->offset(1);
  }
}

template <typename Dtype>
void MergeSecondLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    // const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    int data_index;

    for (int i = 0; i < num; ++i) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int c = 0;
            data_index = (c * height + h) * width + w;
            bottom_diff[data_index] = top_diff[data_index];
        }
      }
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          for (int c = 1; c < num_labels_; ++c) {
            data_index = (c * height + h) * width + w;
            if (c == 1) {
              bottom_diff[(1 * height + h) * width + w] = top_diff[data_index];
              bottom_diff[(3 * height + h) * width + w] = top_diff[data_index];
              bottom_diff[(4 * height + h) * width + w] = top_diff[data_index];
            } else if (c == 2) {
              bottom_diff[(2 * height + h) * width + w] = top_diff[data_index];
            } else if (c == 3) {
              bottom_diff[(5 * height + h) * width + w] = top_diff[data_index];
            } else {
              LOG(FATAL) << "Unexpected cls " << c;
            }
          }
        }
      }
      top_diff += top[0]->offset(1);
      bottom_diff += bottom[0]->offset(1);
    }    
  }
}

#ifdef CPU_ONLY
STUB_GPU(MergeSecondLayer);
#endif

INSTANTIATE_CLASS(MergeSecondLayer);
REGISTER_LAYER_CLASS(MergeSecond);

}  // namespace caffe
