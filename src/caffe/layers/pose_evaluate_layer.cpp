#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <numeric>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/pose_evaluate_layer.hpp"

namespace caffe {

template <typename Dtype>
void PoseEvaluateLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    num_joint_ = this->layer_param_.pose_evaluate_param().num_joint();
}

template <typename Dtype>
void PoseEvaluateLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), 1)
    << "The bottom channels should be 1.";
  top[0]->Reshape(bottom[0]->num(), 1, 1, num_joint_ * 2);  
}
int clsToJointFirst(int cls_) {
  switch(cls_) {
    case 1:  return 0; break;
    case 2:  return 0; break;
    case 4:  return 0; break;
    case 13: return 0; break;
    case 5:  return 1; break;
    case 7:  return 1; break;
    case 11: return 1; break;
    case 9:  return 2; break;
    case 12: return 2; break;
    case 14: return 3; break;
    case 15: return 4; break;
    case 16: return 5; break;
    case 17: return 6; break;
    case 18: return 7; break;
    case 19: return 8; break;
    default: return -1; break;
  }
}
int clsToJointSecond(int cls_) {
  switch(cls_) {
    case 4:  return 0; break;
    case 3:  return 1; break;
    case 2:  return 2; break;
    default: return -1; break;
  }
}
int clsToJointThird(int cls_) {
  switch(cls_) {
    case 1:  return 0; break;
    case 2:  return 1; break;
    default: return -1; break;
  }
}
int selectJointFun(int num_joint_, int cls_) {
  if (num_joint_ == 9) {
    return clsToJointFirst(cls_);
  } else if (num_joint_ == 3) {
    return clsToJointSecond(cls_);
  } else if (num_joint_ == 2) {
    return clsToJointThird(cls_);
  } else {
    LOG(FATAL) << "Unexpected num_joint " << num_joint_;
  }
}
template <typename Dtype>
void PoseEvaluateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  // LOG(INFO) << "pose_len: " << pose_len;

  std::vector<int > x_sum_vector[num_joint_];
  std::vector<int > y_sum_vector[num_joint_];
  int cls_, joint_id;

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        cls_ =  bottom_data[h * width + w];
        joint_id = selectJointFun(num_joint_, cls_);
        if (joint_id >= 0 && joint_id < num_joint_) {
          x_sum_vector[joint_id].push_back(w);
          y_sum_vector[joint_id].push_back(h);
        }
      }
    }
    for (int w = 0; w < num_joint_ * 2; ++w) {
      top_data[w] = 0;
    }
    for (int n = 0; n < num_joint_; n++) {
      if (x_sum_vector[n].size() > 0 && y_sum_vector[n].size() > 0) {
        double ave_x = std::accumulate(x_sum_vector[n].begin(), x_sum_vector[n].end(), 0.0)
                                      / x_sum_vector[n].size();
        double ave_y = std::accumulate(y_sum_vector[n].begin(), y_sum_vector[n].end(), 0.0)
                                      / y_sum_vector[n].size();
        // LOG(INFO) << "ave_x: " << ave_x << "  ave_y:" << ave_y;
        top_data[n*2] = int(ave_x);
        top_data[n*2+1] = int(ave_y);
        // LOG(INFO) << "cls: " << n << "  x: " << int(ave_x) << "  y: " << int(ave_y);
      } 
    }
    bottom_data += bottom[0]->offset(1);
    top_data += top[0]->offset(1);
  }
}

//virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//
//  const Dtype* top_diff = top[0]->cpu_diff();
//  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
////  if (propagate_down[0]) {
//    
////  }
//}
//
INSTANTIATE_CLASS(PoseEvaluateLayer);
REGISTER_LAYER_CLASS(PoseEvaluate);

}  // namespace caffe
