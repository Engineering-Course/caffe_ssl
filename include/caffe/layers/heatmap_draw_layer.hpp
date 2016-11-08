#ifndef CAFFE_HEATMAP_DRAW_LAYER_HPP_
#define CAFFE_HEATMAP_DRAW_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
  HeatmapDrawLayer
*/
template <typename Dtype>
class HeatmapDrawLayer : public Layer<Dtype> {
 public:
  explicit HeatmapDrawLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "HeatmapDraw"; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int iter_;
  string prefix_;
  vector<string> fnames_;
};

}  // namespace caffe

#endif  // CAFFE_HEATMAP_DRAW_LAYER_HPP_
