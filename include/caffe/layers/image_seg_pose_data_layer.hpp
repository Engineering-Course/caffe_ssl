#ifndef CAFFE_IMAGE_SEG_POSE_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_POSE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class ImageSegPoseDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSegPoseDataLayer(const LayerParameter& param)
    : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageSegPoseDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageSegPoseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 4; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  Blob<Dtype> transformed_label_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::pair<std::string, std::string>, vector<float> > > lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_POSE_DATA_LAYER_HPP_
