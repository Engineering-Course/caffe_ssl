// Copyright 2015 Tomas Pfister

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <stdint.h>

#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/data_heatmap.hpp"
#include "caffe/util/benchmark.hpp"
#include <unistd.h>

namespace caffe
{
template <typename Dtype>
DataHeatmapLayer<Dtype>::~DataHeatmapLayer<Dtype>() {
    this->StopInternalThread();
}

template<typename Dtype>
void DataHeatmapLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();

    // Shortcuts
    const int batchsize = heatmap_data_param.batchsize();
    const int label_width = heatmap_data_param.label_width();
    const int label_height = heatmap_data_param.label_height();
    const int label_batchsize = batchsize;
    root_img_dir_ = heatmap_data_param.root_img_dir();

    // initialise rng seed
    const unsigned int rng_seed = caffe_rng_rand();
    srand(rng_seed);

    // load GT
    std::string gt_path = heatmap_data_param.source();
    LOG(INFO) << "Loading annotation from " << gt_path;

    std::ifstream infile(gt_path.c_str());
    string img_name, labels;

    // sequential sampling
    while (infile >> img_name >> labels) {
        // read comma-separated list of regression labels
        std::vector <float> label;
        std::istringstream ss(labels);
        int labelCounter = 1;
        while (ss) {
            std::string s;
            if (!std::getline(ss, s, ',')) break;
            label.push_back(atof(s.c_str()));
            labelCounter++;
        }
        img_label_list_.push_back(std::make_pair(img_name, label));
    }
    this->datum_channels_ = 1;
    // init data
    this->transformed_data_.Reshape(batchsize, 1, 1, 1);
    top[0]->Reshape(batchsize, 1, 1, 1);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
        this->prefetch_[i].data_.Reshape(batchsize, 1, 1, 1);
    this->datum_size_ = 1 * 1 * 1;

    // init label
    int label_num_channels;

    label_num_channels = img_label_list_[0].second.size();
    label_num_channels /= 2;
    top[1]->Reshape(label_batchsize, label_num_channels, label_height, label_width);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
        this->prefetch_[i].label_.Reshape(label_batchsize, label_num_channels, label_height, label_width);

    LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
    LOG(INFO) << "output label size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();
    LOG(INFO) << "number of label channels: " << label_num_channels;
    LOG(INFO) << "datum channels: " << this->datum_channels_;
}

template<typename Dtype>
void DataHeatmapLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

    CPUTimer batch_timer;
    batch_timer.Start();
    CHECK(batch->data_.count());
    HeatmapDataParameter heatmap_data_param = this->layer_param_.heatmap_data_param();

    // Pointers to blobs' float data
    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = batch->label_.mutable_cpu_data();

    cv::Mat img;
    // Shortcuts to params

    const int batchsize = heatmap_data_param.batchsize();
    const int label_height = heatmap_data_param.label_height();
    const int label_width = heatmap_data_param.label_width();
    const int outsize = heatmap_data_param.outsize();
    
    // collect "batchsize" images
    std::vector<float> cur_label;
    std::string img_name;

    // loop over non-augmented images
    for (int idx_img = 0; idx_img < batchsize; idx_img++) {
        // get image name and class
        this->GetCurImg(img_name, cur_label);

        // get number of channels for image label
        int label_num_channels = cur_label.size();

        std::string img_path = this->root_img_dir_ + img_name;
        DLOG(INFO) << "img: " << img_path;
        img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
        int width = img.cols;
        int height = img.rows;
        float resizeFact_x = (float)outsize / (float)width;
        float resizeFact_y = (float)outsize / (float)height;
        // LOG(INFO) << "width: " << width;
        const int idx_img_aug = idx_img;
        // resize to output image size

        // "resize" annotations
        for (int i = 0; i < label_num_channels; i += 2) {
            cur_label[i] *= resizeFact_x;
            cur_label[i + 1] *= resizeFact_y;
        }
        // store image data
        DLOG(INFO) << "storing image";

        top_data[idx_img] = 1;

        // store label as gaussian
        DLOG(INFO) << "storing labels";
        const int label_channel_size = label_height * label_width;
        const int label_img_size = label_channel_size * label_num_channels / 2;
        cv::Mat dataMatrix = cv::Mat::zeros(label_height, label_width, CV_32FC1);
        float label_resize_fact = (float) label_height / (float) outsize;
        float sigma = 1.5;

        for (int idx_ch = 0; idx_ch < label_num_channels / 2; idx_ch++) {
            float x = label_resize_fact * cur_label[2 * idx_ch];
            float y = label_resize_fact * cur_label[2 * idx_ch + 1];
            for (int i = 0; i < label_height; i++) {
                for (int j = 0; j < label_width; j++) {
                    int label_idx = idx_img_aug * label_img_size + idx_ch * label_channel_size + i * label_height + j;
                    float gaussian = ( 1 / ( sigma * sqrt(2 * M_PI) ) ) * exp( -0.5 * ( pow(i - y, 2.0) + pow(j - x, 2.0) ) * pow(1 / sigma, 2.0) );
                    gaussian = 4 * gaussian;
                    top_label[label_idx] = gaussian;
                    if (idx_ch == 0)
                        dataMatrix.at<float>((int)j, (int)i) = gaussian;
                }
            }
        }
        // move to the next image
        this->AdvanceCurImg();
    } // original image loop

    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

template<typename Dtype>
void DataHeatmapLayer<Dtype>::GetCurImg(string& img_name, std::vector<float>& img_label)
{
    img_name = img_label_list_[cur_img_].first;
    img_label = img_label_list_[cur_img_].second;
}

template<typename Dtype>
void DataHeatmapLayer<Dtype>::AdvanceCurImg()
{
    if (cur_img_ < img_label_list_.size() - 1)
        cur_img_++;
    else
        cur_img_ = 0;
}

INSTANTIATE_CLASS(DataHeatmapLayer);
REGISTER_LAYER_CLASS(DataHeatmap);

} // namespace caffe
