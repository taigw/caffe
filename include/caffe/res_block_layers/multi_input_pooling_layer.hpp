#ifndef CAFFE_MULTI_INPUT_POOLING_LAYER_HPP_
#define CAFFE_MULTI_INPUT_POOLING_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
//#include "caffe/crf_layers/pairwise_potential_layer.hpp"
//#include "caffe/util/modified_permutohedral.hpp"
//#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
class MultiInputPoolingLayer:public Layer<Dtype>{
public:
    explicit MultiInputPoolingLayer(const LayerParameter& param)
    : Layer<Dtype>(param){}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const {
        return "MultiInputPoolingLayer";
    }
//    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
private:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);

    int count_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int num_pixels_;
    int inputN_;
};
}  // namespace caffe

#endif  // CAFFE_RESIDUAL_BLOCK_LAYER_HPP_
