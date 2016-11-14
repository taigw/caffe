#ifndef CAFFE_MESSAGE_PASSING_LAYER_HPP_
#define CAFFE_MESSAGE_PASSING_LAYER_HPP_

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
#include "caffe/crf_layers/pairwise_potential_layer.hpp"
//#include "caffe/util/modified_permutohedral.hpp"
//#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
class MessagePassingLayer:public Layer<Dtype>{
public:
    explicit MessagePassingLayer(const LayerParameter& param)
    : Layer<Dtype>(param){}
    // bottom[0] is unary term image: size N, C, H, W
    // bottom[1] is pairwise term image: size N, neighN, H, W
    // bottom[2] interaction_mask
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const {
        return "MessagePassingLayer";
    }
    virtual inline int ExactNumBottomBlobs() const { return 3; }
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
    int kernel_size_;
    int neighN_;
    bool user_interaction_constrain_;
};
}  // namespace caffe

#endif  // CAFFE_MESSAGE_PASSING_LAYER_HPP_
