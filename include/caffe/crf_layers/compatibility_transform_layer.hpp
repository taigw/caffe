#ifndef CAFFE_COMPATIBILITY_TRANSFORM_LAYER_HPP_
#define CAFFE_COMPATIBILITY_TRANSFORM_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#include "caffe/layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"


#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
class CompatibilityTransformLayer:public Layer<Dtype>{
public:
    explicit CompatibilityTransformLayer(const LayerParameter& param)
    : Layer<Dtype>(param){}
    // bottom[0] is unary term image
    // bottom[1] is compabitility matrix
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

private:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);

    virtual inline const char* type() const {
        return "CompatibilityTransform";
    }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    
    int count_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int num_pixels_;
};
}  // namespace caffe

#endif  // CAFFE_COMPATIBILITY_TRANSFORM_LAYER_HPP_
