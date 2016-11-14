#ifndef CAFFE_UNARY_COMPOSITE_LAYER_HPP_
#define CAFFE_UNARY_COMPOSITE_LAYER_HPP_

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
#include "caffe/layers/reshape_layer.hpp"

#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
class UnaryCompositeLayer: public Layer<Dtype>{
public:
    // bottom[0] unary potential learned from CNN for C channels
    // bottom[1] origin image data with scribble distance in the last C channels
    // top[0] compositied unary potential
    // top[1] interaction mask to indicate whether one pixel belongs to interaction
    explicit UnaryCompositeLayer(const LayerParameter& param)
    : Layer<Dtype>(param){};
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
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
        return "UnaryCompositeLayer";
    }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 2; }
    
private:
    int count_;
    int num_;
    int unary_channels_;
    int image_channels_;
    int height_;
    int width_;
    int num_pixels_;
};
    
}  // namespace caffe

#endif  // CAFFE_UNARY_COMPOSITE_LAYER_HPP_
