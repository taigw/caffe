#ifndef CAFFE_PIXEL_ACCESS_HPP_
#define CAFFE_PIXEL_ACCESS_HPP_

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
#include "caffe/crf_layers/compatibility_transform_layer.hpp"
#include "caffe/crf_layers/message_passing_layer.hpp"
//#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
class CRFIterationLayer :public Layer<Dtype>{
public:
    explicit CRFIterationLayer(const LayerParameter& param)
    : Layer<Dtype>(param){}
    // bottom[0] unary_term
    // bottom[1] softmax_input
    // bottom[2] pairwise_term
    // bottom[3] compatibility param
    // bottom[4] interaction_mask
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
        return "CRFIteration";
    }
    virtual inline int ExactNumBottomBlobs() const { return 5; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
protected:
    int count_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int num_pixels_;
    

    shared_ptr<Blob<Dtype> > softmax_output_blob_;
    shared_ptr<Blob<Dtype> > message_passing_output_blob_;
    shared_ptr<Blob<Dtype> > compatibility_output_blob_;
    // sum_layer_output_blob_ is the top blob
    
    
    vector<Blob<Dtype>*> softmax_top_vec_;
    vector<Blob<Dtype>*> softmax_bottom_vec_;
    vector<Blob<Dtype>*> message_passing_top_vec_;
    vector<Blob<Dtype>*> message_passing_bottom_vec_;
    vector<Blob<Dtype>*> compatibility_trans_top_vec_;
    vector<Blob<Dtype>*> compatibility_trans_bottom_vec_;
    vector<Blob<Dtype>*> sum_top_vec_;
    vector<Blob<Dtype>*> sum_bottom_vec_;
    
    shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
    shared_ptr<MessagePassingLayer<Dtype> > message_passing_layer_;
    shared_ptr<CompatibilityTransformLayer<Dtype> > compatibility_trans_layer_;
    shared_ptr<EltwiseLayer<Dtype> > sum_layer_;
};

}

#endif  // CAFFE_PIXEL_ACCESS_HPP_
