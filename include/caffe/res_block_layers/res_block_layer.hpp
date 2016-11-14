#ifndef CAFFE_RESIDUAL_BLOCK_LAYER_HPP_
#define CAFFE_RESIDUAL_BLOCK_LAYER_HPP_

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
#include "caffe/layers/dropout_layer.hpp"
//#include "caffe/crf_layers/pairwise_potential_layer.hpp"
//#include "caffe/util/modified_permutohedral.hpp"
//#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
class ResidualBlockLayer:public Layer<Dtype>{
public:
    explicit ResidualBlockLayer(const LayerParameter& param)
    : Layer<Dtype>(param){}
    // bottom[0] is unary term image: size N, C, H, W
    // bottom[1] is pairwise term image: size N, neighN, H, W
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const {
        return "ResidualBlockLayer";
    }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
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
    int output_channels_;
    int height_;
    int width_;
    int num_pixels_;
    bool enable_residual_;
    int layerN_;
    vector<shared_ptr<Blob<Dtype> > > split_out_blobs_;
    vector<shared_ptr<Blob<Dtype> > > conv_out_blobs_;
    vector<shared_ptr<Blob<Dtype> > > drop_out_blobs_;
    vector<shared_ptr<Blob<Dtype> > > relu_out_blobs_;
    
    vector<Blob<Dtype> *> split_bottom_vec_;
    vector<Blob<Dtype> *> split_top_vec_;

    vector<vector<Blob<Dtype> *> > conv_bottom_vec_;
    vector<vector<Blob<Dtype> *> > conv_top_vec_;
    vector<vector<Blob<Dtype> *> > drop_bottom_vec_;
    vector<vector<Blob<Dtype> *> > drop_top_vec_;
    vector<vector<Blob<Dtype> *> > relu_bottom_vec_;
    vector<vector<Blob<Dtype> *> > relu_top_vec_;
    
    vector<Blob<Dtype> *> sum_bottom_vec_;
    vector<Blob<Dtype> *> sum_top_vec_;
    vector<shared_ptr<ConvolutionLayer<Dtype> > > conv_layers_;
    vector<shared_ptr<ReLULayer<Dtype> > > relu_layers_;
    vector<shared_ptr<DropoutLayer<Dtype> > > drop_layers_;
    shared_ptr<SplitLayer<Dtype> > split_layer_;
    shared_ptr<EltwiseLayer<Dtype> > sum_layer_;
    
};
}  // namespace caffe

#endif  // CAFFE_RESIDUAL_BLOCK_LAYER_HPP_
