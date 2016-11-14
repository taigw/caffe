#ifndef CAFFE_PAIRWISE_FUNCTION_FREEFORM_LAYER_HPP_
#define CAFFE_PAIRWISE_FUNCTION_FREEFORM_LAYER_HPP_

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
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
//#include "caffe/util/modified_permutohedral.hpp"
//#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

namespace caffe {
    
template <typename Dtype>
class PairwiseFunctionFreeformLayer:public Layer<Dtype>{
public:
    explicit PairwiseFunctionFreeformLayer(const LayerParameter& param)
    : Layer<Dtype>(param){}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
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

    virtual inline const char* type() const {
        return "PairwiseFunctionFreeformLayer";
    }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
private:
    bool LoadParamFromFile(shared_ptr<vector<vector<Dtype> > > matrix,
                         int H, int W, std::string filename);
    bool CopyParamToBlob(shared_ptr<vector<vector<Dtype> > > matrix,
                         shared_ptr<Blob<Dtype> > blob);
    int count_;
    int num_;
    int channels_;
    int height_;
    int width_;

    vector<int> inner_layer_size_;
    int layerN_;
    
    vector<shared_ptr<Blob<Dtype> > > conv_out_blobs_;
    vector<shared_ptr<Blob<Dtype> > > relu_out_blobs_;
    vector<vector<Blob<Dtype> *> > conv_bottom_vec_;
    vector<vector<Blob<Dtype> *> > conv_top_vec_;
    vector<vector<Blob<Dtype> *> > relu_bottom_vec_;
    vector<vector<Blob<Dtype> *> > relu_top_vec_;
    vector<shared_ptr<ConvolutionLayer<Dtype> > > conv_layers_;
    vector<shared_ptr<ReLULayer<Dtype> > > relu_layers_;
    
};
    
}  // namespace caffe

#endif  // CAFFE_PAIRWISE_FUNCTION_LAYER_HPP_
