#ifndef CAFFE_PAIRWISE_POTENTIAL_LAYER_HPP_
#define CAFFE_PAIRWISE_POTENTIAL_LAYER_HPP_

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
#include "caffe/crf_layers/pairwise_feature_layer.hpp"
#include "caffe/crf_layers/pairwise_function_intensity_gaussian_layer.hpp"
#include "caffe/crf_layers/pairwise_function_bilateral_gaussian_layer.hpp"
#include "caffe/crf_layers/pairwise_function_freeform_layer.hpp"
//#include "caffe/crf_layers/feature_normalization_layer.hpp"
//#include "caffe/util/modified_permutohedral.hpp"
//#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
class PairwisePotentialLayer: public Layer<Dtype>{
public:
    // botom[0] pre-pairwise features or origin image data
    explicit PairwisePotentialLayer(const LayerParameter& param)
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
        return "PairwisePotentialLayer";
    }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
    
private:
    int count_;
    int num_;
    int channels_;
    int height_;
    int width_;
    int num_pixels_;
    int kernel_size_;
    int neighbour_number_;
    
    shared_ptr<PairwiseFeatureLayer<Dtype> > feature_layer_;
    shared_ptr<PairwiseFunctionIntensityGaussianLayer<Dtype> > function_intensity_gaussian_layer_;
    shared_ptr<PairwiseFunctionBilateralGaussianLayer<Dtype> > function_bilateral_gaussian_layer_;
    shared_ptr<PairwiseFunctionFreeformLayer<Dtype> > function_freeform_layer_;
    shared_ptr<ReshapeLayer<Dtype> > rearrange_layer_;
    
    shared_ptr<Blob<Dtype> > feature_layer_output_blob_;
    shared_ptr<Blob<Dtype> > function_output_blob_;
    shared_ptr<Blob<Dtype> > rearrange_output_blob_;
    
    vector<Blob<Dtype>* > feature_layer_bottom_vec_;
    vector<Blob<Dtype>* > feature_layer_top_vec_;
    vector<Blob<Dtype>* > function_layer_bottom_vec_;
    vector<Blob<Dtype>* > function_layer_top_vec_;
    vector<Blob<Dtype>* > rearrange_layer_bottom_vec_;
    vector<Blob<Dtype>* > rearrange_layer_top_vec_;
};
    
}  // namespace caffe

#endif  // CAFFE_PAIRWISE_POTENTIAL_LAYER_HPP_
