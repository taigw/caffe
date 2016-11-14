#ifndef CAFFE_MULTI_STAGE_CRF_LAYER_HPP_
#define CAFFE_MULTI_STAGE_CRF_LAYER_HPP_

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
#include "caffe/crf_layers/unary_composite_layer.hpp"
#include "caffe/crf_layers/pairwise_potential_layer.hpp"
//#include "caffe/crf_layers/pairwise_potential_simple_layer.hpp"
#include "caffe/crf_layers/crf_iteration_layer.hpp"
//#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
class MultiStageCRFLayer : public Layer<Dtype> {

 public:
  // bottom[0] original image data
  // bottom[1] unary potential input
  // bottom[2] pre-pairwise potential input
  explicit MultiStageCRFLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "MultiStageCRF";
  }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int count_;
  int num_;
  int img_channels_;
  int cls_channels_;
  int height_;
  int width_;
  int num_pixels_;
  bool user_interaction_constrain_;


  int num_iterations_;
  shared_ptr<Blob<Dtype> > compatibility_blob_;
  vector<shared_ptr<Blob<Dtype> > > compatibility_split_blobs_;
  vector<Blob<Dtype>*> compatibility_split_layer_bottom_vec_;
  vector<Blob<Dtype>*> compatibility_split_layer_top_vec_;
  shared_ptr<SplitLayer<Dtype> > compatibility_split_layer_;
 
  shared_ptr<Blob<Dtype> > unary_composite_blob_;
  shared_ptr<Blob<Dtype> > interaction_mask_blob_;
  vector<Blob<Dtype>*> unary_composite_layer_bottom_vec_;
  vector<Blob<Dtype>*> unary_composite_layer_top_vec_;
  shared_ptr<UnaryCompositeLayer<Dtype> > unary_composite_layer_;
    
  vector<Blob<Dtype>*> unary_split_layer_bottom_vec_;
  vector<Blob<Dtype>*> unary_split_layer_top_vec_;
  vector<shared_ptr<Blob<Dtype> > > unary_split_output_blobs_;
  shared_ptr<SplitLayer<Dtype> > unary_split_layer_;
    
  vector<Blob<Dtype>*> pair_split_layer_bottom_vec_;
  vector<Blob<Dtype>*> pair_split_layer_top_vec_;
  vector<shared_ptr<Blob<Dtype> > > pair_split_output_blobs_;
  shared_ptr<SplitLayer<Dtype> > pair_split_layer_;
    
  vector<Blob<Dtype>*> pairwise_layer_bottom_vec_;
  vector<Blob<Dtype>*> pairwise_layer_top_vec_;
  shared_ptr<Blob<Dtype> > pairwise_layer_output_blob_;
  shared_ptr<PairwisePotentialLayer<Dtype> > pairwise_layer_;

  vector<shared_ptr<Blob<Dtype> > > iteration_output_blobs_;
  vector<vector<Blob<Dtype>* >  > interation_bottom_vecs_;
  vector<vector<Blob<Dtype>* >  > interation_top_vecs_;
  vector<shared_ptr<CRFIterationLayer<Dtype> > > crf_iterations_;
};

}  // namespace caffe

#endif  // CAFFE_MULTI_STAGE_MEAN_FIELD_LAYER_HPP_
