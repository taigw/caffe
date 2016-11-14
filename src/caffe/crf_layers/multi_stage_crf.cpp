/*!
 *  \brief     The Caffe layer that implements the CRF-RNN described in the paper:
 *             Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *  \authors   Sadeep Jayasumana, Bernardino Romera-Paredes, Shuai Zheng, Zhizhong Su.
 *  \version   1.0
 *  \date      2015
 *  \copyright Torr Vision Group, University of Oxford.
 *  \details   If you use this code, please consider citing the paper:
 *             Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du,
 *             Chang Huang, Philip H. S. Torr. Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
 */
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/tvg_util.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/modified_permutohedral.hpp"
#include "caffe/crf_layers/multi_stage_crf_layer.hpp"

#include <cmath>

namespace caffe {

template <typename Dtype>
void MultiStageCRFLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  //LOG(INFO) << ("MultiStageCRFLayer start.");
  const caffe::MultiStageCRFParameter multi_crf_param = this->layer_param_.multi_stage_crf_param();
  num_iterations_ = multi_crf_param.num_iterations();
  user_interaction_constrain_ = multi_crf_param.user_interaction_constrain();
  CHECK_GT(num_iterations_, 1) << "Number of iterations must be greater than 1.";

  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  img_channels_ = bottom[0]->channels();
  cls_channels_ = bottom[1]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
    
  num_pixels_ = height_ * width_;

  // set compatibility param blob and split for each iteration, the blob size is fixed
  compatibility_blob_.reset(new Blob<Dtype>(1, 1, cls_channels_, cls_channels_));
  caffe_set(cls_channels_ * cls_channels_, Dtype(1.), compatibility_blob_->mutable_cpu_data());
  for (int c = 0; c < cls_channels_; ++c) {
    (compatibility_blob_->mutable_cpu_data())[c * cls_channels_ + c] = Dtype(0.0);
  }
  compatibility_split_layer_bottom_vec_.clear();
  compatibility_split_layer_bottom_vec_.push_back(compatibility_blob_.get());
  compatibility_split_layer_top_vec_.clear();
  compatibility_split_blobs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_ ; i++) {
      compatibility_split_blobs_[i].reset(new Blob<Dtype>());
      compatibility_split_layer_top_vec_.push_back(compatibility_split_blobs_[i].get());
  }
  LayerParameter split_layer_param;
  compatibility_split_layer_.reset(new SplitLayer<Dtype>(split_layer_param));
  compatibility_split_layer_->SetUp(compatibility_split_layer_bottom_vec_, compatibility_split_layer_top_vec_);
    
  //  generate the pairwise potential. size: (N, neighN, H, W)
  //  where neighN is the number of neighbour, equal to kernel_size_*kernel_size_-1.
  pairwise_layer_bottom_vec_.clear();
//  pairwise_layer_bottom_vec_.push_back(bottom[2]);
  pairwise_layer_bottom_vec_.push_back(bottom[0]);

  pairwise_layer_output_blob_.reset(new Blob<Dtype>());
  pairwise_layer_top_vec_.clear();
  pairwise_layer_top_vec_.push_back(pairwise_layer_output_blob_.get());
  
  //LOG(INFO) << ("trying to create PairwisePotentialLayer");
  pairwise_layer_.reset(new PairwisePotentialLayer<Dtype>(this->layer_param_));
  pairwise_layer_->SetUp(pairwise_layer_bottom_vec_, pairwise_layer_top_vec_);
  //LOG(INFO) << ("create PairwisePotentialLayer done.");
  // Configure the split layer that is used to make copies of the unary term. One copy for each iteration.
  // An extra copy is created for the softmax input of iteration 0
  // It may be possible to optimize this calculation later.
  
  // add user interaction constrain to unary potential
  if(user_interaction_constrain_)
  {
      unary_composite_blob_.reset(new Blob<Dtype>());
      interaction_mask_blob_.reset(new Blob<Dtype>());
      unary_composite_layer_bottom_vec_.clear();
      unary_composite_layer_bottom_vec_.push_back(bottom[1]); // unary potential input
      unary_composite_layer_bottom_vec_.push_back(bottom[0]); // original image data

      unary_composite_layer_top_vec_.clear();
      unary_composite_layer_top_vec_.push_back(unary_composite_blob_.get());
      unary_composite_layer_top_vec_.push_back(interaction_mask_blob_.get());

      unary_composite_layer_.reset(new UnaryCompositeLayer<Dtype>(this->layer_param_));
      unary_composite_layer_->SetUp(unary_composite_layer_bottom_vec_,unary_composite_layer_top_vec_);
  }
    
  // gnerate unary potential for each iteration
  unary_split_layer_bottom_vec_.clear();
  if(user_interaction_constrain_){
    unary_split_layer_bottom_vec_.push_back(unary_composite_blob_.get());
  }
  else{
    unary_split_layer_bottom_vec_.push_back(bottom[1]);
  }

  unary_split_layer_top_vec_.clear();
  unary_split_output_blobs_.resize(num_iterations_ + 1); // the last one serves as the softmax input of iter 0
  for (int i = 0; i < num_iterations_ + 1; i++) {
    unary_split_output_blobs_[i].reset(new Blob<Dtype>());
    unary_split_layer_top_vec_.push_back(unary_split_output_blobs_[i].get());
  }

  unary_split_layer_.reset(new SplitLayer<Dtype>(split_layer_param));
  unary_split_layer_->SetUp(unary_split_layer_bottom_vec_, unary_split_layer_top_vec_);
  //LOG(INFO) << ("unary split layer done.");

  // make copies of the pairwise potential term. One copy for each iteration.
  pair_split_layer_bottom_vec_.clear();
  pair_split_layer_bottom_vec_.push_back(pairwise_layer_output_blob_.get());

  pair_split_layer_top_vec_.clear();
  pair_split_output_blobs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; i++) {
    pair_split_output_blobs_[i].reset(new Blob<Dtype>());
    pair_split_layer_top_vec_.push_back(pair_split_output_blobs_[i].get());
  }

  pair_split_layer_.reset(new SplitLayer<Dtype>(split_layer_param));
  pair_split_layer_->SetUp(pair_split_layer_bottom_vec_, pair_split_layer_top_vec_);
  //LOG(INFO) << ("pair split layer done.");
    
  // Make blobs to store outputs of each meanfield iteration. Output of the last iteration is stored in top[0].
  // So we need only (num_iterations_ - 1) blobs.
  iteration_output_blobs_.resize(num_iterations_ - 1);
  for (int i = 0; i < num_iterations_ - 1; ++i) {
    iteration_output_blobs_[i].reset(new Blob<Dtype>(num_, cls_channels_, height_, width_));
  }

  // Make instances of CRFIteration and initialize them.
  crf_iterations_.resize(num_iterations_);
  interation_bottom_vecs_.resize(num_iterations_);
  interation_top_vecs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    vector<Blob<Dtype>* > one_iter_bottom_vec_;
    one_iter_bottom_vec_.resize(5);
    one_iter_bottom_vec_[0] = unary_split_output_blobs_[i].get(); //unary_term
    one_iter_bottom_vec_[1] = (i==0)? unary_split_output_blobs_[num_iterations_].get() : iteration_output_blobs_[i-1].get();
    one_iter_bottom_vec_[2] = pair_split_output_blobs_[i].get();
    one_iter_bottom_vec_[3] = compatibility_split_blobs_[i].get();
    one_iter_bottom_vec_[4] = user_interaction_constrain_ ? interaction_mask_blob_.get(): NULL;
    interation_bottom_vecs_[i] = one_iter_bottom_vec_;
      
    vector<Blob<Dtype>* > one_iter_top_vec_;
    one_iter_top_vec_.resize(1);
    one_iter_top_vec_[0] = (i==num_iterations_-1)? top[0] : iteration_output_blobs_[i].get();
    interation_top_vecs_[i] = one_iter_top_vec_;
    
    //LOG(INFO) << ("crf iteration start ")<< i;
    crf_iterations_[i].reset(new CRFIterationLayer<Dtype>(this->layer_param_));
    crf_iterations_[i]->SetUp(interation_bottom_vecs_[i], interation_top_vecs_[i]);
  }
  //LOG(INFO) << ("MultiStageCRFLayer initialized.");
    
   //
  this->blobs_.clear();
  this->blobs_.insert(this->blobs_.begin(), pairwise_layer_->blobs().begin(), pairwise_layer_->blobs().end());
  this->blobs_.push_back(compatibility_blob_);
  LOG(INFO) << "multi stage crf blob size "<< this->blobs_.size();
}

template <typename Dtype>
void MultiStageCRFLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//    std::cout<<"multistagecrf reshape start"<<std::endl;
    pairwise_layer_bottom_vec_[0] = bottom[0];
    pairwise_layer_->Reshape(pairwise_layer_bottom_vec_, pairwise_layer_top_vec_);
//    std::cout<<"multistagecrf reshape pairwise layer finished"<<std::endl;
    if(user_interaction_constrain_)
    {
        unary_composite_layer_bottom_vec_[0]=bottom[1]; // unary potential input
        unary_composite_layer_bottom_vec_[1]=bottom[0]; // original image data
        unary_composite_layer_->Reshape(unary_composite_layer_bottom_vec_,unary_composite_layer_top_vec_);
    }
//    std::cout<<"multistagecrf reshape unary composite finished"<<std::endl;
    if(user_interaction_constrain_){
        unary_split_layer_bottom_vec_[0] = unary_composite_blob_.get();
    }
    else{
        unary_split_layer_bottom_vec_[0] = bottom[1];
    }
    unary_split_layer_->Reshape(unary_split_layer_bottom_vec_, unary_split_layer_top_vec_);
//    std::cout<<"multistagecrf reshape unary split finished"<<std::endl;
    
    pair_split_layer_->Reshape(pair_split_layer_bottom_vec_, pair_split_layer_top_vec_);
    for (int i = 0; i < num_iterations_; ++i) {
        if(i==num_iterations_ -1)
        {
            interation_top_vecs_[i][0] = top[0];
        }
        crf_iterations_[i]->Reshape(interation_bottom_vecs_[i], interation_top_vecs_[i]);
    }
//    std::cout<<"multistagecrf reshape crf iteration finished"<<std::endl;
}

template <typename Dtype>
void MultiStageCRFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

//  pairwise_layer_bottom_vec_[0] = bottom[0];
//  unary_split_layer_bottom_vec_[0] = bottom[1];
//  interation_top_vecs_[num_iterations_-1][0] = top[0];

  if(user_interaction_constrain_){
      unary_composite_layer_->Forward(unary_composite_layer_bottom_vec_, unary_composite_layer_top_vec_);
//      LOG(INFO) << ("unary_composite_layer_. Forward_cpu done.");
  }
//  std::cout<<"multistagecrf unary composite finished"<<std::endl;

  compatibility_split_layer_->Forward(compatibility_split_layer_bottom_vec_, compatibility_split_layer_top_vec_);
//  std::cout<<"multistagecrf composite split finished"<<std::endl;
  unary_split_layer_->Forward(unary_split_layer_bottom_vec_, unary_split_layer_top_vec_);
//  std::cout<<"multistagecrf unary split finished"<<std::endl;

  pairwise_layer_->Forward(pairwise_layer_bottom_vec_, pairwise_layer_top_vec_);
//  std::cout<<"multistagecrf pairwise layer finished"<<std::endl;
  pair_split_layer_->Forward(pair_split_layer_bottom_vec_, pair_split_layer_top_vec_);
//  std::cout<<"multistagecrf pairwise split layer finished"<<std::endl;

  for (int i = 0; i < num_iterations_; i++) {
    crf_iterations_[i]->Forward(interation_bottom_vecs_[i], interation_top_vecs_[i]);
  }
//  std::cout<<"multistagecrf finished"<<std::endl;

//  LOG(INFO) << ("MultiStageCRFLayer. Forward_cpu done.");
}

template <typename Dtype>
void MultiStageCRFLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
    Forward_cpu(bottom, top);
}
/**
 * Backprop through filter-based mean field inference.
 */
template<typename Dtype>
void MultiStageCRFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& bottom)
{
//  pairwise_layer_bottom_vec_[0] = bottom[0];
//  unary_split_layer_bottom_vec_[0] = bottom[1];
//  interation_top_vecs_[num_iterations_-1][0] = top[0];
//  std::cout<<"back multistagecrf start"<<std::endl;
  for (int i = (num_iterations_ - 1); i >= 0; i--) {
    vector<bool> iter_propagate_down(3, true);
    crf_iterations_[i]->Backward(interation_top_vecs_[i], iter_propagate_down, interation_bottom_vecs_[i]);
    //LOG(INFO) << ("crf iteration done ")<<i;
  }
//  std::cout<<"back crf iteration finished"<<std::endl;
  vector<bool> pair_split_propagate_down(1, true);
  pair_split_layer_->Backward(pair_split_layer_top_vec_, pair_split_propagate_down, pair_split_layer_bottom_vec_);
//  std::cout<<"back pair split  finished"<<std::endl;
  vector<bool> pairwise_propagate_down(1, true);
  pairwise_layer_->Backward(pairwise_layer_top_vec_, pairwise_propagate_down, pairwise_layer_bottom_vec_);
//  std::cout<<"back pairwise layer finished"<<std::endl;
  vector<bool> unary_split_propagate_down(1, true);
  unary_split_layer_->Backward(unary_split_layer_top_vec_, unary_split_propagate_down, unary_split_layer_bottom_vec_);
//  std::cout<<"back unary split finished"<<std::endl;
  vector<bool> compatibility_split_propagate_down(1, true);
  compatibility_split_layer_->Backward(compatibility_split_layer_top_vec_, compatibility_split_propagate_down, compatibility_split_layer_bottom_vec_);

  if(user_interaction_constrain_){
    vector<bool> unary_composite_propagate_down(1, true);
    unary_composite_layer_->Backward(unary_composite_layer_top_vec_, unary_composite_propagate_down,
                                 unary_composite_layer_bottom_vec_);
  }
//  std::cout<<"back multistage finished"<<std::endl;
}

template<typename Dtype>
void MultiStageCRFLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}
INSTANTIATE_CLASS(MultiStageCRFLayer);
REGISTER_LAYER_CLASS(MultiStageCRF);
}  // namespace caffe
