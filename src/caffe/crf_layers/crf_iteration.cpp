/*!
 *  \brief     A helper class for {@link MultiStageMeanfieldLayer} class, which is the Caffe layer that implements the
 *             CRF-RNN described in the paper: Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             This class itself is not a proper Caffe layer although it behaves like one to some degree.
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
#include <math.h>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/crf_layers/crf_iteration_layer.hpp"
#include "caffe/crf_layers/pixel_access.hpp"

namespace caffe {
/**
 * To be invoked once only immediately after construction.
 */
template <typename Dtype>
void CRFIterationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top)
{
    // bottom[0] unary_term
    // bottom[1] softmax_input
    // bottom[2] pairwise_term
    // bottom[3] compatibility param
    // bottom[4] interaction_mask
  //LOG(INFO) << ("entered CRFIterationLayer ");
  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_pixels_ = height_ * width_;
 
  // Initialize the blob
  softmax_output_blob_.reset(new Blob<Dtype>());
  message_passing_output_blob_.reset(new Blob<Dtype>());
  compatibility_output_blob_.reset(new Blob<Dtype>(num_, channels_, height_, width_));
  
  // Softmax layer configuration
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[1]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(softmax_output_blob_.get());

  LayerParameter softmax_param;
  softmax_layer_.reset(new SoftmaxLayer<Dtype>(softmax_param));
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  //LOG(INFO) << ("softmax layer created ");

  // Message passing layer configuration
  message_passing_bottom_vec_.clear();
  message_passing_bottom_vec_.push_back(softmax_output_blob_.get());
  message_passing_bottom_vec_.push_back(bottom[2]);
  message_passing_bottom_vec_.push_back(bottom[4]);
  message_passing_top_vec_.clear();
  message_passing_top_vec_.push_back(message_passing_output_blob_.get());
  
  message_passing_layer_.reset(new MessagePassingLayer<Dtype>(this->layer_param_));
  message_passing_layer_->SetUp(message_passing_bottom_vec_, message_passing_top_vec_);
  //LOG(INFO) << ("message passing layer created ");

  // Compatibility Transform Layer configuration
  compatibility_trans_bottom_vec_.clear();
  compatibility_trans_bottom_vec_.push_back(message_passing_output_blob_.get());
  compatibility_trans_bottom_vec_.push_back(bottom[3]);
  compatibility_trans_top_vec_.clear();
  compatibility_trans_top_vec_.push_back(compatibility_output_blob_.get());
    
  LayerParameter layer_param;
  compatibility_trans_layer_.reset(new CompatibilityTransformLayer<Dtype>(layer_param));
  compatibility_trans_layer_->SetUp(compatibility_trans_bottom_vec_,compatibility_trans_top_vec_);
  //LOG(INFO) << ("compatibility layer created ");

  // Sum layer configuration
  sum_bottom_vec_.clear();
  sum_bottom_vec_.push_back(bottom[0]);
  sum_bottom_vec_.push_back(compatibility_output_blob_.get());
  sum_top_vec_.clear();
  sum_top_vec_.push_back(top[0]);

  LayerParameter sum_param;
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.));
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(-1.));
  sum_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  sum_layer_.reset(new EltwiseLayer<Dtype>(sum_param));
  sum_layer_->SetUp(sum_bottom_vec_, sum_top_vec_);
  //LOG(INFO) << ("sum layer created ");

}

/**
 * To be invoked before every call to the Forward_cpu() method.
 */
template <typename Dtype>
void CRFIterationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top)
{
    softmax_bottom_vec_[0] = bottom[1];
    softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
    
    message_passing_bottom_vec_[1]=bottom[2];
    message_passing_bottom_vec_[2]=bottom[4];
    message_passing_layer_->Reshape(message_passing_bottom_vec_, message_passing_top_vec_);
    
    compatibility_trans_bottom_vec_[1]=bottom[3];
    compatibility_trans_layer_->Reshape(compatibility_trans_bottom_vec_,compatibility_trans_top_vec_);
    
    sum_bottom_vec_[0] = bottom[0];
    sum_top_vec_[0] = top[0];
    sum_layer_->Reshape(sum_bottom_vec_, sum_top_vec_);
}

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void CRFIterationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top)
{
    
//    const Dtype * bottom_data = bottom[0]->cpu_data();
//    Dtype * top_data = top[0]->mutable_cpu_data();
//    for(int n=0; n<bottom[0]->count(); n++)
//    {
//        top_data[n] = bottom_data[n];
//    }
//  sum_bottom_vec_[0] = bottom[0];
//  softmax_bottom_vec_[0] = bottom[1];
//  message_passing_bottom_vec_[1]=bottom[2];
//  compatibility_trans_bottom_vec_[1] = bottom[3];
//  sum_top_vec_[0]=top[0];
    
  //------------------------------- Softmax normalization--------------------
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  CHECK(message_passing_bottom_vec_[0]->height() == message_passing_bottom_vec_[1]->height() &&
        message_passing_bottom_vec_[0]->width() == message_passing_bottom_vec_[1]->width())<<
    ("input image and kernel shoud have the pixel number");
    
  //-----------------------------------Message passing-----------------------
  message_passing_layer_->Forward(message_passing_bottom_vec_, message_passing_top_vec_);
  //LOG(INFO)<<"message_passing_layer_->Forward done.";
  //--------------------------- Compatibility multiplication ----------------
  //Result from message passing needs to be multiplied with compatibility values.
  compatibility_trans_layer_->Forward(compatibility_trans_bottom_vec_, compatibility_trans_top_vec_);

  //------------------------- Adding unaries, normalization is left to the next iteration --------------
  // Add unary
  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
  
}

template <typename Dtype>
void CRFIterationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}

template<typename Dtype>
void CRFIterationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
//    Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
//    const Dtype * top_diff = top[0]->cpu_diff();
//    for(int n=0; n<bottom[0]->count(); n++)
//    {
//        bottom_diff[n] = top_diff[n];
//    }
    
//  sum_bottom_vec_[0] = bottom[0];
//  softmax_bottom_vec_[0] = bottom[1];
//  message_passing_bottom_vec_[1]=bottom[2];
//  compatibility_trans_bottom_vec_[1] = bottom[3];
//  sum_top_vec_[0]=top[0];
  //LOG(INFO) << ("crf iteration start.");
  //---------------------------- Add unary gradient --------------------------
  vector<bool> eltwise_propagate_down(2, true);
  sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);
  //LOG(INFO) << ("sum layer backward done.");

  //---------------------------- Update compatibility diffs ------------------

  //-------------------------- Gradient after compatibility transform--- -----
  vector<bool> compatibility_propagate_down(2, true);
  compatibility_trans_layer_->Backward(compatibility_trans_top_vec_, compatibility_propagate_down, compatibility_trans_bottom_vec_);
  //LOG(INFO) << ("compatibility backward done.");

  //--------------------------- Gradient for message passing ---------------
  vector<bool> message_passing_propagate_down(2, true);
  message_passing_layer_->Backward(message_passing_top_vec_, message_passing_propagate_down,
                                       message_passing_bottom_vec_);
  //LOG(INFO) << ("message pasing backward done.");
  //--------------------------------------------------------------------------------
  vector<bool> softmax_propagate_down(1, true);
  softmax_layer_->Backward(softmax_top_vec_, softmax_propagate_down, softmax_bottom_vec_);
  //LOG(INFO) << ("softmax backward done.");
}

template<typename Dtype>
void CRFIterationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(CRFIterationLayer);
}  // namespace caffe
