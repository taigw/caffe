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
#include "caffe/crf_layers/pairwise_potential_layer.hpp"
#include <iostream>

namespace caffe {
    
template <typename Dtype>
void PairwisePotentialLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    kernel_size_ =  this->layer_param_.multi_stage_crf_param().kernel_size();
//    caffe::PairwisePotentialType potential_type = ;
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    
    feature_layer_output_blob_.reset(new Blob<Dtype>());
    function_output_blob_.reset(new Blob<Dtype>());
    rearrange_output_blob_.reset(new Blob<Dtype>());
    
    feature_layer_bottom_vec_.clear();
    feature_layer_bottom_vec_.push_back(bottom[0]);
    feature_layer_top_vec_.clear();
    feature_layer_top_vec_.push_back(feature_layer_output_blob_.get());
    feature_layer_.reset(new PairwiseFeatureLayer<Dtype>(this->layer_param_));// construct function without parameter
    feature_layer_->SetUp(feature_layer_bottom_vec_, feature_layer_top_vec_);

    LayerParameter layer_param;
    function_layer_bottom_vec_.clear();
    function_layer_bottom_vec_.push_back(feature_layer_output_blob_.get());
    function_layer_top_vec_.clear();
    function_layer_top_vec_.push_back(function_output_blob_.get());
    
    if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
       MultiStageCRFParameter_PairwisePotentialType_INTENSITY_GAUSSIAN)
    {
        function_intensity_gaussian_layer_.reset(
            new PairwiseFunctionIntensityGaussianLayer<Dtype>(this->layer_param_));
        function_intensity_gaussian_layer_->SetUp(function_layer_bottom_vec_, function_layer_top_vec_);
    }
    else if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
       MultiStageCRFParameter_PairwisePotentialType_BILATERAL_GAUSSIAN)
    {
        function_bilateral_gaussian_layer_.reset(
                new PairwiseFunctionBilateralGaussianLayer<Dtype>(this->layer_param_));
        function_bilateral_gaussian_layer_->SetUp(function_layer_bottom_vec_, function_layer_top_vec_);
    }
    else{
        function_freeform_layer_.reset(new PairwiseFunctionFreeformLayer<Dtype>(this->layer_param_));
        function_freeform_layer_->SetUp(function_layer_bottom_vec_, function_layer_top_vec_);
    }
    
    
    rearrange_layer_bottom_vec_.clear();
    rearrange_layer_bottom_vec_.push_back(function_output_blob_.get());
    rearrange_layer_top_vec_.clear();
    rearrange_layer_top_vec_.push_back(top[0]);
    
    LayerParameter reshape_param;
    reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(num_);
    reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(kernel_size_*kernel_size_-1);
    reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(height_);
    reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(width_);
    rearrange_layer_.reset(new ReshapeLayer<Dtype>(reshape_param)); // construct function without parameter
    rearrange_layer_->SetUp(rearrange_layer_bottom_vec_, rearrange_layer_top_vec_);
    
    this->blobs_.clear();
    if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
       MultiStageCRFParameter_PairwisePotentialType_INTENSITY_GAUSSIAN)
    {
        this->blobs_.insert(this->blobs_.begin(),
                            function_intensity_gaussian_layer_->blobs().begin(),
                            function_intensity_gaussian_layer_->blobs().end());
    }
    else if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
       MultiStageCRFParameter_PairwisePotentialType_BILATERAL_GAUSSIAN)
    {
        this->blobs_.insert(this->blobs_.begin(),
                            function_bilateral_gaussian_layer_->blobs().begin(),
                            function_bilateral_gaussian_layer_->blobs().end());
    }
    else{
        this->blobs_.insert(this->blobs_.begin(),
                            function_freeform_layer_->blobs().begin(),
                            function_freeform_layer_->blobs().end());
    }
}


template <typename Dtype>
void PairwisePotentialLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top)
{
    
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();

//    std::cout<<"pair wise potential layer start"<<std::endl;
    feature_layer_bottom_vec_[0] = bottom[0];
    feature_layer_->Reshape(feature_layer_bottom_vec_, feature_layer_top_vec_);
    
    if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
       MultiStageCRFParameter_PairwisePotentialType_INTENSITY_GAUSSIAN)
    {
        function_intensity_gaussian_layer_->Reshape(function_layer_bottom_vec_, function_layer_top_vec_);
    }
    else if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
            MultiStageCRFParameter_PairwisePotentialType_BILATERAL_GAUSSIAN)
    {
        function_bilateral_gaussian_layer_->Reshape(function_layer_bottom_vec_, function_layer_top_vec_);
    }
    else{
        function_freeform_layer_->Reshape(function_layer_bottom_vec_, function_layer_top_vec_);
    }
    
    rearrange_layer_top_vec_[0] = top[0];
//    rearrange_layer_->Reshape(rearrange_layer_bottom_vec_, rearrange_layer_top_vec_);
    LayerParameter reshape_param;
    reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(num_);
    reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(kernel_size_*kernel_size_-1);
    reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(height_);
    reshape_param.mutable_reshape_param()->mutable_shape()->add_dim(width_);
    rearrange_layer_.reset(new ReshapeLayer<Dtype>(reshape_param)); // construct function without parameter
    rearrange_layer_->SetUp(rearrange_layer_bottom_vec_, rearrange_layer_top_vec_);
    
}
    
template <typename Dtype>
void PairwisePotentialLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
//    std::cout<<"pair wise potential layer start"<<std::endl;
    feature_layer_->Forward(feature_layer_bottom_vec_, feature_layer_top_vec_);
    if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
       MultiStageCRFParameter_PairwisePotentialType_INTENSITY_GAUSSIAN)
    {
        function_intensity_gaussian_layer_->Forward(function_layer_bottom_vec_, function_layer_top_vec_);
    }
    else if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
       MultiStageCRFParameter_PairwisePotentialType_BILATERAL_GAUSSIAN){
        function_bilateral_gaussian_layer_->Forward(function_layer_bottom_vec_, function_layer_top_vec_);
    }
    else{
        function_freeform_layer_->Forward(function_layer_bottom_vec_, function_layer_top_vec_);
    }
//    std::cout<<"pair wise function layer done"<<std::endl;
    rearrange_layer_->Forward(rearrange_layer_bottom_vec_, rearrange_layer_top_vec_);
//    std::cout<<"pair wise rearrange layer done"<<std::endl;
}
    
template <typename Dtype>
void PairwisePotentialLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}
    
template <typename Dtype>
void PairwisePotentialLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom)
{
    
    vector<bool> rerrange_prop_down(1, true);
    rearrange_layer_->Backward(rearrange_layer_top_vec_, rerrange_prop_down, rearrange_layer_bottom_vec_);
    
    vector<bool> function_prop_down(1, true);
    if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
       MultiStageCRFParameter_PairwisePotentialType_INTENSITY_GAUSSIAN)
    {
        function_intensity_gaussian_layer_->Backward(function_layer_top_vec_, function_prop_down, function_layer_bottom_vec_);
    }
    else if(this->layer_param_.multi_stage_crf_param().pair_wise_potential_type() ==
       MultiStageCRFParameter_PairwisePotentialType_BILATERAL_GAUSSIAN){
        function_bilateral_gaussian_layer_->Backward(function_layer_top_vec_, function_prop_down, function_layer_bottom_vec_);
    }
    else{
        function_freeform_layer_->Backward(function_layer_top_vec_, function_prop_down, function_layer_bottom_vec_);
    }
    
    vector<bool> feature_prop_down(1, true);
    feature_layer_->Backward(feature_layer_top_vec_, feature_prop_down, feature_layer_bottom_vec_);
}

template <typename Dtype>
void PairwisePotentialLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down,
                                                 const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}
INSTANTIATE_CLASS(PairwisePotentialLayer);
}  // namespace caffe
