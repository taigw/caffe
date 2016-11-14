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
#include "caffe/res_block_layers/res_block_layer.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

namespace caffe {
    
template <typename Dtype>
void ResidualBlockLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top)
{
    ConvolutionParameter conv_param = this->layer_param_.convolution_param();
    output_channels_ = conv_param.num_output();
    
    enable_residual_ = this->layer_param_.res_block_param().enable_residual();
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    
    if(enable_residual_)
    {
        split_out_blobs_.resize(2);
        split_out_blobs_[0].reset(new Blob<Dtype>());
        split_out_blobs_[1].reset(new Blob<Dtype>());
        
        split_top_vec_.clear();
        split_top_vec_.push_back(split_out_blobs_[0].get());
        split_top_vec_.push_back(split_out_blobs_[1].get());
    
        split_bottom_vec_.clear();
        split_bottom_vec_.push_back(bottom[0]);
        
        LayerParameter split_param;
        split_layer_.reset(new SplitLayer<Dtype>(split_param));
        split_layer_->SetUp(split_bottom_vec_,split_top_vec_);
    }

    layerN_ = 4;
    int dropN = layerN_/2;
    conv_out_blobs_.resize(layerN_);
    conv_layers_.resize(layerN_);
    conv_bottom_vec_.resize(layerN_);
    conv_top_vec_.resize(layerN_);
    
    relu_out_blobs_.resize(layerN_);
    relu_layers_.resize(layerN_);
    relu_bottom_vec_.resize(layerN_);
    relu_top_vec_.resize(layerN_);
    
    drop_out_blobs_.resize(dropN);
    drop_layers_.resize(dropN);
    drop_bottom_vec_.resize(dropN);
    drop_top_vec_.resize(dropN);
    
    LOG(INFO) << ("create convlolution layers ");
    for(int i=0; i< layerN_; i++)
    {
        conv_out_blobs_[i].reset(new Blob<Dtype>());
        relu_out_blobs_[i].reset(new Blob<Dtype>());
        
        // convolution layers
        vector<Blob<Dtype> *> temp_conv_bottom_vec_;
        vector<Blob<Dtype> *> temp_conv_top_vec_;
        temp_conv_top_vec_.clear();
        temp_conv_top_vec_.push_back(conv_out_blobs_[i].get());
        temp_conv_bottom_vec_.clear();
        if(i==0){
            temp_conv_bottom_vec_.push_back(enable_residual_ ? split_out_blobs_[1].get() : bottom[0]);
        }
        else{
            if(i%2==1)
            {
                temp_conv_bottom_vec_.push_back(drop_out_blobs_[(i-1)/2].get());
            }
            else{
                temp_conv_bottom_vec_.push_back(relu_out_blobs_[i-1].get());
            }
        }
        
        conv_bottom_vec_[i]=temp_conv_bottom_vec_;
        conv_top_vec_[i]=temp_conv_top_vec_;
        
        if(i%2==1)
        {
            LayerParameter temp_conv_param;
            temp_conv_param.mutable_convolution_param()->set_num_output(output_channels_);
            temp_conv_param.mutable_convolution_param()->set_kernel_h(1);
            temp_conv_param.mutable_convolution_param()->set_kernel_w(1);
            temp_conv_param.mutable_convolution_param()->set_pad_h(0);
            temp_conv_param.mutable_convolution_param()->set_pad_w(0);
            temp_conv_param.mutable_convolution_param()->mutable_weight_filler()->set_type("gaussian");
            temp_conv_param.mutable_convolution_param()->mutable_weight_filler()->set_mean(0.0);
            temp_conv_param.mutable_convolution_param()->mutable_weight_filler()->set_std(sqrt(2.0/output_channels_));
            conv_layers_[i].reset(new ConvolutionLayer<Dtype>(temp_conv_param));
        }
        else{
            conv_layers_[i].reset(new ConvolutionLayer<Dtype>(this->layer_param_));
        }
        conv_layers_[i]->SetUp(conv_bottom_vec_[i], conv_top_vec_[i]);
      
        //  relu layers
        vector<Blob<Dtype> *> temp_relu_bottom_vec_;
        vector<Blob<Dtype> *> temp_relu_top_vec_;
        temp_relu_top_vec_.clear();
        temp_relu_top_vec_.push_back( (i==layerN_-1 && (!enable_residual_)) ? top[0] : relu_out_blobs_[i].get());
        temp_relu_bottom_vec_.clear();
        temp_relu_bottom_vec_.push_back(conv_out_blobs_[i].get());
        
        relu_bottom_vec_[i]=temp_relu_bottom_vec_;
        relu_top_vec_[i] = temp_relu_top_vec_;
        
        LayerParameter relu_param;
        relu_param.mutable_relu_param()->set_negative_slope(0.01);
        relu_layers_[i].reset(new ReLULayer<Dtype>(relu_param));
        relu_layers_[i]->SetUp(relu_bottom_vec_[i], relu_top_vec_[i]);
        
        //drop out
        if(i%2==0)
        {
            drop_out_blobs_[i/2].reset(new Blob<Dtype>());
            drop_top_vec_[i/2].clear();
            drop_top_vec_[i/2].push_back(drop_out_blobs_[i/2].get());
            drop_bottom_vec_[i/2].clear();
            drop_bottom_vec_[i/2].push_back(relu_out_blobs_[i].get());
            LayerParameter drop_param;
            drop_layers_[i/2].reset(new DropoutLayer<Dtype>(drop_param));
            drop_layers_[i/2]->SetUp(drop_bottom_vec_[i/2],drop_top_vec_[i/2]);
        }
    }

    if(enable_residual_)
    {
        sum_top_vec_.clear();
        sum_top_vec_.push_back(top[0]);
        sum_bottom_vec_.clear();
        sum_bottom_vec_.push_back(split_out_blobs_[0].get());
        sum_bottom_vec_.push_back(relu_out_blobs_[layerN_-1].get());
        
        LayerParameter sum_param;
        sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.));
        sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.));
        sum_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
        sum_layer_.reset(new EltwiseLayer<Dtype>(sum_param));
        sum_layer_->SetUp(sum_bottom_vec_,sum_top_vec_);

    }
    
    this->blobs_.clear();
    for(int i=0; i< layerN_; i++)
    {
        this->blobs_.insert(this->blobs_.end(), conv_layers_[i]->blobs().begin(), conv_layers_[i]->blobs().end());
    }
}

template <typename Dtype>
void ResidualBlockLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top)
{
    
    top[0]->Reshape(num_, output_channels_, height_, width_);
}
template <typename Dtype>
void ResidualBlockLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    if(enable_residual_)
    {
        split_layer_->Forward(split_bottom_vec_, split_top_vec_);
    }

    for(int i=0; i < layerN_; i++)
    {
        conv_layers_[i]->Forward(conv_bottom_vec_[i], conv_top_vec_[i]);
        relu_layers_[i]->Forward(relu_bottom_vec_[i], relu_top_vec_[i]);
        if(i%2==0)
        {
            drop_layers_[i/2]->Forward(drop_bottom_vec_[i/2], drop_top_vec_[i/2]);
        }
    }
    
    if(enable_residual_)
    {
        sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
    }
}

template <typename Dtype>
void ResidualBlockLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}
    
template <typename Dtype>
void ResidualBlockLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom)
{
    if(!propagate_down[0])return;
    if(enable_residual_)
    {
        vector<bool> sum_down(2, true);
        sum_layer_->Backward(sum_top_vec_, sum_down, sum_bottom_vec_);
    }
    for(int i=layerN_-1; i >=0; i--)
    {
        if(i%2==0)
        {
            vector<bool> drop_prop_down(1, true);
            drop_layers_[i/2]->Backward(drop_top_vec_[i/2], drop_prop_down, drop_bottom_vec_[i/2]);
        }
        vector<bool> relu_prop_down(1, true);
        relu_layers_[i]->Backward(relu_top_vec_[i], relu_prop_down, relu_bottom_vec_[i]);
        
        vector<bool> conv_prop_down(1, true);
        conv_layers_[i]->Backward(conv_top_vec_[i], conv_prop_down, conv_bottom_vec_[i]);
    }
    if(enable_residual_)
    {
        vector<bool> split_down(1, true);
        split_layer_->Backward(split_top_vec_, split_down, split_bottom_vec_);
    }
}

template <typename Dtype>
void ResidualBlockLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                      const vector<bool>& propagate_down,
                                                      const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}
INSTANTIATE_CLASS(ResidualBlockLayer);
REGISTER_LAYER_CLASS(ResidualBlock);
}  // namespace caffe
