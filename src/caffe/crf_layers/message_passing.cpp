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
#include "caffe/crf_layers/message_passing_layer.hpp"
#include "caffe/crf_layers/pixel_access.hpp"

namespace caffe {
template <typename Dtype>
void MessagePassingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top)
{
    kernel_size_ = this->layer_param_.multi_stage_crf_param().kernel_size();
    user_interaction_constrain_ = this->layer_param_.multi_stage_crf_param().user_interaction_constrain();
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    num_pixels_ = height_ * width_;
    
    neighN_= bottom[1]->channels();
    CHECK_EQ(neighN_ , kernel_size_*kernel_size_-1)
    << "MessagePassingLayer should have consistant filter kernel size !";
    CHECK( bottom[0]->num()==bottom[1]->num() &&
           bottom[0]->height()==bottom[1]->height() &&
           bottom[0]->width()==bottom[1]->width())
    << "MessagePassingLayer should have consistant image size !";
    if(bottom[2]==NULL)
    {
        LOG(INFO)<<"interaction mask is NULL";
        CHECK(user_interaction_constrain_==false)<<"interaction mask is NULL";
    }
    else{
        CHECK(bottom[0]->num()==bottom[2]->num() &&
              bottom[0]->height()==bottom[2]->height() &&
              bottom[0]->width()==bottom[2]->width() &&
              bottom[2]->channels()==1)
        << "The unary potential and interaction mask should have consistant image size !";
    }
}
    
template <typename Dtype>
void MessagePassingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top)
{
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    num_pixels_ = height_ * width_;
    top[0]->Reshape(num_, channels_, height_, width_);
}
template <typename Dtype>
void MessagePassingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
{
    const Dtype * input_data  = bottom[0]->cpu_data();
    const Dtype * kernel_data = bottom[1]->cpu_data();
    const Dtype * mask_data   = (user_interaction_constrain_)? bottom[2]->cpu_data(): NULL;
    Dtype * output_data=top[0]->mutable_cpu_data();
    int kr=(kernel_size_-1)/2;
    for(int n=0; n<num_; n++)
    {
        for(int h=0; h< height_; h++)
        {
            for(int w=0; w<width_; w++)
            {
                bool interaction_exist=false;
                if(user_interaction_constrain_){
                    interaction_exist = get_pixel(mask_data, num_, 1, height_, width_, n, 0, h, w)>0;
                }
                for(int c=0; c<channels_; c++)
                {
                    Dtype sum_value=0.0;
                    int neighIdx=0;
                    if(!(user_interaction_constrain_ && interaction_exist)){
                        for(int i=-kr; i<=kr; i++)
                        {
                            for(int j=-kr; j<=kr; j++)
                            {
                                if(i==0 && j==0) continue;
                                Dtype value = get_pixel(input_data, num_, channels_, height_, width_,
                                                        n, c, h+i, w+j);
                                Dtype weight = get_pixel(kernel_data, num_, neighN_, height_, width_,
                                                         n, neighIdx, h, w);
                                sum_value  += value*weight;
                                neighIdx++;
                            }
                        }
                    }
                    set_pixel(output_data, num_, channels_, height_, width_,
                              n, c, h, w, sum_value);
                }
            }
        }
    }
}

template <typename Dtype>
void MessagePassingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
    const Dtype * top_diff = top[0]->cpu_diff();
    const Dtype * bottom_data = bottom[0]->cpu_data();
    Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype * kernel_data = bottom[1]->cpu_data();
    Dtype * kernel_diff = bottom[1]->mutable_cpu_diff();
    const Dtype * mask_data   = (user_interaction_constrain_)? bottom[2]->cpu_data(): NULL;
    int kr=(kernel_size_-1)/2;
    for(int n=0; n<num_; n++)
    {
        for(int c=0; c<channels_; c++)
        {
            for(int h=0; h< height_; h++)
            {
                for(int w=0; w<width_; w++)
                {
                    int q_index=0;
                    Dtype value_diff = 0.0;
                    for(int i=-kr; i<=kr; i++)
                    {
                        for(int j=-kr; j<=kr; j++)
                        {
                            if(i==0 && j==0) continue;
                            int nq_index = neighN_ -1 -q_index;
                            Dtype weight_nq = get_pixel(kernel_data, num_, neighN_, height_, width_,
                                                     n, nq_index, h+i, w+j);
                            if(user_interaction_constrain_ &&
                               get_pixel(mask_data, num_, 1, height_, width_, n, 0, h+i, w+j)){
                                weight_nq = 0;
                            }
                            Dtype t_diff_nq = get_pixel(top_diff, num_, channels_, height_, width_,
                                                        n, c, h+i, w+j);
                            value_diff += weight_nq * t_diff_nq;
                            q_index++;
                        }
                    }
                    set_pixel(bottom_diff, num_, channels_, height_, width_,
                              n, c, h, w, value_diff);
                }
            }
        }
    }
    
    for(int n=0; n<num_; n++)
    {
        for(int h=0; h< height_; h++)
        {
            for(int w=0; w<width_; w++)
            {
                if(user_interaction_constrain_ &&
                   get_pixel(mask_data, num_, 1, height_, width_, n, 0, h, w)){
                    continue;
                }
                int q_index=0;
                for(int i=-kr; i<=kr; i++)
                {
                    for(int j=-kr; j<=kr; j++)
                    {
                        if(i==0 && j==0) continue;
                        Dtype k_diff = 0.0;
                        for(int c=0; c< channels_; c++)
                        {
                            Dtype value = get_pixel(bottom_data, num_, channels_, height_, width_, n, c, h+i, w+j);
                            Dtype t_diff = get_pixel(top_diff, num_, channels_, height_, width_,
                                                        n, c, h, w);
                            k_diff += t_diff*value;
                        }
                        set_pixel(kernel_diff, num_, neighN_, height_, width_, n, q_index, h, w, k_diff);
                        q_index++;
                    }
                }
            }
        }
    }
}

INSTANTIATE_CLASS(MessagePassingLayer);
}  // namespace caffe
