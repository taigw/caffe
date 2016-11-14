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
#include "caffe/crf_layers/unary_composite_layer.hpp"
#include "caffe/crf_layers/pixel_access.hpp"

namespace caffe {
    
template <typename Dtype>
void UnaryCompositeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    num_ = bottom[0]->num();
    unary_channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    
    image_channels_ = bottom[1]->channels();
    
    CHECK(bottom[0]->num()==bottom[1]->num() &&
          bottom[0]->height()==bottom[1]->height() &&
          bottom[0]->width()==bottom[1]->width())
    << "Unary potial size and input image size do not match!";
    LOG(INFO)<<"unary potential and input image channel "<<unary_channels_<<" "<<image_channels_;
}


template <typename Dtype>
void UnaryCompositeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top)
{
    num_ = bottom[0]->num();
    unary_channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(num_, unary_channels_, height_, width_);
    top[1]->Reshape(num_, 1, height_, width_);
}
template <typename Dtype>
void UnaryCompositeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    top[0]->CopyFrom(*bottom[0], false);// data is copied
    
    Dtype user_interaction_potential = this->layer_param_.multi_stage_crf_param().user_interaction_potential();
    Dtype dis_mean = this->layer_param_.multi_stage_crf_param().interaction_dis_mean();
    Dtype dis_std  = this->layer_param_.multi_stage_crf_param().interaction_dis_std();
    Dtype dis_cv = dis_mean/dis_std;
    
    const Dtype * bottom_data = bottom[0]->cpu_data();
    const Dtype * image_data = bottom[1]->cpu_data();
    Dtype * top_data = top[0]->mutable_cpu_data();
    Dtype * mask_data = top[1]->mutable_cpu_data();
    int scribble_point = 0;
    for(int n=0; n<num_; n++)
    {
        for(int h=0; h<height_; h++)
        {
            for(int w=0; w<width_; w++)
            {
                int scribble_channel = -1;
                for(int c = 0; c<unary_channels_; c++)
                {
                    int ci = image_channels_-unary_channels_ + c;
                    Dtype d = get_pixel(image_data, num_, image_channels_, height_, width_,
                                        n, ci, h, w);
                    if((d + dis_cv) < 1e-5 && (d + dis_cv) > -1e-5)
                    {
                        scribble_channel = c;
                        scribble_point++;
                        break;
                    }
                }
                
                Dtype mask_value = (scribble_channel>-1)? 1.0 : 0.0;
                set_pixel(mask_data, num_, 1, height_, width_, n, 0, h, w, mask_value);
                if(scribble_channel > -1)
                {
                    for (int c = 0; c<unary_channels_; c++)
                    {
                        Dtype u_value = 0;//get_pixel(bottom_data, num_, unary_channels_, height_, width_,n, c, h, w);
                        u_value = (c == scribble_channel)? u_value + user_interaction_potential :
                                                           u_value - user_interaction_potential;
                        set_pixel(top_data, num_, unary_channels_, height_, width_, n, c, h, w, u_value);
                    }
                }
            } //for w
        } // for h
    } // for n
    LOG(INFO)<<"scribble point: "<<scribble_point;
}
    
template <typename Dtype>
void UnaryCompositeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom)
{
    if(propagate_down[0]==false)return;
    bottom[0]->CopyFrom(*top[0], true);
}
    
INSTANTIATE_CLASS(UnaryCompositeLayer);
}  // namespace caffe
