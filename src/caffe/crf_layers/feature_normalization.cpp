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
#include "caffe/crf_layers/feature_normalization_layer.hpp"
#include "caffe/crf_layers/pixel_access.hpp"

namespace caffe {
template <typename Dtype>
void FeatureNormalizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top)
{
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
}

template <typename Dtype>
void FeatureNormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top)
{
    top[0]->Reshape(num_, channels_, height_, width_);
}
    
template <typename Dtype>
void FeatureNormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top)
{
    for(int n=0; n<num_; n++)
    {
        for(int h=0; h< height_; h++)
        {
            for(int w=0; w<width_; w++)
            {
                Dtype sum=0;
                for(int c=0; c<channels_; c++)
                {
                    sum +=get_pixel(bottom[0]->cpu_data(), num_, channels_, height_, width_, n, c, h, w);
                }
                if(sum!=0)
                {
                    for(int c=0; c<channels_; c++)
                    {
                        Dtype temp_value =get_pixel(bottom[0]->cpu_data(), num_, channels_, height_, width_, n, c, h, w)/sum;
                        set_pixel(top[0]->mutable_cpu_data(), num_, channels_, height_, width_, n, c, h, w, temp_value);
                    }
                }
            }
        }
    }
}
    
template <typename Dtype>
void FeatureNormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom)
{
    if(propagate_down[0]==false)return;
    
    for(int n=0; n<num_; n++)
    {
        for(int h=0; h< height_; h++)
        {
            for(int w=0; w<width_; w++)
            {
                Dtype sum=0;
                for(int c=0; c<channels_; c++)
                {
                    sum +=get_pixel(bottom[0]->cpu_data(), num_, channels_, height_, width_, n, c, h, w);
                }
                if(sum!=0)
                {
                    for(int c=0; c<channels_; c++)
                    {
                        Dtype top_diff = get_pixel(top[0]->cpu_diff(), num_, channels_, height_, width_, n, c, h, w);
                        Dtype bottom_diff =top_diff/sum;
                        set_pixel(bottom[0]->mutable_cpu_diff(), num_, channels_, height_, width_, n, c, h, w, bottom_diff);
                    }
                }
            }
        }
    }
}

INSTANTIATE_CLASS(FeatureNormalizationLayer);
}  // namespace caffe
