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
#include "caffe/crf_layers/pairwise_feature_layer.hpp"
#include "caffe/crf_layers/pixel_access.hpp"
#include <iostream>
namespace caffe {
template <typename Dtype>
void PairwiseFeatureLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top)
{
    kernel_size_ = this->layer_param().multi_stage_crf_param().kernel_size();
    featureN_    = this->layer_param().multi_stage_crf_param().feature_length();
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    num_pixels_ = height_ * width_;
    CHECK((channels_==featureN_) || (channels_==featureN_+3))<<
    ("input data channel should be 3 (rgb) or 6 (rgb + initial seg + scribble distance)");
    
    Reshape(bottom,top);

}

template <typename Dtype>
void PairwiseFeatureLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top)
{
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    num_pixels_ = height_ * width_;
    
    neighN_=kernel_size_*kernel_size_-1;
    output_shape_.resize(4);
    output_shape_[0] = num_;
    // an additional channel storing the spatial distance of two pixels
    output_shape_[1] = featureN_ + 1;
    output_shape_[2] = neighN_;
    output_shape_[3] = num_pixels_;
    top[0]->Reshape(output_shape_);
    
//    std::cout<<"bottom height and width "<<bottom[0]->height()<<" "<<bottom[0]->width()<<" "<<output_shape_[3]<<std::endl;
}
    
template <typename Dtype>
void PairwiseFeatureLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top)
{
    int kr = (kernel_size_ - 1 )/2;
    //float max_dis = sqrt(height_*height_ + width_*width_);
    for(int n=0; n<num_; n++)
    {
        for(int h=0; h< height_; h++)
        {
            for(int w=0; w<width_; w++)
            {
                int p_index = h*width_ + w;
                int q_index = 0;
                for(int i=-kr; i<=kr; i++)
                {
                    for(int j=-kr; j<=kr; j++)
                    {
                        if(i==0 && j==0)continue;
                        Dtype value_diff, p_value, q_value;
                        for(int c=0; c<output_shape_[1]; c++)
                        {
                            if(c< output_shape_[1]-1)
                            {
                                p_value = get_pixel(bottom[0]->cpu_data(), num_, channels_, height_, width_,
                                                  n, c, h, w);
                                q_value = get_pixel(bottom[0]->cpu_data(), num_, channels_, height_, width_,
                                                  n, c, h+i, w+j);
                                value_diff = (p_value-q_value); //assume p_value and q_value are in the range of (-1,1)
                            }
                            else
                            {
                                value_diff= sqrt(i*i + j*j);
                            }
                            set_pixel(top[0]->mutable_cpu_data(), output_shape_[0], output_shape_[1],
                                      output_shape_[2], output_shape_[3],
                                      n, c, q_index, p_index, value_diff);
                        }
                        q_index++;
                    } // for j
                } // for j
            }
        }
    }
}
    
template <typename Dtype>
void PairwiseFeatureLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom)
{
// where use image itensity to compute pairwise feature, back propagation is omitted.
}

INSTANTIATE_CLASS(PairwiseFeatureLayer);
}  // namespace caffe
