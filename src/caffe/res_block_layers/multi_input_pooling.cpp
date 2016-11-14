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
#include "caffe/res_block_layers/multi_input_pooling_layer.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

namespace caffe {
    
template <typename Dtype>
void MultiInputPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top)
{
    inputN_ = bottom.size();
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
}

template <typename Dtype>
void MultiInputPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top)
{
    
    top[0]->Reshape(num_, channels_, height_, width_);
}
template <typename Dtype>
void MultiInputPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    for(int i=0;i<count_; i++)
    {
        Dtype max_value = -1e8;
        for(int bn=0; bn<inputN_; bn++)
        {
            Dtype temp_value = bottom[bn]->cpu_data()[i];
            if(temp_value>max_value) max_value = temp_value;
        }
        top[0]->mutable_cpu_data()[i] = max_value;
    }
}

template <typename Dtype>
void MultiInputPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}
    
template <typename Dtype>
void MultiInputPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom)
{
    if(!propagate_down[0])return;
    for(int i=0;i<count_; i++)
    {
        Dtype max_value = -1e8;
        int max_index =0;
        for(int bn=0; bn<inputN_; bn++)
        {
            Dtype temp_value = bottom[bn]->cpu_data()[i];
            if(temp_value>max_value) {
                max_value = temp_value;
                max_index = bn;
            }
        }
        bottom[max_index]->mutable_cpu_diff()[i] = top[0]->cpu_diff()[i];
    }
}

template <typename Dtype>
void MultiInputPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                      const vector<bool>& propagate_down,
                                                      const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}
INSTANTIATE_CLASS(MultiInputPoolingLayer);
REGISTER_LAYER_CLASS(MultiInputPooling);
}  // namespace caffe
