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
#include "caffe/crf_layers/pairwise_function_bilateral_gaussian_layer.hpp"
#include "caffe/crf_layers/pixel_access.hpp"


namespace caffe {
    
#define MAX(A,B) A>B?A:B
#define MIN(A,B) A>B?B:A
template <typename Dtype>
void PairwiseFunctionBilateralGaussianLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    
    param_blobs_.resize(5);
    param_blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
    param_blobs_[1].reset(new Blob<Dtype>(1, 1, 1, 1));
    param_blobs_[2].reset(new Blob<Dtype>(1, 1, 1, 1));
    param_blobs_[3].reset(new Blob<Dtype>(1, 1, 1, 1));
    param_blobs_[4].reset(new Blob<Dtype>(1, 1, 1, 1));
    
    param_blobs_[0]->mutable_cpu_data()[0] = this->layer_param_.multi_stage_crf_param().w1(); //w1
    param_blobs_[1]->mutable_cpu_data()[0] = this->layer_param_.multi_stage_crf_param().w2(); //w2
    param_blobs_[2]->mutable_cpu_data()[0] = this->layer_param_.multi_stage_crf_param().theta_alpha(); // theta_alpha
    param_blobs_[3]->mutable_cpu_data()[0] = this->layer_param_.multi_stage_crf_param().theta_beta()*(channels_-1);  //theta_beta
    param_blobs_[4]->mutable_cpu_data()[0] = this->layer_param_.multi_stage_crf_param().theta_gamma(); //theta_gamma
    
    this->blobs_.insert(this->blobs_.end(), param_blobs_.begin(), param_blobs_.end());
}


template <typename Dtype>
void PairwiseFunctionBilateralGaussianLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top)
{
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(num_, 1, height_, width_);
}
template <typename Dtype>
void PairwiseFunctionBilateralGaussianLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    const Dtype * bottom_data = bottom[0]->cpu_data();
    Dtype * top_data = top[0]->mutable_cpu_data();

    float w1 = param_blobs_[0]->cpu_data()[0];
    float w2 = param_blobs_[1]->cpu_data()[0];
    float theta_alpha = param_blobs_[2]->cpu_data()[0];
    float theta_beta  = param_blobs_[3]->cpu_data()[0];
    float theta_gamma = param_blobs_[4]->cpu_data()[0];
    
    float dw1 = param_blobs_[0]->cpu_diff()[0];
    float dw2 = param_blobs_[1]->cpu_diff()[0];
    float dtheta_alpha = param_blobs_[2]->cpu_diff()[0];
    float dtheta_beta  = param_blobs_[3]->cpu_diff()[0];
    float dtheta_gamma = param_blobs_[4]->cpu_diff()[0];
    
    LOG(INFO)<<"params: "<<w1<<" "<<w2<<" "<<theta_alpha<<" "<<theta_beta<<" "<<theta_gamma;
    LOG(INFO)<<"params diff: "<<dw1<<" "<<dw2<<" "<<dtheta_alpha<<" "<<dtheta_beta<<" "<<dtheta_gamma;
    for(int n=0; n<num_; n++)
    {
        for( int h=0; h< height_; h++)
        {
            for(int w=0; w<width_; w++)
            {
                Dtype isq=0;
                Dtype dsq;
                for(int c=0; c<channels_; c++)
                {
                    Dtype p_value = get_pixel(bottom_data, num_, channels_, height_, width_,
                                        n, c, h, w);
                    if(c<channels_-1){
                        isq += p_value*p_value;
                    }
                    else{
                        dsq = p_value*p_value;
                    }
                }
                Dtype p_term = dsq/(2 * theta_alpha * theta_alpha);
                Dtype i_term = isq/(2 * theta_beta * theta_beta);
                Dtype bilateral = exp( - i_term - p_term);
                Dtype spatial = exp(- dsq/(2 * theta_gamma * theta_gamma));
                Dtype pair_potential = w1*bilateral + w2* spatial;
                set_pixel(top_data, num_, 1, height_, width_,
                          n, 0, h, w, pair_potential);
            }
        }
    }
}


template <typename Dtype>
void PairwiseFunctionBilateralGaussianLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom)
{
    bool fix_param = this->layer_param_.multi_stage_crf_param().fix_param();
    if(fix_param){
        return;
    }
    const Dtype * top_diff = top[0]->cpu_diff();
    const Dtype * bottom_data = bottom[0]->cpu_data();
    
    float w1 = param_blobs_[0]->cpu_data()[0];
    float w2 = param_blobs_[1]->cpu_data()[0];
    float theta_alpha = param_blobs_[2]->cpu_data()[0];
    float theta_beta  = param_blobs_[3]->cpu_data()[0];
    float theta_gamma = param_blobs_[4]->cpu_data()[0];
    
    float diff_w1 =0;
    float diff_w2 =0;
    float diff_alpha = 0;
    float diff_beta  = 0;
    float diff_gamma = 0;
    
    for(int n=0; n<num_; n++)
    {
        for( int h=0; h< height_; h++)
        {
            for(int w=0; w<width_; w++)
            {
                Dtype isq=0;
                Dtype dsq;
                for(int c=0; c<channels_; c++)
                {
                    Dtype p_value = get_pixel(bottom_data, num_, channels_, height_, width_,
                                        n, c, h, w);
                    if(c<channels_-1){
                        isq += p_value*p_value;
                    }
                    else{
                        dsq = p_value*p_value;
                    }
                }
                Dtype p_term = dsq/(2 * theta_alpha * theta_alpha);
                Dtype i_term = isq/(2 * theta_beta * theta_beta);
                Dtype bilateral = exp( - i_term - p_term);
                Dtype spatial = exp(- dsq/(2 * theta_gamma * theta_gamma));
                
                Dtype t_diff = get_pixel(top_diff, num_, 1, height_, width_, n, 0, h, w);
                diff_w1 += t_diff * bilateral;
                diff_w2 += t_diff * spatial;
                diff_alpha += t_diff*w1*bilateral*dsq / (theta_alpha*theta_alpha*theta_alpha);
                diff_beta  += t_diff*w1*bilateral*isq / (theta_beta*theta_beta*theta_beta);
                diff_gamma += t_diff*w2*spatial*dsq / (theta_gamma*theta_gamma*theta_gamma);
            }
        }
    }
//    
//    LOG(INFO)<<"diff w1, beta "<<diff_w1<<" "<<diff_beta;
//    LOG(INFO)<<"pair potential min, max "<<potential_min<<" "<<potential_max;
//    LOG(INFO)<<"dsp min, max "<<dsq_min<<" "<<dsq_max;
//    LOG(INFO)<<"isp min, max "<<isq_min<<" "<<isq_max;
    param_blobs_[0]->mutable_cpu_diff()[0] = diff_w1;
    param_blobs_[1]->mutable_cpu_diff()[0] = diff_w2;
    param_blobs_[2]->mutable_cpu_diff()[0] = diff_alpha;
    param_blobs_[3]->mutable_cpu_diff()[0] = diff_beta;
    param_blobs_[4]->mutable_cpu_diff()[0] = diff_gamma;
}
INSTANTIATE_CLASS(PairwiseFunctionBilateralGaussianLayer);
}  // namespace caffe
