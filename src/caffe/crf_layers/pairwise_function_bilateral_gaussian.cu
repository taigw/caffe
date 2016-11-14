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
#include "caffe/crf_layers/pairwise_function_bilateral_gaussian_layer.hpp"
#include "pixel_access.cu"
namespace caffe {

template <typename Dtype>
__global__ void gaussian_function_kernel(const int nthreads, const Dtype* bottom_data, Dtype * top_data,
                                          int N, int C, int H, int W,
                                         float w1, float w2, float theta_alpha, float theta_beta, float theta_gamma)
{
    CUDA_KERNEL_LOOP(index, nthreads){
        const int w = index % W;
        const int h = (index/W) % H;
        const int n = index / W / H ;
        
        Dtype isq=0;
        Dtype dsq;
        for(int c=0; c<C; c++)
        {
            Dtype p_value = get_gpu_pixel(bottom_data, N, C, H, W, n, c, h, w);
            if(c<C-1){
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
        set_gpu_pixel(top_data, N, 1, H, W, n, 0, h, w, pair_potential);
    }
}

template <typename Dtype>
void PairwiseFunctionBilateralGaussianLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top)
{
    //Forward_cpu(bottom, top);
    const Dtype * input_data  = bottom[0]->gpu_data();
//    const Dtype * param_data = param_blob_->gpu_data();
    Dtype * output_data=top[0]->mutable_gpu_data();

//    const Dtype * param_datacpu = param_blob_->cpu_data();
//    LOG(INFO)<<"params: "<<param_datacpu[0]<<" "<<param_datacpu[1]<<" "<<param_datacpu[2]<<" "<<param_datacpu[3]<<" "<<param_datacpu[4];
//
//    const Dtype * param_diff = param_blob_->cpu_diff();
//    LOG(INFO)<<"params diff: "<<param_diff[0]<<" "<<param_diff[1]<<" "<<param_diff[2]<<" "<<param_diff[3]<<" "<<param_diff[4];
    

    CHECK(bottom[0]->height() == top[0]->height() && bottom[0]->width() == top[0]->width() &&
          bottom[0]->num() == top[0]->num() )<<
    ("input size and output size does not match");
    CHECK(top[0]->channels() ==1 )<<
    ("number of output channel should be 1");
    
    float w1 = param_blobs_[0]->cpu_data()[0];
    float w2 = param_blobs_[1]->cpu_data()[0];
    float theta_alpha = param_blobs_[2]->cpu_data()[0];
    float theta_beta  = param_blobs_[3]->cpu_data()[0];
    float theta_gamma = param_blobs_[4]->cpu_data()[0];
    
    int count = top[0]->count();
    gaussian_function_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        (count, input_data, output_data,
         bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
         w1, w2, theta_alpha, theta_beta, theta_gamma);

//    LOG(INFO)<<"w1, w2, alpha, beta, gamma= "<<param_blob_->cpu_data()[0]<<" "<<param_blob_->cpu_data()[1]<<" "<<
//        param_blob_->cpu_data()[2]<<" "<<param_blob_->cpu_data()[3]<<" "<<param_blob_->cpu_data()[4];
}
    
template <typename Dtype>
void PairwiseFunctionBilateralGaussianLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                        const vector<bool>& propagate_down,
                                                        const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}
    
INSTANTIATE_LAYER_GPU_FUNCS(PairwiseFunctionBilateralGaussianLayer);
}  // namespace caffe
