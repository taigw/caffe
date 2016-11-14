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
#include "pixel_access.cu"
namespace caffe {
    
template <typename Dtype>
__global__ void conv_kernel(const int nthreads, const Dtype* bottom, const Dtype* kernel, const Dtype* mask_data,Dtype* top, int N, int C, int H, int W, int neighN, bool user_interaction_constrain) {
    CUDA_KERNEL_LOOP(index, nthreads){
        const int w = index % W;
        const int h = (index/W) % H;
        const int c = (index / W / H) % C;
        const int n = index / W / H / C;
        
        int kernel_size = sqrt(double(neighN+1));
        int kr = (kernel_size -1)/2;
       
        int neighIdx=0;
        Dtype sum_value=0.0;
        bool interaction_exist =false;
        if(user_interaction_constrain){
         interaction_exist = get_gpu_pixel(mask_data, N, 1, H, W, n, 0, h, w)>0.0;
        }
        if(!(user_interaction_constrain && interaction_exist)){
            for(int i = -kr; i <= kr; i++)
            {
                for(int j = -kr; j <= kr; j++)
                {
                    if(i==0 && j==0) continue;
                    Dtype value = get_gpu_pixel(bottom, N, C, H, W, n, c, i+h, j+w);
                    Dtype weidht= get_gpu_pixel(kernel, N, neighN, H, W, n, neighIdx, h, w);
                    sum_value += value*weidht;
                    neighIdx++;
                }
            }
        }
        set_gpu_pixel(top, N, C, H, W, n, c, h, w, sum_value);
    }
}


template <typename Dtype>
__global__ void conv_gradient_to_input_kernel(const int nthreads, const Dtype* top_diff, const Dtype* kernel_data, const Dtype * mask_data, Dtype* bottom_diff, int N, int C, int H, int W, int neighN, bool user_interaction_constrain) {
    CUDA_KERNEL_LOOP(index, nthreads){
        const int w = index % W;
        const int h = (index/W) % H;
        const int c = (index / W / H) % C;
        const int n = index / W / H / C;
        
        int kernel_size = sqrt(double(neighN+1));
        int kr = (kernel_size -1)/2;

        int q_index=0;
        Dtype value_diff = 0.0;
        for(int i = -kr; i <= kr; i++)
        {
            for(int j = -kr; j <= kr; j++)
            {
                if(i==0 && j==0) continue;
                int nq_index = neighN-1 -q_index;
                Dtype weight_nq = get_gpu_pixel(kernel_data, N, neighN, H, W, n, nq_index, i+h, j+w);
                if(user_interaction_constrain &&
                   get_gpu_pixel(mask_data, N, 1, H, W, n, 0, i+h, j+w)){
                    weight_nq = 0;
                }
                Dtype t_diff_nq = get_gpu_pixel(top_diff, N, C, H, W, n, c, i+h, j+w);
                value_diff += weight_nq* t_diff_nq;
                q_index++;
            }
        }
        set_gpu_pixel(bottom_diff, N, C, H, W, n, c, h, w, value_diff);
    }
}
    
template <typename Dtype>
__global__ void conv_gradient_to_weight_kernel(const int nthreads, const Dtype* top_diff, const Dtype* bottom_data, const Dtype * mask_data, Dtype* kernel_diff, int N, int C, int H, int W, int neighN, bool user_interaction_constrain) {
    CUDA_KERNEL_LOOP(index, nthreads){
        const int w = index % W;
        const int h = (index/W) % H;
        const int c = (index / W / H) % neighN;
        const int n = index / W / H / neighN;
        
        if(user_interaction_constrain &&
           get_gpu_pixel(mask_data, N, 1, H, W, n, 0, h, w)){
            return;
        }
        int kernel_size = sqrt(double(neighN+1));
        int kr = (kernel_size -1)/2;
        
        int cN = (c >= kr*kernel_size + kr)? c+1 : c;
       
        int j = cN % kernel_size - kr;
        int i = cN / kernel_size - kr;
        Dtype k_diff = 0;
        for(int cIdx = 0; cIdx<C; cIdx++)
        {
            Dtype t_diff = get_gpu_pixel(top_diff, N, C, H, W, n, cIdx, h, w);
            Dtype value = get_gpu_pixel(bottom_data, N, C, H, W, n, cIdx, h+i, w+j);
            k_diff += value * t_diff;
        }
        //k_diff = k_diff / C;
        set_gpu_pixel(kernel_diff, N, neighN, H, W, n, c, h, w, k_diff);
    }
}

template <typename Dtype>
void MessagePassingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
{
    const Dtype * input_data  = bottom[0]->gpu_data();
    const Dtype * kernel_data = bottom[1]->gpu_data();
    const Dtype * mask_data   = (user_interaction_constrain_)? bottom[2]->gpu_data(): NULL;
    
    Dtype * output_data=top[0]->mutable_gpu_data();
    CHECK_EQ(bottom[0]->count(), top[0]->count())<<
        ("input image and output image shoud have the same size");
    CHECK(bottom[0]->height() == bottom[1]->height() && bottom[0]->width() == bottom[1]->width())<<
        ("input image and kernel shoud have the pixel number");

    int count = top[0]->count();
    conv_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        (count, input_data, kernel_data, mask_data, output_data,
        bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
        bottom[1]->channels(), user_interaction_constrain_);
}

template <typename Dtype>
void MessagePassingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
    //LOG(INFO) << ("message pasing backward_gpu start.");
    const Dtype * top_diff = top[0]->gpu_diff();
    const Dtype * bottom_data = bottom[0]->gpu_data();
    Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype * kernel_data = bottom[1]->gpu_data();
    Dtype * kernel_diff = bottom[1]->mutable_gpu_diff();
    const Dtype * mask_data = (user_interaction_constrain_)? bottom[2]->gpu_data(): NULL;
    
    int bottom_count = bottom[0]->count();
    conv_gradient_to_input_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>
        (bottom_count, top_diff, kernel_data, mask_data, bottom_diff,
         bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
         bottom[1]->channels(), user_interaction_constrain_);
   
    int kernel_count = bottom[1]->count();
    conv_gradient_to_weight_kernel<Dtype><<<CAFFE_GET_BLOCKS(kernel_count), CAFFE_CUDA_NUM_THREADS>>>
        (kernel_count, top_diff, bottom_data, mask_data, kernel_diff,
        bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
        bottom[1]->channels(), user_interaction_constrain_);
}

INSTANTIATE_LAYER_GPU_FUNCS(MessagePassingLayer);
}  // namespace caffe
