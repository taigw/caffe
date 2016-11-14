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
#include "pixel_access.cu"
namespace caffe {
    
template <typename Dtype>
__global__ void normalization_forward_kernel(const int nthreads, const Dtype* bottom_data, Dtype* top_data,int N, int C, int H, int W) {
    CUDA_KERNEL_LOOP(index, nthreads){
    const int w = index % W;
    const int h = (index/W) % H;
    const int n = (index / W / H);

    Dtype sum = 0;
    for( int c=0; c<C; c++)
    {
        sum +=get_gpu_pixel(bottom_data, N, C, H, W, n, c, h, w);
    }
    if(sum!=0)
    {
        for( int c=0; c<C; c++)
        {
            Dtype temp_value = get_gpu_pixel(bottom_data, N, C, H, W, n, c, h, w)/sum;
            set_gpu_pixel(top_data, N, C, H, W, n, c, h, w, temp_value);
        }
    }
}
}

template <typename Dtype>
__global__ void normalization_backward_kernel(const int nthreads, const Dtype* bottom_data, const Dtype* top_diff, Dtype * bottom_diff, int N, int C, int H, int W) {
    CUDA_KERNEL_LOOP(index, nthreads){
    const int w = index % W;
    const int h = (index/W) % H;
    const int n = (index / W / H);
    
    Dtype sum = 0;
    for( int c=0; c<C; c++)
    {
        sum +=get_gpu_pixel(bottom_data, N, C, H, W, n, c, h, w);
    }
    if(sum!=0)
    {
        for( int c=0; c<C; c++)
        {
            Dtype temp_diff = get_gpu_pixel(top_diff, N, C, H, W, n, c, h, w)/sum;
            set_gpu_pixel(bottom_diff, N, C, H, W, n, c, h, w, temp_diff);
        }
    }
}
}
    


template <typename Dtype>
void FeatureNormalizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
{
    const Dtype * bottom_data  = bottom[0]->gpu_data();
    Dtype * top_data = top[0]->mutable_gpu_data();
    CHECK_EQ(bottom[0]->count(), top[0]->count())<<
        ("input image and output image shoud have the same size");
    
    int count = num_ * height_ * width_;
    normalization_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (count, bottom_data, top_data, num_, channels_, height_, width_);
}

template <typename Dtype>
void FeatureNormalizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
    if (!propagate_down[0]) {
        return;
    }
    //LOG(INFO) << ("message pasing backward_gpu start.");
    const Dtype * top_diff = top[0]->gpu_diff();
    const Dtype * bottom_data = bottom[0]->gpu_data();
    Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
    
    int count = num_ * height_ * width_;
    normalization_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (count, bottom_data, top_diff, bottom_diff, num_, channels_, height_, width_);
}

INSTANTIATE_LAYER_GPU_FUNCS(FeatureNormalizationLayer);
}  // namespace caffe
