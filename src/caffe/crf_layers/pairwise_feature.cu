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
#include "pixel_access.cu"

namespace caffe {

template <typename Dtype>
__global__ void feature_forward_kernel(int nthreads, const Dtype* bottom, Dtype* top,
                            int N, int C, int H, int W, int featureN, int neighN) {
    CUDA_KERNEL_LOOP(index, nthreads){
        const int w = index % W;
        const int h = (index / W) % H;
        const int n   = index / W / H;
        // bottom size C, H, W
        // top size C+1, neighN, H*W

        int outC = featureN+1;
        int kernel_size = sqrt(double(neighN+1));
        int kr = (kernel_size -1)/2;
        int h_index = 0;
        int w_index = h*W + w;

        for(int i=-kr; i<=kr; i++)
        {
            for(int j=-kr; j<=kr; j++)
            {
                if(i==0 && j==0)continue;
                Dtype value_diff, p_value, q_value;
                for(int c=0; c<outC; c++)
                {
                    if(c<outC-1)
                    {
                        p_value = get_gpu_pixel(bottom, N, C, H, W, n, c, h, w);
                        q_value = get_gpu_pixel(bottom, N, C, H, W, n, c, i+h, j+w);
                        value_diff=(p_value-q_value);
                    }
                    else
                    {
                        value_diff= sqrt(double(i*i + j*j));
                    }
                    set_gpu_pixel(top, N, outC, neighN, H*W, n, c, h_index, w_index, value_diff);
                }
                h_index++;
            }
        }
    }
}


//template <typename Dtype>
//__global__ void feature_backward_kernel(const Dtype* top_diff,
//                                        const Dtype* bottom_data,
//                                        Dtype * bottom_diff,
//                                        int C, int H, int W, int neighN) {
//    // bottom size C, H, W
//    // top size C+1, neighN, H*W
//    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
//    uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
//    if( i>= H || j>=W) return;
//    int kernel_size = sqrt(double(neighN+1));
//    int kr = (kernel_size -1)/2;
//    
//    for(int c=0; c<C; c++)
//    {
//        int p_index = i*W+j;
//        int q_index = 0;
//        Dtype p_diff = 0;
//        for(int m=-kr; m<=kr; m++)
//        {
//            for(int n=-kr; n<=kr; n++)
//            {
//                if(m==0 && n==0) continue;
//                Dtype t_diff_p, t_diff_q, p_value, q_value;
//            
//                p_value = get_gpu_pixel(bottom_data, C, H, W, c, i, j);
//                q_value = get_gpu_pixel(bottom_data, C, H, W, c, i+m, j+n);
//                t_diff_p = get_gpu_pixel(top_diff, C+1, neighN, H*W, c, q_index, p_index);
//            
//                if(i+m>=0 && i+m<H && j+n>=0 && j+n<W)
//                {
//                    int np_index = (i+m)*W + (j+n);
//                    int nq_index = neighN -1 -q_index;
//                    t_diff_q = get_gpu_pixel(top_diff, C+1, neighN, H*W, c, nq_index, np_index);
//                }
//                else
//                {
//                    t_diff_q = 0;
//                }
//                p_diff += (t_diff_p + t_diff_q)* 2 *(p_value - q_value);
//                q_index++;
//            }
//        }
//        set_gpu_pixel(bottom_diff, C, H, W, c, i, j, p_diff);
//    }
//}

template <typename Dtype>
void feature_backward(const Dtype* top_diff, const Dtype* bottom_data, Dtype * bottom_diff,
                     int N, int C, int H, int W, int neighN) {
//    const Dtype * 
//    dim3 threadsPerBlock(16, 16);
//    dim3 numBlocks((H+threadsPerBlock.x-1)/threadsPerBlock.x,
//                   (W+threadsPerBlock.y-1)/threadsPerBlock.y);
//    long bottom_offset = C*H*W;
//    long top_offset = (C+1)*neighN*H*W;;
//    for(int i=0;i<N; i++)
//    {
//        feature_backward_kernel<Dtype><<<numBlocks, threadsPerBlock>>>(
//            top_diff + i*top_offset,
//            bottom_data + i*bottom_offset,
//            bottom_diff + i*bottom_offset,
//            C, H, W, neighN);
//    }
}
template <typename Dtype>
void PairwiseFeatureLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top)
{
    const Dtype * input_data  = bottom[0]->gpu_data();
    Dtype * output_data=top[0]->mutable_gpu_data();
    CHECK(top[0]->channels() == featureN_ + 1)<<
    ("pairwise feature channel does not match output shape");
    
    CHECK(bottom[0]->height() * bottom[0]->width() == top[0]->width())<<
    ("input image and kernel shoud have matched pixel number")<< bottom[0]->height() <<" "<<bottom[0]->width()<<" "<<top[0]->width();
    
    int count = bottom[0]->num() * bottom[0]->height() * bottom[0]->width();
    feature_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, input_data, output_data,bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(), featureN_, top[0]->height());
//    Forward_cpu(bottom, top);
}
template <typename Dtype>
void PairwiseFeatureLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom)
{
//    const Dtype * top_diff = top[0]->gpu_diff();
//    const Dtype * bottom_data = bottom[0]->gpu_data();
//    Dtype * bottom_diff =  bottom[0]->mutable_gpu_diff();
//    feature_backward(top_diff, bottom_data, bottom_diff,
//                     bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
//                     top[0]->height());
}

INSTANTIATE_LAYER_GPU_FUNCS(PairwiseFeatureLayer);
}  // namespace caffe
