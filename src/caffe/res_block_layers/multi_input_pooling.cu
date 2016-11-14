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


namespace caffe {

template <typename Dtype>
__global__ void pooling_forward_kernel(int nthreads, Dtype** bottom, Dtype* top, int bottomN) {
    CUDA_KERNEL_LOOP(index, nthreads){
        
        Dtype max_value = 1e-8;
        for(int i=0; i<bottomN; i++)
        {
            Dtype temp_value = bottom[i][index];
            if(temp_value > max_value) max_value = temp_value;
        }
        top[index] = max_value;
    }
}


template <typename Dtype>
__global__ void pooling_backward_kernel(int nthreads,const Dtype* top_diff,
                                        Dtype ** bottom_data,
                                        Dtype ** bottom_diff,int bottomN) {
    CUDA_KERNEL_LOOP(index, nthreads){
        
        Dtype max_value = 1e-8;
        int max_Index = 0;
        for(int i=0; i<bottomN; i++)
        {
            Dtype temp_value = bottom_data[i][index];
            if(temp_value > max_value) {
                max_value = temp_value;
                max_Index = i;
            }
        }
        bottom_diff[max_Index][index] = top_diff[index];
    }
}

template <typename Dtype>
void MultiInputPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top)
{
    Dtype ** bottom_data = (Dtype **)malloc(sizeof(Dtype*)*bottom.size());
    for(int i=0;i<bottom.size(); i++)
    {
        bottom_data[i] = (Dtype *) bottom[i]->gpu_data();
    }

    Dtype * top_data=top[0]->mutable_gpu_data();
    
    int count = bottom[0]->count();
    pooling_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, bottom_data, top_data,  bottom.size());
    
    free(bottom_data);
}
template <typename Dtype>
void MultiInputPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom)
{
    Dtype ** bottom_data = (Dtype **)malloc(sizeof(Dtype*)*bottom.size());
    for(int i=0;i<bottom.size(); i++)
    {
        bottom_data[i] = (Dtype *) bottom[i]->gpu_data();
    }
    
    Dtype ** bottom_diff = (Dtype **)malloc(sizeof(Dtype*)*bottom.size());
    for(int i=0;i<bottom.size(); i++)
    {
        bottom_diff[i] = bottom[i]->mutable_gpu_diff();
    }
    
    Dtype * top_diff=(Dtype *) top[0]->gpu_diff();
    
    int count = bottom[0]->count();
    pooling_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (count, top_diff,  bottom_data, bottom_diff, bottom.size());
    
    free(bottom_data);
    free(bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiInputPoolingLayer);
}  // namespace caffe
