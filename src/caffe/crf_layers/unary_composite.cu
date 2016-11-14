#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/crf_layers/unary_composite_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "pixel_access.cu"

namespace caffe {
    
    
template <typename Dtype>
__global__ void unary_composite_kernel(const int nthreads, const Dtype* bottom_data, const Dtype * image_data,  Dtype* top_data, Dtype * mask_data, int N, int C, int H, int W, int C_image, Dtype u_potential, Dtype dis_cv)
{
    CUDA_KERNEL_LOOP(index, nthreads){
        const int w = index % W;
        const int h = (index/W) % H;
        const int n = index / W / H;
        
        int scribble_channel = -1;
        for(int c=0; c<C; c++)
        {
            int ci = C_image - C +c;
            Dtype d = get_gpu_pixel(image_data, N, C_image, H, W, n, ci, h, w);
            
            if((d + dis_cv) < 1e-5 && (d + dis_cv) > -1e-5)
            {
                scribble_channel = c;
                break;
            }
        }
        Dtype mask_value = (scribble_channel>-1)? 1.0 : 0.0;
        set_gpu_pixel(mask_data, N, 1, H, W, n, 0, h, w, mask_value);
        if(scribble_channel>-1)
        {
            for(int c=0; c<C; c++)
            {
                Dtype u_value = 0;//get_gpu_pixel(bottom_data, N, C, H, W, n, c, h, w);
                u_value = (c == scribble_channel)? u_value + u_potential : u_value - u_potential;
                set_gpu_pixel(top_data, N, C, H, W, n, c, h, w, u_value);
            }
        }
    }
}
    
template <typename Dtype>
void UnaryCompositeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    top[0]->CopyFrom(*bottom[0], false);// data is copied
    
    const Dtype * bottom_data = bottom[0]->gpu_data();
    const Dtype * image_data = bottom[1]->gpu_data();
    
    Dtype * top_data = top[0]->mutable_gpu_data();
    Dtype * mask_data = top[1]->mutable_gpu_data();
    int count = num_ * height_ * width_;
    Dtype user_potential = this->layer_param_.multi_stage_crf_param().user_interaction_potential();
//    LOG(INFO) << "user_interaction_potential "<< user_potential;
    Dtype dis_mean = this->layer_param_.multi_stage_crf_param().interaction_dis_mean();
    Dtype dis_std  = this->layer_param_.multi_stage_crf_param().interaction_dis_std();
    Dtype dis_cv = dis_mean/dis_std;
    
    unary_composite_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (count, bottom_data, image_data, top_data, mask_data, num_, unary_channels_, height_, width_, image_channels_, user_potential, dis_cv);
}
 
template <typename Dtype>
void UnaryCompositeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                              const vector<bool>& propagate_down,
                                              const vector<Blob<Dtype>*>& bottom)
{
    if(propagate_down[0]==false)return;
    bottom[0]->CopyFrom(*top[0], true);
}
INSTANTIATE_LAYER_GPU_FUNCS(UnaryCompositeLayer);


} // namespace caffe