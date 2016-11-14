#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/crf_layers/resampling_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "pixel_access.cu"

namespace caffe {
    
    
template <typename Dtype>
__global__ void resample_forward_kernel(const int nthreads, const Dtype* data, Dtype* sampled_data,
        int N, int C, int H, int W, int sH, int sW, float sample_rate)
{
    CUDA_KERNEL_LOOP(index, nthreads){
        const int w = index % sW;
        const int h = (index/sW) % sH;
        const int c = (index / sW / sH) % C;
        const int n = index / sW / sH / C;

        float hy = h*sample_rate;
        float wx = w*sample_rate;
        if(hy > H -1 ) hy = H - 1;
        if(wx > W -1) wx = W -1;
        
        int h1 = floor(hy);
        int h2 = ceil(hy);
        int w1 = floor(wx);
        int w2 = ceil(wx);
        float x = (w2-w1>0)? (wx-w1)/(w2-w1) : 0;
        float y = (h2-h1>0)? (hy-h1)/(h2-h1) : 0;
        Dtype Q11 = get_gpu_pixel(data, N, C, H, W, n, c, h1, w1);
        Dtype Q12 = get_gpu_pixel(data, N, C, H, W, n, c, h2, w1);
        Dtype Q21 = get_gpu_pixel(data, N, C, H, W, n, c, h1, w2);
        Dtype Q22 = get_gpu_pixel(data, N, C, H, W, n, c, h2, w2);
        Dtype value = Q11*(1-x)*(1-y) + Q12*(1-x)*y +
            Q21*x*(1-y) + Q22*x*y;
        set_gpu_pixel(sampled_data, N, C, sH, sW, n, c, h, w, value);
    }
}
    
template <typename Dtype>
void ResamplingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    const Dtype * bottom_data = bottom[0]->gpu_data();
    Dtype * top_data = top[0]->mutable_gpu_data();
    int count = top[0]->count();
    resample_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
    (count, bottom_data, top_data, num_, channels_, height_, width_,
        sampled_height_, sampled_width_,  sample_rate_);
}
    
template <typename Dtype>
__global__ void resample_backward_kernel(const int nthreads, const Dtype* top_diff,
     Dtype* bottom_diff,int N, int C, int sH, int sW, int H, int W, float sample_rate)
{
    CUDA_KERNEL_LOOP(index, nthreads){
        const int bw = index % W;
        const int bh = (index/W) % H;
        const int c = (index / W / H) % C;
        const int n = index / W / H / C;

        int th = bh/sample_rate;
        int tw = bw/sample_rate;
        int th1 = floor((bh-1)/sample_rate);
        int th2 = ceil((bh+1)/sample_rate);
        if(th1<0) th1 = 0;
        if( th2 > sH-1) th2 = sH-1;
        
        int tw1 = floor((bw-1)/sample_rate);
        int tw2 = ceil((bw+1)/sample_rate);
        if(tw1<0) tw1=0;
        if(tw2 > sW-1) tw2 = sW-1;

        Dtype sum_diff = 0.0;
        for(int thIdx = th1+1; thIdx< th2; thIdx++)
        {
            for(int twIdx = tw1+1; twIdx< tw2; twIdx++)
            {
                Dtype t_diff_value = get_gpu_pixel(top_diff, N, C, sH, sW, n, c, thIdx, twIdx);
                
                float hy = thIdx*sample_rate;
                float wx = twIdx*sample_rate;
                if(hy > H -1 ) hy = H - 1;
                if(wx > W -1) wx = W -1;

                int h1 = floor(hy);
                int h2 = ceil(hy);
                int w1 = floor(wx);
                int w2 = ceil(wx);
                float x = (w2-w1>0)? (wx-w1)/(w2-w1) : 0;
                float y = (h2-h1>0)? (hy-h1)/(h2-h1) : 0;

                Dtype diff = 0.0;
                if(thIdx<= th)
                {
                    diff = (twIdx <= tw)? (x*y*t_diff_value) : ((1.0-x)*y*t_diff_value);
                }
                else
                {
                    diff = (twIdx <= tw)? (x*(1.0-y)*t_diff_value) : ((1.0-x)*(1.0-y)*t_diff_value);
                }
                sum_diff += diff;
            }
        }
        set_gpu_pixel(bottom_diff, N, C, H, W, n, c, bh, bw, sum_diff);
    }
}

template <typename Dtype>
void ResamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
        return;
    }
    const Dtype * top_diff = top[0]->gpu_diff();
    Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
    int count = bottom[0]->count();
    resample_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        (count, top_diff, bottom_diff, num_, channels_, sampled_height_,
         sampled_width_, height_, width_, sample_rate_);
}


INSTANTIATE_LAYER_GPU_FUNCS(ResamplingLayer);


} // namespace caffe