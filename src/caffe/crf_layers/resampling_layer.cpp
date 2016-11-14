#include <algorithm>
#include <cfloat>
#include <vector>
#include <math.h>

#include "caffe/util/math_functions.hpp"
#include "caffe/crf_layers/resampling_layer.hpp"
#include "caffe/crf_layers/pixel_access.hpp"

namespace caffe {
    
using std::min;
using std::max;
template <typename Dtype>
void ResamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top)
{
    sample_rate_ = this->layer_param_.resample_param().sample_rate();
}

template <typename Dtype>
void ResamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
    << "corresponding to (num, channels, height, width)";
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
 
    sampled_height_ = height_/sample_rate_;
    sampled_width_ = width_/sample_rate_;
 
    top[0]->Reshape(num_, channels_, sampled_height_, sampled_width_);
}

    
// revise from Backword_cpu of pooling layer
template <typename Dtype>
void ResamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    const Dtype * bottom_data = bottom[0]->cpu_data();
    Dtype * top_data = top[0]->mutable_cpu_data();
    for( int n=0; n<num_; n++)
    {
        for(int c=0; c<channels_; c++)
        {
            for(int h=0; h<sampled_height_; h++)
            {
                for(int w=0; w<sampled_width_; w++)
                {
                    float hy = h*sample_rate_;
                    float wx = w*sample_rate_;
                    if(hy > height_ -1 ) hy = height_ - 1;
                    if(wx > width_ -1) wx = width_ -1;
                    
                    int h1 = floor(hy);
                    int h2 = ceil(hy);
                    int w1 = floor(wx);
                    int w2 = ceil(wx);
                    float x = (w2-w1>0)? (wx-w1)/(w2-w1) : 0;
                    float y = (h2-h1>0)? (hy-h1)/(h2-h1) : 0;
                    Dtype Q11 = get_pixel(bottom_data, num_, channels_, height_, width_, n, c, h1, w1);
                    Dtype Q12 = get_pixel(bottom_data, num_, channels_, height_, width_, n, c, h2, w1);
                    Dtype Q21 = get_pixel(bottom_data, num_, channels_, height_, width_, n, c, h1, w2);
                    Dtype Q22 = get_pixel(bottom_data, num_, channels_, height_, width_, n, c, h2, w2);
                    Dtype value = Q11*(1-x)*(1-y) + Q12*(1-x)*y + Q21*x*(1-y) + Q22*x*y;
                    set_pixel(top_data, num_, channels_, sampled_height_, sampled_width_, n, c, h, w, value);
                    
                }
            }
        }
    }
  }
    
template <typename Dtype>
void ResamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
        return;
    }
    
    for( int n=0; n<num_; n++)
    {
        for(int c=0; c<channels_; c++)
        {
            for(int bh=0; bh<height_; bh++)
            {
                for(int bw=0; bw<width_; bw++)
                {
                    int th = bh/sample_rate_;
                    int tw = bw/sample_rate_;
                    int th1 = floor((bh-1)/sample_rate_);
                    int th2 = ceil((bh+1)/sample_rate_);
                    if(th1<0) th1 = 0;
                    if( th2 > sampled_height_-1) th2 = sampled_height_-1;
                    
                    int tw1 = floor((bw-1)/sample_rate_);
                    int tw2 = ceil((bw+1)/sample_rate_);
                    if(tw1<0) tw1=0;
                    if(tw2 > sampled_width_-1) tw2 =  sampled_width_-1;
                    
                    Dtype sum_diff = 0.0;
                    for(int thIdx = th1+1; thIdx< th2; thIdx++)
                    {
                        for(int twIdx = tw1+1; twIdx< tw2; twIdx++)
                        {
                            Dtype t_diff_value = get_pixel(top[0]->cpu_diff(), num_, channels_,
                               sampled_height_, sampled_width_, n, c, thIdx, twIdx);

                            float hy = thIdx*sample_rate_;
                            float wx = twIdx*sample_rate_;
                            if(hy > height_ -1 ) hy = height_ - 1;
                            if(wx > width_ -1) wx = width_ -1;
                            
                            int h1 = floor(hy);
                            int h2 = ceil(hy);
                            int w1 = floor(wx);
                            int w2 = ceil(wx);
                            float x = (w2-w1>0)? (wx-w1)/(w2-w1) : 0;
                            float y = (h2-h1>0)? (hy-h1)/(h2-h1) : 0;
                            
                            Dtype diff;
                            if(thIdx<= th)
                            {
                                diff = (twIdx <= tw)? x*y*t_diff_value : (1-x)*y*t_diff_value;
                            }
                            else
                            {
                                diff = (twIdx <= tw)? x*(1-y)*t_diff_value : (1-x)*(1-y)*t_diff_value;
                            }
                            sum_diff += diff;
                        }
                    }
                    set_pixel(bottom[0]->mutable_cpu_diff(), num_, channels_, height_, width_,
                              n, c, bh, bw, sum_diff);
                }
            }
        }
    }
}
    
#ifdef CPU_ONLY
    STUB_GPU(ResamplingLayer);
#endif
    
INSTANTIATE_CLASS(ResamplingLayer);
REGISTER_LAYER_CLASS(Resampling);
    
} // namespace caffe