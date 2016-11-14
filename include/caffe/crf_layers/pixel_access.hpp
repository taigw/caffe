#ifndef CAFFE_PIXEL_ACCESS_HPP_
#define CAFFE_PIXEL_ACCESS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/util/modified_permutohedral.hpp"
//#include "caffe/proto/caffe.pb.h"

#include <boost/shared_array.hpp>

namespace caffe {

template <typename Dtype>
Dtype get_pixel(const Dtype * data, int N, int C, int H, int W, int n, int c, int h, int w)
{
    Dtype result;
    if(n<0 || n>=N || c<0 || c>=C || h<0 || h>=H || w<0 || w>=W)
    {
        result=0;
    }
    else
    {
        result=data[((n * C + c) * H + h) * W + w];
    }
    return result;
};
    
template <typename Dtype>
void set_pixel(Dtype * data, int N, int C, int H, int W, int n, int c, int h, int w, Dtype value)
{
    if(n<0 || n>=N || c<0 || c>=C || h<0 || h>=H || w<0 || w>=W)
    {
        return;
    }
    else
    {
        data[((n * C + c) * H + h) * W + w]=value;
    }
}
    
}

#endif  // CAFFE_PIXEL_ACCESS_HPP_
