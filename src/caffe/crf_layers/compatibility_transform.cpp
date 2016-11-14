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
#include "caffe/crf_layers/compatibility_transform_layer.hpp"
#include "caffe/crf_layers/pixel_access.hpp"

namespace caffe {
template <typename Dtype>
void CompatibilityTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top)
{
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    num_pixels_ = height_ * width_;

    CHECK((bottom[0]->channels() == bottom[1]->height()) && (bottom[0]->channels() == bottom[1]->width()))<<
    ("input image and compatibility matrix shoud have the channel number");
    // bottom[1] is compatibility_param_blob, size: (channels_, channels_, 1, 1)
//    compatibility_param_blob_.reset(new Blob<Dtype>(channels_, channels_, 1, 1));
//    caffe_set(channels_ * channels_, Dtype(1.), compatibility_param_blob_->mutable_cpu_data());
//    for (int c = 0; c < channels_; ++c) {
//        (compatibility_param_blob_->mutable_cpu_data())[c * channels_ + c] = Dtype(0.);
//    }
}
    
template <typename Dtype>
void CompatibilityTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top)
{
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    top[0]->Reshape(num_, channels_, height_, width_);
}
template <typename Dtype>
void CompatibilityTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                     const vector<Blob<Dtype>*>& top)
{
//    Dtype * compa_data = bottom[1]->mutable_cpu_data();
//    compa_data[0]=-1;
//    compa_data[1]=0;
//    compa_data[2]=0;
//    compa_data[3]=-1;
    //Result from message passing needs to be multiplied with compatibility values.
    LOG(INFO)<<"cpu Compatibility "<<bottom[1]->cpu_data()[0]<<" "<<bottom[1]->cpu_data()[1]<<" "<<
        bottom[1]->cpu_data()[2]<<" "<<bottom[1]->cpu_data()[3];
    for (int n = 0; n < num_; ++n) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                              bottom[1]->cpu_data(),
                              bottom[0]->cpu_data() + bottom[0]->offset(n), (Dtype) 0.,
                              top[0]->mutable_cpu_data() + top[0]->offset(n));
    }
}

template <typename Dtype>
void CompatibilityTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{

    
    for (int n = 0; n < num_; ++n) {
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                              channels_, (Dtype) 1., bottom[1]->cpu_data(),
                              top[0]->cpu_diff() + top[0]->offset(n), (Dtype) 0.,
                              bottom[0]->mutable_cpu_diff() + bottom[0]->offset(n));
        
        // gardient to compatibility values
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_, num_pixels_,
                              (Dtype) 1., top[0]->cpu_diff() + top[0]->offset(n),
                              bottom[0]->cpu_data()+bottom[0]->offset(n),(Dtype) 0.,
                              bottom[1]->mutable_cpu_diff());
    }
}

template <typename Dtype>
void CompatibilityTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                     const vector<Blob<Dtype>*>& top)
{
//    LOG(INFO)<<"gpu Compatibility "<<bottom[1]->cpu_data()[0]<<" "<<bottom[1]->cpu_data()[1]<<" "<<
//        bottom[1]->cpu_data()[2]<<" "<<bottom[1]->cpu_data()[3];
    //Result from message passing needs to be multiplied with compatibility values.
    for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                              bottom[1]->gpu_data(),
                              bottom[0]->gpu_data() + bottom[0]->offset(n), (Dtype) 0.,
                              top[0]->mutable_gpu_data() + top[0]->offset(n));
    }
}

template <typename Dtype>
void CompatibilityTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                      const vector<bool>& propagate_down,
                                                      const vector<Blob<Dtype>*>& bottom)
{
    for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                              channels_, (Dtype) 1., bottom[1]->gpu_data(),
                              top[0]->gpu_diff() + top[0]->offset(n), (Dtype) 0.,
                              bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n));
        
        // gardient to compatibility values
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_, num_pixels_,
                              (Dtype) 1., top[0]->gpu_diff() + top[0]->offset(n),
                              bottom[0]->gpu_data()+bottom[0]->offset(n),(Dtype) 0.,
                              bottom[1]->mutable_gpu_diff());
    }
}
INSTANTIATE_CLASS(CompatibilityTransformLayer);
}  // namespace caffe
