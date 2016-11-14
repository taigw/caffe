#ifndef CAFFE_DEPOOLING_LAYER_HPP_
#define CAFFE_DEPOOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
//#include "caffe/layers/pooling_layer.h"

namespace caffe {
    
    /**
     * @brief Pools the input image by taking the max, average, etc. within regions.
     *
     * TODO(dox): thorough documentation for Forward, Backward, and proto params.
     */
    template <typename Dtype>
    class ResamplingLayer : public Layer<Dtype> {
    public:
        // bottom[0] the blob will be resampled
        explicit ResamplingLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const { return "Resampling"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
//        virtual inline int MinTopBlobs() const { return 1; }
//        // MAX POOL layers can output an extra top blob for the mask;
//        // others can only output the pooled inputs.
//        virtual inline int MaxTopBlobs() const {
//            return (this->layer_param_.pooling_param().pool() ==
//                    PoolingParameter_PoolMethod_MAX) ? 2 : 1;
//        }
        
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    void resample(const Dtype * data, Dtype * sampled_data, int N, int C, int H, int W, int sH, int sW, float sample_rate);
    int num_;
    int channels_;
    int height_, width_;
    int sampled_height_, sampled_width_;
    float sample_rate_;


    };
    
} // namespace caffe
#endif  // CAFFE_DEPOOLING_LAYER_HPP_
