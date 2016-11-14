#include <vector>
#include <iostream>
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/crf_layers/pairwise_potential_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
void print_blob(const Blob<Dtype>* in, bool diff)
{
    int N = in->num();
    int C = in->channels();
    int H = in->height();
    int W = in->width();
    for(int n=0; n<N; n++)
    {
        std::cout<<"n = "<<n<<std::endl;
        for(int c=0; c<C; c++)
        {
            std::cout<<"c = "<<c<<std::endl;
            for(int h=0; h<H; h++)
            {
                for(int w=0; w<W; w++)
                {
                    if(diff)
                    {
                        std::cout<<in->diff_at(n,c,h,w)<<" ";
                    }
                    else
                    {
                        std::cout<<in->data_at(n,c,h,w)<<" ";
                    }
                }
                std::cout<<std::endl;
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}
template void print_blob(const Blob<float>* in, bool diff);
template void print_blob(const Blob<double>* in, bool diff);
    
template <typename TypeParam>
class PairwisePotentialLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 public:
  PairwisePotentialLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 3, 5, 5)),
    blob_top_(new Blob<Dtype>(1,8, 5, 5)) {};
  virtual void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~PairwisePotentialLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }
  Blob<Dtype>* GetBlobBottom(){return blob_bottom_;}
  Blob<Dtype>* GetBlobTop(){return blob_top_;}
  Blob<Dtype>* GetRefBlobTop(){return ref_blob_top_.get();}
protected:
  Blob<Dtype>* blob_bottom_;
  Blob<Dtype>* blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PairwisePotentialLayerTest, TestDtypesAndDevices);

TYPED_TEST(PairwisePotentialLayerTest, TestSetup_gaussian) {
  typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    MultiStageCRFParameter* crf_param =
    layer_param.mutable_multi_stage_crf_param();
    crf_param->set_kernel_size(3);
    crf_param->set_feature_length(3);
    crf_param->set_pair_wise_potential_type(MultiStageCRFParameter_PairwisePotentialType_BILATERAL_GAUSSIAN);
    
  shared_ptr<Layer<Dtype> > layer(
      new PairwisePotentialLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 8);
  EXPECT_EQ(this->blob_top_->height(), 5);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(PairwisePotentialLayerTest, PairwisePotentialForward_gaussian) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    MultiStageCRFParameter* crf_param =
    layer_param.mutable_multi_stage_crf_param();
    crf_param->set_kernel_size(3);
    crf_param->set_feature_length(3);
    crf_param->set_pair_wise_potential_type(MultiStageCRFParameter_PairwisePotentialType_BILATERAL_GAUSSIAN);
   
    Blob<Dtype>* blob_bottom = this->GetBlobBottom();
    int N=blob_bottom->num();
    int C=blob_bottom->channels();
    int H=blob_bottom->height();
    int W=blob_bottom->width();
    Dtype* data = blob_bottom->mutable_cpu_data();
    Dtype* top_diff = this->GetBlobTop()->mutable_cpu_diff();
    for(int n=0; n< N; n++)
    {
        for( int h=0; h<H; h++)
        {
            for(int w=0; w<W; w++)
            {
                for (int c=0; c<C; c++)
                {
                Dtype p0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                data[n*C*H*W + c*H*W + h*W + w] = p0-0.5;
                }
            }
        }
    }
   
    shared_ptr<Layer<Dtype> > layer(
                                    new PairwisePotentialLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//    vector<bool> propagate_down(2, true);
    layer->Forward( this->blob_bottom_vec_, this->blob_top_vec_);
    
//    std::cout<<"bottom blob "<<std::endl;
//    print_blob(this->GetBlobBottom(), false);
//    std::cout<<"top blob "<<std::endl;
//    print_blob(this->blob_top_, false);
}

TYPED_TEST(PairwisePotentialLayerTest, PairwisePotentialBackward_gaussian) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    MultiStageCRFParameter* crf_param =
    layer_param.mutable_multi_stage_crf_param();
    crf_param->set_kernel_size(3);
    crf_param->set_feature_length(3);
    crf_param->set_pair_wise_potential_type(MultiStageCRFParameter_PairwisePotentialType_BILATERAL_GAUSSIAN);
    
    Blob<Dtype>* blob_bottom = this->GetBlobBottom();
    int N=blob_bottom->num();
    int C=blob_bottom->channels();
    int H=blob_bottom->height();
    int W=blob_bottom->width();
    Dtype* data = blob_bottom->mutable_cpu_data();
    
    for(int n=0; n< N; n++)
    {
        for( int h=0; h<H; h++)
        {
            for(int w=0; w<W; w++)
            {
                for (int c=0; c<C; c++)
                {
                Dtype p0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

                data[n*C*H*W + c*H*W + h*W + w] = p0-0.5;
                }
            }
        }
    }
    
   
    shared_ptr<Layer<Dtype> > layer(
                                    new PairwisePotentialLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    
    Dtype* top_diff = this->GetBlobTop()->mutable_cpu_diff();
    for(int n=0; n< this->blob_top_->count(); n++)
    {
        top_diff[n]=1.0;
    }
    
    vector<bool> propagate_down(1, true);
    layer->Backward( this->blob_top_vec_, propagate_down, this->blob_bottom_vec_ );
    
//    std::cout<<"top diff "<<std::endl;
//    print_blob(this->blob_top_, true);
//    std::cout<<"bottom diff "<<std::endl;
//    print_blob(this->GetBlobBottom(), true);
//    std::cout<<"bottom data "<<std::endl;
//    print_blob(this->GetBlobBottom(), false);
}

TYPED_TEST(PairwisePotentialLayerTest, PairwisePotentialForward_freeform) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    MultiStageCRFParameter* crf_param =
    layer_param.mutable_multi_stage_crf_param();
    crf_param->set_kernel_size(3);
    crf_param->set_feature_length(3);
    crf_param->set_pair_wise_potential_type(MultiStageCRFParameter_PairwisePotentialType_FREEFORM_FUNCTION);
    
    crf_param->mutable_pairwise_potential_net_size()->Add(4);
    crf_param->mutable_pairwise_potential_net_size()->Add(64);
    crf_param->mutable_pairwise_potential_net_size()->Add(32);
    crf_param->mutable_pairwise_potential_net_size()->Add(1);
    crf_param->set_pairwise_potential_net_param_path("/Users/guotaiwang/Documents/workspace/neuronnetwork/pretranPairwiseNet/model");
    
    Blob<Dtype>* blob_bottom = this->GetBlobBottom();
    int N=blob_bottom->num();
    int C=blob_bottom->channels();
    int H=blob_bottom->height();
    int W=blob_bottom->width();
    Dtype* data = blob_bottom->mutable_cpu_data();
    Dtype* top_diff = this->GetBlobTop()->mutable_cpu_diff();
    for(int n=0; n< N; n++)
    {
        for( int h=0; h<H; h++)
        {
            for(int w=0; w<W; w++)
            {
                for (int c=0; c<C; c++)
                {
                    Dtype p0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                    data[n*C*H*W + c*H*W + h*W + w] = p0-0.5;
                }
            }
        }
    }
    
    shared_ptr<Layer<Dtype> > layer(
                                    new PairwisePotentialLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    //    vector<bool> propagate_down(2, true);
    layer->Forward( this->blob_bottom_vec_, this->blob_top_vec_);
    
//        std::cout<<"bottom blob "<<std::endl;
//        print_blob(this->GetBlobBottom(), false);
//        std::cout<<"top blob "<<std::endl;
//        print_blob(this->blob_top_, false);
}
}  // namespace caffe
