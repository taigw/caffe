//#include <vector>
//#include <iostream>
//#include "gtest/gtest.h"
//
//#include "caffe/blob.hpp"
//#include "caffe/common.hpp"
//#include "caffe/filler.hpp"
//#include "caffe/crf_layers/pairwise_feature_layer.hpp"
//
//#ifdef USE_CUDNN
//#include "caffe/layers/cudnn_conv_layer.hpp"
//#endif
//
//#include "caffe/test/test_caffe_main.hpp"
//#include "caffe/test/test_gradient_check_util.hpp"
//
//namespace caffe {
//
//template <typename Dtype>
//void print_blob(const Blob<Dtype>* in, bool diff)
//{
//    int N = in->num();
//    int C = in->channels();
//    int H = in->height();
//    int W = in->width();
//    for(int n=0; n<N; n++)
//    {
//        std::cout<<"n = "<<n<<std::endl;
//        for(int c=0; c<C; c++)
//        {
//            std::cout<<"c = "<<c<<std::endl;
//            for(int h=0; h<H; h++)
//            {
//                for(int w=0; w<W; w++)
//                {
//                    if(diff)
//                    {
//                        std::cout<<in->diff_at(n,c,h,w)<<" ";
//                    }
//                    else
//                    {
//                        std::cout<<in->data_at(n,c,h,w)<<" ";
//                    }
//                }
//                std::cout<<std::endl;
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<std::endl;
//    }
//}
//template void print_blob(const Blob<float>* in, bool diff);
//template void print_blob(const Blob<double>* in, bool diff);
//    
//template <typename TypeParam>
//class PairwiseFeatureLayerTest : public MultiDeviceTest<TypeParam> {
//  typedef typename TypeParam::Dtype Dtype;
//
// public:
//  PairwiseFeatureLayerTest()
//      : blob_bottom_(new Blob<Dtype>(1, 3, 4, 4)),
//    blob_top_(new Blob<Dtype>(1,4,8,16)) {};
//  virtual void SetUp() {
//    blob_bottom_vec_.push_back(blob_bottom_);
//    blob_top_vec_.push_back(blob_top_);
//  }
//
//  virtual ~PairwiseFeatureLayerTest() {
//    delete blob_bottom_;
//    delete blob_top_;
//  }
//
//  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
//    this->ref_blob_top_.reset(new Blob<Dtype>());
//    this->ref_blob_top_->ReshapeLike(*top);
//    return this->ref_blob_top_.get();
//  }
//  Blob<Dtype>* GetBlobBottom(){return blob_bottom_;}
//  Blob<Dtype>* GetBlobTop(){return blob_top_;}
//  Blob<Dtype>* GetRefBlobTop(){return ref_blob_top_.get();}
//protected:
//  Blob<Dtype>* blob_bottom_;
//  Blob<Dtype>* blob_top_;
//  shared_ptr<Blob<Dtype> > ref_blob_top_;
//  vector<Blob<Dtype>*> blob_bottom_vec_;
//  vector<Blob<Dtype>*> blob_top_vec_;
//};
//
//TYPED_TEST_CASE(PairwiseFeatureLayerTest, TestDtypesAndDevices);
//
//TYPED_TEST(PairwiseFeatureLayerTest, TestSetup) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  MultiStageCRFParameter* crf_param =
//      layer_param.mutable_multi_stage_crf_param();
//  crf_param->set_kernel_size(3);
//  
//  shared_ptr<Layer<Dtype> > layer(
//      new PairwiseFeatureLayer<Dtype>(layer_param));
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  EXPECT_EQ(this->blob_top_->num(), 1);
//  EXPECT_EQ(this->blob_top_->channels(), 4);
//  EXPECT_EQ(this->blob_top_->height(), 8);
//  EXPECT_EQ(this->blob_top_->width(), 16);
//
//}
//
//TYPED_TEST(PairwiseFeatureLayerTest, PairwiseFeatureForward) {
//    typedef typename TypeParam::Dtype Dtype;
//    LayerParameter layer_param;
//    MultiStageCRFParameter* crf_param =
//    layer_param.mutable_multi_stage_crf_param();
//    crf_param->set_kernel_size(3);
//    crf_param->set_feature_length(3);
//    
//    Blob<Dtype>* blob_bottom = this->GetBlobBottom();
//    int N=blob_bottom->num();
//    int C=blob_bottom->channels();
//    int H=blob_bottom->height();
//    int W=blob_bottom->width();
//    Dtype* data = blob_bottom->mutable_cpu_data();
//    Dtype* top_diff = this->GetBlobTop()->mutable_cpu_diff();
//    for(int n=0; n< N; n++)
//    {
//        for( int h=0; h<H; h++)
//        {
//            for(int w=0; w<W; w++)
//            {
//                Dtype p0 = 0.9;//static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//                Dtype p1 = 1.0- p0;
//                
//                int c=0;
//                data[n*C*H*W + c*H*W + h*W + w] = p0;
//                c=1;
//                data[n*C*H*W + c*H*W + h*W + w] = p1;
//                
//            }
//        }
//    }
//   
//    
//    shared_ptr<Layer<Dtype> > layer(
//                                    new PairwiseFeatureLayer<Dtype>(layer_param));
//    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
////    vector<bool> propagate_down(2, true);
//    layer->Forward( this->blob_bottom_vec_, this->blob_top_vec_);
//    
////    std::cout<<"bottom blob "<<std::endl;
////    print_blob(this->GetBlobBottom(), false);
////    std::cout<<"top blob "<<std::endl;
////    print_blob(this->blob_top_, false);
//}
//
//TYPED_TEST(PairwiseFeatureLayerTest, PairwiseFeatureBackward) {
//    typedef typename TypeParam::Dtype Dtype;
//    LayerParameter layer_param;
//    MultiStageCRFParameter* crf_param =
//    layer_param.mutable_multi_stage_crf_param();
//    crf_param->set_kernel_size(3);
//    crf_param->set_feature_length(3);
//    
//    Blob<Dtype>* blob_bottom = this->GetBlobBottom();
//    int N=blob_bottom->num();
//    int C=blob_bottom->channels();
//    int H=blob_bottom->height();
//    int W=blob_bottom->width();
//    Dtype* data = blob_bottom->mutable_cpu_data();
//    Dtype* top_diff = this->GetBlobTop()->mutable_cpu_diff();
//    for(int n=0; n< N; n++)
//    {
//        for( int h=0; h<H; h++)
//        {
//            for(int w=0; w<W; w++)
//            {
//                Dtype p0 = 0.9;//static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//                Dtype p1 = 1.0- p0;
//                
//                int c=0;
//                data[n*C*H*W + c*H*W + h*W + w] = p0;
//                c=1;
//                data[n*C*H*W + c*H*W + h*W + w] = p1;
//            }
//        }
//    }
//    
//    for(int n=0; n< this->blob_top_->count(); n++)
//    {
//        top_diff[n]=1.0;
//    }
//    
//    shared_ptr<Layer<Dtype> > layer(
//                                    new PairwiseFeatureLayer<Dtype>(layer_param));
//    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//    vector<bool> propagate_down(1, true);
//    layer->Backward( this->blob_top_vec_, propagate_down, this->blob_bottom_vec_ );
//    
////    std::cout<<"top diff "<<std::endl;
////    print_blob(this->blob_top_, true);
////    std::cout<<"bottom diff "<<std::endl;
////    print_blob(this->GetBlobBottom(), true);
////    std::cout<<"bottom data "<<std::endl;
////    print_blob(this->GetBlobBottom(), false);
//}
//}  // namespace caffe
