//#include <vector>
//#include <iostream>
//#include "gtest/gtest.h"
//
//#include "caffe/blob.hpp"
//#include "caffe/common.hpp"
//#include "caffe/filler.hpp"
//#include "caffe/crf_layers/unary_composite_layer.hpp"
//
////#ifdef USE_CUDNN
////#include "caffe/layers/cudnn_conv_layer.hpp"
////#endif
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
//
//template <typename TypeParam>
//class UnaryPotentialLayerTest : public MultiDeviceTest<TypeParam> {
//  typedef typename TypeParam::Dtype Dtype;
//
// public:
//  UnaryPotentialLayerTest()
//      : blob_bottom_(new Blob<Dtype>(1, 2, 6, 6)),
//    blob_top_(new Blob<Dtype>()) {};
//  virtual void SetUp() {
//    blob_bottom_vec_.push_back(blob_bottom_);
//    blob_top_vec_.push_back(blob_top_);
//  }
//
//  virtual ~UnaryPotentialLayerTest() {
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
//protected:
//  Blob<Dtype>* blob_bottom_;
//  Blob<Dtype>* blob_top_;
//  shared_ptr<Blob<Dtype> >ref_blob_top_;
//  vector<Blob<Dtype>*> blob_bottom_vec_;
//  vector<Blob<Dtype>*> blob_top_vec_;
//};
//
//TYPED_TEST_CASE(UnaryPotentialLayerTest, TestDtypesAndDevices);
//
//TYPED_TEST(UnaryPotentialLayerTest, TestSetup) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  shared_ptr<Layer<Dtype> > layer(
//      new UnaryPotentialLayer<Dtype>(layer_param));
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  EXPECT_EQ(this->blob_top_->num(), 1);
//  EXPECT_EQ(this->blob_top_->channels(), 2);
//  EXPECT_EQ(this->blob_top_->height(), 6);
//  EXPECT_EQ(this->blob_top_->width(), 6);
//
//}
//
//TYPED_TEST(UnaryPotentialLayerTest, UnaryPotentialForwardTest) {
//    typedef typename TypeParam::Dtype Dtype;
//    LayerParameter layer_param;
//    Blob<Dtype>* blob_bottom = this->GetBlobBottom();
//    int N=blob_bottom->num();
//    int C=blob_bottom->channels();
//    int H=blob_bottom->height();
//    int W=blob_bottom->width();
//    for(int n=0; n< N; n++)
//    {
//            for( int h=0; h<H; h++)
//            {
//                for(int w=0; w<W; w++)
//                {
//                    Dtype* data=blob_bottom->mutable_cpu_data();
//                    int c=0;
//                    data[n*C*H*W + c*H*W + h*W + w] = 0.2;
//                    c=1;
//                    data[n*C*H*W + c*H*W + h*W + w] =0.8;
//                }
//            }
//    }
//    
//
//  shared_ptr<Layer<Dtype> > layer(
//      new UnaryPotentialLayer<Dtype>(layer_param));
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//
////  std::cout<<"bottom blob "<<std::endl;
////  print_blob(this->GetBlobBottom(), false);
////  std::cout<<"top blob "<<std::endl;
////  print_blob(this->blob_top_, false);
//
//}
//
//    
//TYPED_TEST(UnaryPotentialLayerTest, UnaryPotentialBackwardTest) {
//    typedef typename TypeParam::Dtype Dtype;
//    LayerParameter layer_param;
//    
//    Blob<Dtype>* blob_bottom = this->GetBlobBottom();
//    int N=blob_bottom->num();
//    int C=blob_bottom->channels();
//    int H=blob_bottom->height();
//    int W=blob_bottom->width();
//    for(int n=0; n< N; n++)
//    {
//        for( int h=0; h<H; h++)
//        {
//            for(int w=0; w<W; w++)
//            {
//                Dtype* data=blob_bottom->mutable_cpu_data();
//                int c=0;
//                data[n*C*H*W + c*H*W + h*W + w] = 0.2;
//                c=1;
//                data[n*C*H*W + c*H*W + h*W + w] =0.8;
//            }
//        }
//    }
//    
//    
//    shared_ptr<Layer<Dtype> > layer(
//                                    new UnaryPotentialLayer<Dtype>(layer_param));
//    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//    
//    Blob<Dtype>* blob_top = this->GetBlobTop();
//    int Nt=blob_top->num();
//    int Ct=blob_top->channels();
//    int Ht=blob_top->height();
//    int Wt=blob_top->width();
//    for(int n=0; n< Nt; n++)
//    {
//        for( int c=0; c<Ct; c++)
//        {
//            for( int h=0; h<Ht; h++)
//            {
//                for(int w=0; w<Wt; w++)
//                {
//                    Dtype* top_diff=blob_top->mutable_cpu_diff();
//                    top_diff[n*Ct*Ht*Wt + c*Ht*Wt + h*Wt + w] = 1.0;//(c+1)*(h+w);
//                }
//            }
//        }
//    }
//
//    vector<bool> propagate_down(1,true);
//    layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
//
////    std::cout<<"top blob diff"<<std::endl;
////    print_blob(this->blob_top_, true);
////    
////    std::cout<<"bottom blob diff"<<std::endl;
////    print_blob(this->GetBlobBottom(), true);
//}
//}  // namespace caffe
