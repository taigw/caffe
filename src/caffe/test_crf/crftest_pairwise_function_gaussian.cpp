//#include <vector>
//#include <iostream>
//#include "gtest/gtest.h"
//
//#include "caffe/blob.hpp"
//#include "caffe/common.hpp"
//#include "caffe/filler.hpp"
//#include "caffe/crf_layers/pairwise_function_intensity_gaussian_layer.hpp"
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
//    template <typename Dtype>
//    void print_blob(const Blob<Dtype>* in, bool diff)
//    {
//        int N = in->num();
//        int C = in->channels();
//        int H = in->height();
//        int W = in->width();
//        for(int n=0; n<N; n++)
//        {
//            std::cout<<"n = "<<n<<std::endl;
//            for(int c=0; c<C; c++)
//            {
//                std::cout<<"c = "<<c<<std::endl;
//                for(int h=0; h<H; h++)
//                {
//                    for(int w=0; w<W; w++)
//                    {
//                        Dtype value = (diff)?in->diff_at(n,c,h,w): in->data_at(n,c,h,w);
//                        std::cout<<value<<" ";
//                    }
//                    std::cout<<std::endl;
//                }
//                std::cout<<std::endl;
//            }
//            std::cout<<std::endl;
//        }
//    }
//    template void print_blob(const Blob<float>* in, bool diff);
//    template void print_blob(const Blob<double>* in, bool diff);
//    
//    // Reference convolution for checking results:
//    // accumulate through explicit loops over input, output, and filters.
//    template <typename Dtype>
//    void caffe_pairwise_potential(const Blob<Dtype>* in, MultiStageCRFParameter* param,
//                                  Blob<Dtype>* out) {
//        int N = in->num();
//        int C = in->channels();
//        int H = in->height();
//        int W = in->width();
//        
//        Dtype omega1 = 5.0;
//        Dtype omega2 = 3.0;
//        Dtype theta_alpha = 15;
//        Dtype theta_beta = 0.05*3;
//        Dtype theta_gamma = 6;
//        
//        for(int n=0; n<N; n++)
//        {
//            for(int h=0; h<H; h++)
//            {
//                for(int w=0; w<W; w++)
//                {
//                    Dtype isq=0;
//                    Dtype dsq;
//                    for(int c=0; c<C; c++)
//                    {
//                        Dtype p_value = in->data_at(n,c, h, w);
//                        if(c<C-1){
//                            isq += p_value*p_value;
//                        }
//                        else {
//                            dsq = p_value*p_value;
//                        }
//                        Dtype p_term = dsq/(2 * theta_alpha * theta_alpha);
//                        Dtype i_term = isq/(2 * theta_beta * theta_beta);
//                        Dtype bilateral = exp(-i_term - p_term);
//                        Dtype spatial = exp( - dsq/(2*theta_gamma*theta_gamma));
//                        
//                        Dtype pair_potential = omega1*bilateral + omega2*spatial;
//                        out->mutable_cpu_data()[n*H*W + h*W +w] = pair_potential;
//                    }
//                    
//                }
//            }
//            
//        }
//    }
//    
//    template void caffe_pairwise_potential(const Blob<float>* in,
//                                           MultiStageCRFParameter* param,
//                                           Blob<float>* out);
//    template void caffe_pairwise_potential(const Blob<double>* in,
//                                           MultiStageCRFParameter* param,
//                                           Blob<double>* out);
//    
//    template <typename TypeParam>
//    class PairwiseFunctionGaussianLayerTest : public MultiDeviceTest<TypeParam> {
//        typedef typename TypeParam::Dtype Dtype;
//        
//    public:
//        PairwiseFunctionGaussianLayerTest()
//        : blob_bottom_(new Blob<Dtype>(1, 4, 100, 100)),
//        blob_top_(new Blob<Dtype>()) {};
//        virtual void SetUp() {
//            blob_bottom_vec_.push_back(blob_bottom_);
//            blob_top_vec_.push_back(blob_top_);
//        }
//        
//        virtual ~PairwiseFunctionGaussianLayerTest() {
//            delete blob_bottom_;
//            delete blob_top_;
//        }
//        
//        virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
//            this->ref_blob_top_.reset(new Blob<Dtype>());
//            this->ref_blob_top_->ReshapeLike(*top);
//            return this->ref_blob_top_.get();
//        }
//        Blob<Dtype>* GetBlobBottom(){return blob_bottom_;}
//        Blob<Dtype>* GetRefBlobTop(){return ref_blob_top_.get();}
//        Blob<Dtype>* GetBlobTop(){return blob_top_;}
//    protected:
//        Blob<Dtype>* blob_bottom_;
//        Blob<Dtype>* blob_top_;
//        shared_ptr<Blob<Dtype> > ref_blob_top_;
//        vector<Blob<Dtype>*> blob_bottom_vec_;
//        vector<Blob<Dtype>*> blob_top_vec_;
//    };
//    
//    TYPED_TEST_CASE(PairwiseFunctionGaussianLayerTest, TestDtypesAndDevices);
//    
//    TYPED_TEST(PairwiseFunctionGaussianLayerTest, TestSetup) {
//        typedef typename TypeParam::Dtype Dtype;
//        LayerParameter layer_param;
//        MultiStageCRFParameter* crf_param =
//        layer_param.mutable_multi_stage_crf_param();
//        
//        
//        shared_ptr<Layer<Dtype> > layer(
//                                        new PairwiseFunctionGaussianLayer<Dtype>(layer_param));
//        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//        EXPECT_EQ(this->blob_top_->num(), 1);
//        EXPECT_EQ(this->blob_top_->channels(), 1);
//        EXPECT_EQ(this->blob_top_->height(), 100);
//        EXPECT_EQ(this->blob_top_->width(), 100);
//        
//    }
//    
//    TYPED_TEST(PairwiseFunctionGaussianLayerTest, TestPairwiseFunctionGaussianForward) {
//        typedef typename TypeParam::Dtype Dtype;
//        LayerParameter layer_param;
//        MultiStageCRFParameter* crf_param =
//        layer_param.mutable_multi_stage_crf_param();
//        //  crf_param->set_kernel_size(3);
//        
//        crf_param->set_theta_alpha(15);
//        crf_param->set_theta_beta(0.05);
//        crf_param->set_theta_gamma(6);
//        
//        Blob<Dtype>* blob_bottom = this->GetBlobBottom();
//        int N=blob_bottom->num();
//        int C=blob_bottom->channels();
//        int H=blob_bottom->height();
//        int W=blob_bottom->width();
//        for(int n=0; n< N; n++)
//        {
//            for( int c=0; c<C; c++)
//            {
//                for( int h=0; h<H; h++)
//                {
//                    for(int w=0; w<W; w++)
//                    {
//                        Dtype* data=blob_bottom->mutable_cpu_data();
//                        Dtype value;
//                        if(c<C-1) {
//                            value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//                        }
//                        else{
//                            value = 3;
//                        }
//                        data[n*C*H*W + c*H*W + h*W + w] = value;
//                        //                    data[n*C*H*W + c*H*W + h*W + w] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//                    }
//                }
//            }
//        }
//        
//        shared_ptr<Layer<Dtype> > layer(
//                                        new PairwiseFunctionGaussianLayer<Dtype>(layer_param));
//        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//        //
//        // Check against reference convolution.
//        const Dtype* top_data;
//        const Dtype* ref_top_data;
//        caffe_pairwise_potential(this->blob_bottom_, crf_param,this->MakeReferenceTop(this->blob_top_));
//        top_data = this->blob_top_->cpu_data();
//        ref_top_data = this->ref_blob_top_->cpu_data();
//        
//        //    std::cout<<"bottom blob "<<std::endl;
//        //    print_blob(this->GetBlobBottom());
//        //    std::cout<<"top blob "<<std::endl;
//        //    print_blob(this->blob_top_);
//        //    std::cout<<"ref top blob "<<std::endl;
//        //    print_blob(this->GetRefBlobTop());
//        
//        for (int i = 0; i < this->blob_top_->count(); ++i) {
//            EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//        }
//        //  caffe_conv(this->blob_bottom_2_, convolution_param, layer->blobs(),
//        //      this->MakeReferenceTop(this->blob_top_2_));
//        //  top_data = this->blob_top_2_->cpu_data();
//        //  ref_top_data = this->ref_blob_top_->cpu_data();
//        //  for (int i = 0; i < this->blob_top_->count(); ++i) {
//        //    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//        //  }
//    }
//    
//    
//    TYPED_TEST(PairwiseFunctionGaussianLayerTest, TestPairwiseFunctionGaussianBackward) {
//        typedef typename TypeParam::Dtype Dtype;
//        LayerParameter layer_param;
//        MultiStageCRFParameter* crf_param =
//        layer_param.mutable_multi_stage_crf_param();
//        //  crf_param->set_kernel_size(3);
//        
//        crf_param->set_theta_alpha(15);
//        crf_param->set_theta_beta(0.05);
//        crf_param->set_theta_gamma(6);
//        
//        Blob<Dtype>* blob_bottom = this->GetBlobBottom();
//        int N=blob_bottom->num();
//        int C=blob_bottom->channels();
//        int H=blob_bottom->height();
//        int W=blob_bottom->width();
//        for(int n=0; n< N; n++)
//        {
//            for( int c=0; c<C; c++)
//            {
//                for( int h=0; h<H; h++)
//                {
//                    for(int w=0; w<W; w++)
//                    {
//                        Dtype* data=blob_bottom->mutable_cpu_data();
//                        Dtype value;
//                        if(c==0){
//                            value = 0.05;
//                        }
//                        else if(c==1)
//                        {
//                            value = 0.2;
//                        }
//                        else if(c==2){
//                            value = 0.1;
//                        }
//                        else{
//                            value = 3;
//                        }
//                        data[n*C*H*W + c*H*W + h*W + w] = value;
//                        //                    data[n*C*H*W + c*H*W + h*W + w] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//                    }
//                }
//            }
//        }
//        
//        shared_ptr<Layer<Dtype> > layer(
//                                        new PairwiseFunctionGaussianLayer<Dtype>(layer_param));
//        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//        
//        Dtype * top_diff = this->GetBlobTop()->mutable_cpu_diff();
//        for(int i=0; i<this->GetBlobTop()->count(); i++ )
//        {
//            top_diff[i]=1.0;
//        }
//        vector<bool> propagate_down(1,true);
//        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//        layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
//        
//        const Dtype* top_data;
//        const Dtype* ref_top_data;
//        caffe_pairwise_potential(this->blob_bottom_, crf_param,this->MakeReferenceTop(this->blob_top_));
//        top_data = this->blob_top_->cpu_data();
//        ref_top_data = this->ref_blob_top_->cpu_data();
//        
//        for (int i = 0; i < this->blob_top_->count(); ++i) {
//            EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
//        }
//        
//        
//        //    const Dtype * param_data = layer->blobs()[0]->cpu_data();
//        //    const Dtype * param_diff = layer->blobs()[0]->cpu_diff();
//        //    for(int i=0;i< 5; i++)
//        //    {
//        //        printf("param %d, %f, %f\n", i, param_data[i], param_diff[i]);
//        //    }
//        //    //
//        //    // Check against reference convolution.
//        //    const Dtype* top_data;
//        //    const Dtype* ref_top_data;
//        //    caffe_pairwise_potential(this->blob_bottom_, crf_param,this->MakeReferenceTop(this->blob_top_));
//        //    top_data = this->blob_top_->cpu_data();
//        //    ref_top_data = this->ref_blob_top_->cpu_data();
//        //    
//        //    std::cout<<"bottom blob "<<std::endl;
//        //    print_blob(this->GetBlobBottom(), false);
//    }
//}  // namespace caffe
