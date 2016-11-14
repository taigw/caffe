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
#include "caffe/crf_layers/pairwise_function_freeform_layer.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

namespace caffe {
template <typename Dtype>
bool PairwiseFunctionFreeformLayer<Dtype>::LoadParamFromFile(shared_ptr<vector<vector<Dtype> > > matrix,
                                                     int H, int W, std::string filename)
{
    vector<vector<Dtype> > * matrix_data = matrix.get();
    matrix_data->resize(H);
    std::ifstream param_fstream;
    param_fstream.open(filename.c_str(), std::ifstream::in);
    int loadedN=0;

    for(int i=0; i<H; i++)
    {
        vector<Dtype> temprow;
        temprow.resize(W);
        for(int j=0; j<W; j++)
        {
            if( !param_fstream.good()) break;
            Dtype a;
            param_fstream >> a;
            temprow[j] = a;
            loadedN++;
        }
        (*matrix_data)[i] = temprow;
    }
    param_fstream.close();
    return H*W == loadedN;
}
    
template <typename Dtype>
bool PairwiseFunctionFreeformLayer<Dtype>::CopyParamToBlob(shared_ptr<vector<vector<Dtype> > > matrix,
                                                   shared_ptr<Blob<Dtype> > blob)
{
    vector<vector<Dtype> > * matrix_data = matrix.get();
    int H = matrix_data->size();
    int W = (*matrix_data)[0].size();
    int blob_size = blob->count();
    Dtype * blob_data = blob->mutable_cpu_data();
    if(H*W == blob_size)
    {
        int idx = 0;
        for(int i=0; i<H; i++)
        {
            for(int j=0; j<W; j++)
            {
                blob_data[idx] = (* matrix_data)[i][j];
                idx++;
            }
        }
        return true;
    }
    else
    {
        return false;
    }
}
    
template <typename Dtype>
void PairwiseFunctionFreeformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top)
{
    LOG(INFO) << ("PairwiseFunctionFreeformLayer entered ");
    count_ = bottom[0]->count();
    num_ = bottom[0]->num();
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
//    num_pixels_ = height_ * width_;
    //construct_hidden_layers(bottom, top);
    inner_layer_size_.clear();
    int tempsize = this->layer_param_.multi_stage_crf_param().pairwise_potential_net_size().size();
    CHECK( tempsize>=2 )<<
        ("pairwise potential network should have at least an input size and output size");
    for(int i=0; i<tempsize; i++)
    {
        int temp_channel=this->layer_param_.multi_stage_crf_param().pairwise_potential_net_size().Get(i);
        inner_layer_size_.push_back(temp_channel);
    }
    CHECK( bottom[0]->channels() == inner_layer_size_[0] )<<
          ("input feature number is not compatible with the pairwise potential network");
    CHECK( 1 == inner_layer_size_[tempsize-1] )<<
          ("output feature number of the pairwise potential network shoud be 1");
    
    string param_dir = this->layer_param_.multi_stage_crf_param().pairwise_potential_net_param_path();

    layerN_ = inner_layer_size_.size()-1;
    LOG(INFO) << ("PairwiseFunctionLayer entered ")<< layerN_;
    conv_out_blobs_.resize(layerN_);
    conv_layers_.resize(layerN_);
    conv_bottom_vec_.resize(layerN_);
    conv_top_vec_.resize(layerN_);
    
    relu_out_blobs_.resize(layerN_);
    relu_layers_.resize(layerN_);
    relu_bottom_vec_.resize(layerN_);
    relu_top_vec_.resize(layerN_);
    LOG(INFO) << ("try to create convlolution layers ");
    for(int i=0; i< layerN_; i++)
    {
        int input_n = inner_layer_size_[i];
        int output_n= inner_layer_size_[i+1];
        conv_out_blobs_[i].reset(new Blob<Dtype>());
        relu_out_blobs_[i].reset(new Blob<Dtype>());
        LOG(INFO) << ("blobs created ");
        // convolution layers
        LayerParameter conv_param;
        conv_param.mutable_convolution_param()->set_num_output(output_n);
        LOG(INFO) << ("conv_param set num_output done ");
        conv_param.mutable_convolution_param()->set_kernel_h(1);
        conv_param.mutable_convolution_param()->set_kernel_w(1);
        LOG(INFO) << ("conv_param set kernel_size done ");
        conv_param.mutable_convolution_param()->mutable_weight_filler()->set_type("gaussian");
        conv_param.mutable_convolution_param()->mutable_weight_filler()->set_mean(0.0);
        conv_param.mutable_convolution_param()->mutable_weight_filler()->set_std(sqrt(2.0/input_n));
        LOG(INFO) << ("conv_param created ");
        conv_layers_[i].reset(new ConvolutionLayer<Dtype>(conv_param));
        LOG(INFO) << ("conv_layers_ created ")<< i;
        vector<Blob<Dtype> *> temp_conv_bottom_vec_;
        vector<Blob<Dtype> *> temp_conv_top_vec_;
        temp_conv_top_vec_.clear();
        temp_conv_top_vec_.push_back(conv_out_blobs_[i].get());
        temp_conv_bottom_vec_.resize(1);
        temp_conv_bottom_vec_[0]= (i==0)? bottom[0] : relu_out_blobs_[i-1].get();
        
        conv_bottom_vec_[i]=temp_conv_bottom_vec_;
        conv_top_vec_[i]=temp_conv_top_vec_;
        conv_layers_[i]->SetUp(conv_bottom_vec_[i], conv_top_vec_[i]);
        LOG(INFO) << ("param_dir ")<< param_dir;
        if(param_dir.length() > 0)
        {
            shared_ptr<Blob<Dtype> > w_blob = conv_layers_[i]->blobs()[0];
            shared_ptr<Blob<Dtype> > b_blob = conv_layers_[i]->blobs()[1];
            
            std::ostringstream ss;
            ss << i;
            string w_file_name = param_dir + "/w" + ss.str() + ".txt";
            shared_ptr<vector<vector<Dtype> > > w_buffer(new vector<vector<Dtype> >);
            if(LoadParamFromFile(w_buffer, output_n, input_n, w_file_name))
            {
                if(CopyParamToBlob(w_buffer, w_blob))
                {
                    LOG(INFO)<<"initialized weight with "<<w_file_name;
                }
            }
            else{
                LOG(INFO)<<"LoadParamFromFile  failed "<<w_file_name;
            }
            
            string b_file_name = param_dir + "/b" + ss.str() + ".txt";
            shared_ptr<vector<vector<Dtype> > > b_buffer(new vector<vector<Dtype> >);
            if(LoadParamFromFile(b_buffer, output_n, 1, b_file_name))
            {
                if(CopyParamToBlob(b_buffer, b_blob))
                {
                    LOG(INFO)<<"initialized bias with "<<b_file_name;
                }
            }
        }
        LOG(INFO) << ("ConvolutionLayer created ")<< i;
        //  relu layers
        LayerParameter relu_param;
        relu_param.mutable_relu_param()->set_negative_slope(0.01);
        relu_layers_[i].reset(new ReLULayer<Dtype>(relu_param));
        vector<Blob<Dtype> *> temp_relu_bottom_vec_;
        vector<Blob<Dtype> *> temp_relu_top_vec_;
        temp_relu_bottom_vec_.clear();
        temp_relu_bottom_vec_.push_back(conv_out_blobs_[i].get());
        temp_relu_top_vec_.resize(1);
        temp_relu_top_vec_[0]= (i==layerN_-1) ? top[0] : relu_out_blobs_[i].get();
        
        relu_bottom_vec_[i]=temp_relu_bottom_vec_;
        relu_top_vec_[i] = temp_relu_top_vec_;
        relu_layers_[i]->SetUp(relu_bottom_vec_[i], relu_top_vec_[i]);
    }

    this->blobs_.clear();
    for(int i=0; i< layerN_; i++)
    {
        this->blobs_.insert(this->blobs_.end(), conv_layers_[i]->blobs().begin(), conv_layers_[i]->blobs().end());
    }
}

template <typename Dtype>
void PairwiseFunctionFreeformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top)
{
    for(int i=0; i< layerN_; i++)
    {
        if(i==0) conv_bottom_vec_[i][0] = bottom[0];
        if(i==layerN_-1) relu_top_vec_[i][0] = top[0];
        conv_layers_[i]->Reshape(conv_bottom_vec_[i], conv_top_vec_[i]);
        relu_layers_[i]->Reshape(relu_bottom_vec_[i], relu_top_vec_[i]);
    }

}
template <typename Dtype>
void PairwiseFunctionFreeformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
//    Blob<Dtype> * bias2 = conv_layers_[2]->blobs()[1].get();
//    std::cout<<"b2 "<< bias2->cpu_data()[0] << " "<<bias2->cpu_diff()[0] <<std::endl;
    
    conv_bottom_vec_[0][0] = bottom[0];
    relu_top_vec_[layerN_-1][0]=top[0];

    for(int i=0; i < layerN_; i++)
    {
        conv_layers_[i]->Forward(conv_bottom_vec_[i], conv_top_vec_[i]);
        relu_layers_[i]->Forward(relu_bottom_vec_[i], relu_top_vec_[i]);
    }
    
//    Dtype * top_data = top[0]->mutable_cpu_data();
//    for(int i=0; i< top[0]->count(); i++)
//    {
//        top_data[0] = 0.0;
//    }
}

template <typename Dtype>
void PairwiseFunctionFreeformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top)
{
    Forward_cpu(bottom, top);
}
    
template <typename Dtype>
void PairwiseFunctionFreeformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom)
{
    bool fix_param = this->layer_param_.multi_stage_crf_param().fix_param();
    if(fix_param){
        return;
    }
    conv_bottom_vec_[0][0] = bottom[0];
    relu_top_vec_[layerN_-1][0]=top[0];
    
    for(int i=layerN_-1; i >=0; i--)
    {
        vector<bool> relu_prop_down(1, true);
        relu_layers_[i]->Backward(relu_top_vec_[i], relu_prop_down, relu_bottom_vec_[i]);
        
        vector<bool> conv_prop_down(1, true);
        conv_layers_[i]->Backward(conv_top_vec_[i], conv_prop_down, conv_bottom_vec_[i]);
    }
}

template <typename Dtype>
void PairwiseFunctionFreeformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                      const vector<bool>& propagate_down,
                                                      const vector<Blob<Dtype>*>& bottom)
{
    Backward_cpu(top, propagate_down, bottom);
}
INSTANTIATE_CLASS(PairwiseFunctionFreeformLayer);
}  // namespace caffe
