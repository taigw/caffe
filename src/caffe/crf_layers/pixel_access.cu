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

namespace caffe {

template <typename Dtype>
__device__ Dtype get_gpu_pixel(const Dtype* data, int C, int H, int W, int c, int h, int w)
{
    Dtype value=0;
    if( c>=0 && c<C && h>=0 && h<H && w>=0 && w<W)
    {
        value = data[H*W*c+h*W+w];
    }
    return value;
}

template <typename Dtype>
__device__ Dtype get_gpu_pixel(const Dtype* data, int N, int C, int H, int W, int n, int c, int h, int w)
{
    Dtype value=0;
    if( n>=0 && n<N && c>=0 && c<C && h>=0 && h<H && w>=0 && w<W)
    {
        value = data[n*C*H*W + c*H*W + h*W + w];
    }
    return value;
}

template <typename Dtype>
__device__ void set_gpu_pixel(Dtype* data, int C, int H, int W, int c, int h, int w, Dtype value)
{
    if( c>=0 && c<C && h>=0 && h<H && w>=0 && w<W)
    {
        data[H*W*c+h*W+w] = value;
    }
}

template <typename Dtype>
__device__ void set_gpu_pixel(Dtype* data, int N, int C, int H, int W, int n, int c, int h, int w, Dtype value)
{
    if(n>=0 && n<N && c>=0 && c<C && h>=0 && h<H && w>=0 && w<W)
    {
        data[n*C*H*W + c*H*W + h*W + w] = value;
    }
}
}  // namespace caffe
