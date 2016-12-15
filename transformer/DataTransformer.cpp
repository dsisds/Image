/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <time.h>
#include <limits>
#include <glog/logging.h>
#include <chrono>
#include <math.h>

#include "DataTransformer.h"

DataTransformer::DataTransformer(int threadNum,
                                 int capacity,
                                 bool isTest,
                                 bool isColor,
                                 int cropHeight,
                                 int cropWidth,
                                 int imgSize,
                                 bool isEltMean,
                                 bool isChannelMean,
                                 float* meanValues)
    : isTest_(isTest),
      isColor_(isColor),
      cropHeight_(cropHeight),
      cropWidth_(cropWidth),
      imgSize_(imgSize),
      capacity_(capacity),
      threadPool_(threadNum),
      prefetchFree_(capacity),
      prefetchFull_(capacity) {
  fetchCount_ = -1;
  scale_ = 1.0;
  isChannelMean_ = isChannelMean;
  isEltMean_ = isEltMean;
  meanValues_ = NULL;
  loadMean(meanValues);
  stdValues_ = NULL;

  imgPixels_ = cropHeight * cropWidth * (isColor_ ? 3 : 1);

  prefetch_.reserve(capacity);
  for (int i = 0; i < capacity; i++) {
    auto d = std::make_shared<DataType>(new float[imgPixels_ * 3], 0);
    prefetch_.push_back(d);
    memset(prefetch_[i]->first, 0, imgPixels_ * sizeof(float));
    prefetchFree_.enqueue(prefetch_[i]);
  }
  numThreads_ = threadNum;
  rng_ = fopen("/dev/urandom", "rb");

  // init default
  brightness_jitter_ratio_ = 0.4;
  saturation_jitter_ratio_ = 0.4;
  contrast_jitter_ratio_ = 0.4;
  if (!stdValues_) {
    stdValues_ = new float[3];
    stdValues_[0] = 0.229;
    stdValues_[1] = 0.224;
    stdValues_[2] = 0.225;
  }
  if (!meanValues_) {
    meanValues_ = new float[3];
    meanValues_[0] = 0.485;
    meanValues_[1] = 0.456;
    meanValues_[2] = 0.406;
  }
  float eigvec_tmp[3 * 3] = {-0.5675, 0.7192, 0.4009,
                             -0.5808, -0.0045, -0.8140,
                             -0.5836, -0.6948, 0.4203};
  float eigval_tmp[3] = {0.2175, 0.0188, 0.0045};
  eigval_ = cv::Mat(3, 1, CV_32FC1, eigval_tmp);
  eigvec_ = cv::Mat(3, 3, CV_32FC1, eigvec_tmp);
}

void DataTransformer::loadMean(float* values) {
  if (values) {
    int c = isColor_ ? 3 : 1;
    int sz = isChannelMean_ ? c : cropHeight_ * cropWidth_ * c;
    meanValues_ = new float[sz];
    memcpy(meanValues_, values, sz * sizeof(float));
  }
}

void DataTransformer::transfromFile(std::string imgFile, float* trg) {
  int cvFlag = (isColor_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
  try {
    cv::Mat im = cv::imread(imgFile, cvFlag);
    if (!im.data) {
      LOG(ERROR) << "Could not decode image";
      LOG(ERROR) << im.channels() << " " << im.rows << " " << im.cols;
    }
    this->transform(im, trg);
  } catch (cv::Exception& e) {
    LOG(ERROR) << "Caught exception in cv::imdecode " << e.msg;
  }
}

void DataTransformer::transfromString(const char* src,
                                      const int size,
                                      float* trg) {
  try {
    cv::_InputArray imbuf(src, size);
    int cvFlag = (isColor_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat im = cv::imdecode(imbuf, cvFlag);
    if (!im.data) {
      LOG(ERROR) << "Could not decode image";
      LOG(ERROR) << im.channels() << " " << im.rows << " " << im.cols;
    }
    this->transform(im, trg);
  } catch (cv::Exception& e) {
    LOG(ERROR) << "Caught exception in cv::imdecode " << e.msg;
  }
}

int DataTransformer::Rand(int min, int max) {
  std::random_device source;
  std::mt19937 rng(source());
  std::uniform_int_distribution<int> dist(min, max);
  return dist(rng);
}

int DataTransformer::Rand(int n) {
    unsigned int d;
    fread((char*)&d, sizeof(unsigned int), 1, rng_);
    return d % n;
}

float DataTransformer::Rand(float min_x, float max_x) {
    if(min_x > max_x){
        LOG(ERROR) << "min_x:" << min_x << " max_x:" << max_x;
    }
    if (min_x == max_x){
        return min_x;
    }
    return min_x + static_cast <float> (Rand(10000)) / 
      (static_cast <float> (10000.0/(max_x - min_x)));
}

cv::Mat DataTransformer::scale(cv::Mat& im, int size) {
    int h = im.rows;
    int w = im.cols;
    if (h > w) {
        h = int(float(h) / w * size);
        w = size;
    } else {
        w = int(float(w) / h * size);
        h = size;
    }
    cv::Mat im_ret;
    cv::resize(im, im_ret, cv::Size(w, h));
    return im_ret;
}

cv::Mat DataTransformer::scale(cv::Mat& im, int minSize, int maxSize) {
    int h = im.rows;
    int w = im.cols;

    int tsize = Rand(minSize, maxSize);
    return scale(im, tsize);
}

cv::Mat DataTransformer::RandomSizedCrop(cv::Mat& im, int size) {
    const int imgChannels = im.channels();
    const int imgHeight = im.rows;
    const int imgWidth = im.cols;
    int attempt = 0;
    cv::Mat cv_cropped_img = im;
    while (attempt < 10) {
        int area = imgHeight * imgWidth;
        float targetArea = Rand(float(0.08), float(1.0)) * area;

        float aspectRatio = Rand(0.75, float(4) / 3);
        int w = int(sqrt(targetArea * aspectRatio));
        int h = int(sqrt(targetArea / aspectRatio));

        if (Rand(2)) {
            int tmp = h;
            h = w;
            w = tmp;
        }
        if (h < imgHeight && w < imgWidth) {
            int h1 = Rand(imgHeight - h);
            int w1 = Rand(imgWidth - w);

            cv::Rect roi(w1, h1, h, w);
            cv_cropped_img = im(roi);
            break;
        }
        attempt++;
    }
    cv::resize(cv_cropped_img, cv_cropped_img, cv::Size(size, size));
    return cv_cropped_img; 
}

int DataTransformer::random_jitter(cv::Mat& cv_img, float saturation_range, float brightness_range,
    float contrast_range) {
    float saturation_ratio = Rand(-saturation_range, saturation_range);
    float brightness_ratio = Rand(-brightness_range, brightness_range);
    float contrast_ratio = Rand(-contrast_range, contrast_range);
    std::vector<int> order(3);
    for(int i = 0; i < 3; i++) {
        order[i] = i;
    }
    std::random_shuffle(order.begin(), order.end());
    for(int i = 0; i < 3; i++) {
        if (order[i] == 0) {
            saturation_jitter(cv_img, saturation_ratio);
        }
        if (order[i] == 1) {
            brightness_jitter(cv_img, brightness_ratio);
        }
        if (order[i] == 2) {
            contrast_jitter(cv_img, contrast_ratio);
        }
    }
    return 0;
}

void DataTransformer::saturation_jitter(cv::Mat& cv_img, float jitter_range) {
    cv::Mat greyMat;
    cv::cvtColor(cv_img, greyMat, CV_BGR2GRAY);
    for(int h = 0; h < cv_img.rows; h++)
       for(int w = 0; w < cv_img.cols; w++)
           for(int c = 0; c < 3; c++)
               cv_img.at<cv::Vec3b>(h,w)[c] = cv::saturate_cast<uchar>((1.0 - jitter_range)*(cv_img.at<cv::Vec3b>(h,w)[c]) + jitter_range * greyMat.at<uchar>(h,w));
    return;
}

void DataTransformer::brightness_jitter(cv::Mat& cv_img, float jitter_range) {
    cv_img.convertTo(cv_img, -1, 1.0 - jitter_range, 0);
    return;
}

void DataTransformer::contrast_jitter(cv::Mat& cv_img, float jitter_range) {
    cv::Mat greyMat;
    cv::cvtColor(cv_img, greyMat, CV_BGR2GRAY);
    cv::Scalar mean = cv::mean(greyMat);
    cv_img.convertTo(cv_img, -1, 1.0 - jitter_range, jitter_range * mean[0]); 
    return;
}

void DataTransformer::crop(cv::Mat& im, float* dst, int crop_h, int crop_w, int h_offset, int w_offset, bool flip) {
  int height = im.rows;
  int width = im.cols;
  int top_index;
  for (int h = 0; h < crop_h; ++h) {
    const uchar* ptr = im.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < crop_w; ++w) {
      for (int c = 0; c < im.channels(); ++c) {
        if (flip) {
          top_index = (c * height + h) * width + width - 1 - w;
        } else {
          top_index = (c * height + h) * width + w;
        }
        float pixel = static_cast<float>(ptr[img_index++]);
        dst[top_index] = pixel;
      }
    }
  }
}

void DataTransformer::color_normalization(cv::Mat& cv_img, float* mean, float* std) {
  std::vector<cv::Mat> ch;
  cv::split(cv_img, ch);
  for (int i = 0; i < ch.size(); i++) {
    ch[i] -= mean[i];
    ch[i] /= std[i];
  }
  cv::merge(ch, cv_img);
}

cv::Mat DataTransformer::convertTotorch(cv::Mat& im) {
  cv::Mat ret;
  im.convertTo(ret, CV_32FC3);
  ret /= 255.0;
  return ret;
}

void DataTransformer::lighting(cv::Mat& im, float alphastd, const cv::Mat& eigval, const cv::Mat& eigvec) {
  if (alphastd == 0) {
    return;
  }
  cv::Mat alpha_t(1, 3, CV_32FC1);
  cv::RNG rnger(cv::getTickCount());
  rnger.fill(alpha_t, cv::RNG::UNIFORM, cv::Scalar::all(0.), cv::Scalar::all(alphastd));
  cv::Mat alpha(3, 3, CV_32FC1);
  for (int i = 0; i < 3; i++) {
    alpha_t.copyTo(alpha.row(i));
  }
  alpha = alpha.mul(eigvec);
  /* alpha: 3 * 3    eigval: 3 * 1 */
  alpha = alpha * eigval;
  std::vector<cv::Mat> ch;
  cv::split(im, ch);
  for (int i = 0; i < 3; i++) {
    ch[i] += alpha.at<float>(i, 0);
  }
  cv::merge(ch, im);
}

void DataTransformer::preprocess(cv::Mat& cvImgOri, float* dst) {
  const int imgChannels = cvImgOri.channels();
  const int imgHeight = cvImgOri.rows;
  const int imgWidth = cvImgOri.cols;
  const bool doMirror = (!isTest_) && Rand(0, 1);

  if (!isTest_) {
    cv::Mat cv_cropped_img = RandomSizedCrop(cvImgOri, cropHeight_);
    random_jitter(cv_cropped_img, saturation_jitter_ratio_, brightness_jitter_ratio_, 
      contrast_jitter_ratio_);
    cv_cropped_img = convertTotorch(cv_cropped_img);
    lighting(cv_cropped_img, 0.1, eigval_, eigvec_);
    color_normalization(cv_cropped_img, meanValues_, stdValues_);
    crop(cv_cropped_img, dst, cropHeight_, cropHeight_, 0, 0, doMirror);
  } else {
    cv::Mat cv_cropped_img;
    if (imgSize_ > 0) {
      cv_cropped_img = scale(cvImgOri, imgSize_);
      cv_cropped_img = convertTotorch(cv_cropped_img);
      color_normalization(cv_cropped_img, meanValues_, stdValues_);
      crop(cv_cropped_img, dst, cropHeight_, cropWidth_, (cv_cropped_img.rows - cropHeight_) / 2, 
        (cv_cropped_img.cols - cropWidth_) / 2, false);
    } 
  }
}



void DataTransformer::transform(cv::Mat& cvImgOri, float* target) {
  const int imgChannels = cvImgOri.channels();
  const int imgHeight = cvImgOri.rows;
  const int imgWidth = cvImgOri.cols;
  const bool doMirror = (!isTest_) && Rand(0, 1);
  int h_off = 0;
  int w_off = 0;
  int th = imgHeight;
  int tw = imgWidth;
  cv::Mat img;
  if (imgSize_ > 0) {
    if (imgHeight > imgWidth) {
      tw = imgSize_;
      th = int(double(imgHeight) / imgWidth * tw);
      th = th > imgSize_ ? th : imgSize_;
    } else {
      th = imgSize_;
      tw = int(double(imgWidth) / imgHeight * th);
      tw = tw > imgSize_ ? tw : imgSize_;
    }
    cv::resize(cvImgOri, img, cv::Size(tw, th));
  } else {
    cv::Mat img = cvImgOri;
  }

  cv::Mat cv_cropped_img = img;
  if (cropHeight_ && cropWidth_) {
    if (!isTest_) {
      h_off = Rand(0, th - cropHeight_);
      w_off = Rand(0, tw - cropWidth_);
    } else {
      h_off = (th - cropHeight_) / 2;
      w_off = (tw - cropWidth_) / 2;
    }
    cv::Rect roi(w_off, h_off, cropWidth_, cropHeight_);
    cv_cropped_img = img(roi);
  } else {
    CHECK_EQ(cropHeight_, imgHeight);
    CHECK_EQ(cropWidth_, imgWidth);
  }
  int height = cropHeight_;
  int width = cropWidth_;
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < imgChannels; ++c) {
        if (doMirror) {
          top_index = (c * height + h) * width + width - 1 - w;
        } else {
          top_index = (c * height + h) * width + w;
        }
        float pixel = static_cast<float>(ptr[img_index++]);
        if (isEltMean_) {
          int mean_index = (c * imgHeight + h) * imgWidth + w;
          target[top_index] = (pixel - meanValues_[mean_index]) * scale_;
        } else {
          if (isChannelMean_) {
            target[top_index] = (pixel - meanValues_[c]) * scale_;
          } else {
            target[top_index] = pixel * scale_;
          }
        }
      }
    }
  }  // target: BGR
}

void DataTransformer::processImgString(std::vector<std::string>& data,
                                       int* labels) {
  results_.clear();
  for (size_t i = 0; i < data.size(); ++i) {
    results_.emplace_back(threadPool_.enqueue([this, &data, labels, i]() {
      DataTypePtr ret = this->prefetchFree_.dequeue();
      std::string buf = data[i];
      int size = buf.length();
      ret->second = labels[i];
      this->transfromString(buf.c_str(), size, ret->first);
      return ret;
    }));
  }
  fetchCount_ = data.size();
  fetchId_ = 0;
}

void DataTransformer::processImgFile(std::vector<std::string>& data,
                                     int* labels) {
  results_.clear();
  for (size_t i = 0; i < data.size(); ++i) {
    results_.emplace_back(threadPool_.enqueue([this, &data, labels, i]() {
      DataTypePtr ret = this->prefetchFree_.dequeue();
      std::string file = data[i];
      ret->second = labels[i];
      this->transfromFile(file, ret->first);
      return ret;
    }));
  }
  fetchCount_ = data.size();
  fetchId_ = 0;
}

void DataTransformer::obtain(float* data, int* label) {
  if (fetchId_ >= fetchCount_) {
    LOG(FATAL) << "Empty data";
  }
  DataTypePtr ret = results_[fetchId_].get();
  *label = ret->second;
  memcpy(data, ret->first, sizeof(float) * imgPixels_);
  prefetchFree_.enqueue(ret);
  ++fetchId_;
}
