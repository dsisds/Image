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

#ifndef DATATRANSFORMER_H_
#define DATATRANSFORMER_H_

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

#include "ThreadPool.h"
#include "Queue.h"

/**
 * This is an image processing module with OpenCV, such as
 * resizing, scaling, mirroring, substracting the image mean...
 *
 * This class has a double BlockQueue and they shared the same memory.
 * It is used to avoid create memory each time. And it also can
 * return the data even if the data are processing in multi-threads.
 */
class DataTransformer {
public:
  DataTransformer(int threadNum,
                  int capacity,
                  bool isTest,
                  bool isColor,
                  int cropHeight,
                  int cropWidth,
                  int imgSize,
                  bool isEltMean,
                  bool isChannelMean,
                  float* meanValues);
  virtual ~DataTransformer() {
    if (meanValues_) {
      free(meanValues_);
    }
    if (stdValues_) {
      free(stdValues_);
    }
    /*
    if (eigval_) {
      free(eigval_);
    }
    if (eigvec_) {
      free(eigvec_);
    }*/
  }

  /**
   * @brief Start multi-threads to transform a list of input data.
   * The processed data will be saved in Queue of prefetchFull_.
   *
   * @param data   Data containing the image string to be transformed.
   * @param label  The label of input image.
   */
  void processImgString(std::vector<std::string>& data, int* labels);

  /**
   * @brief Start multi-threads to transform a list of input data.
   * The processed data will be saved in Queue of prefetchFull_.
   *
   * @param data   Data containing the image string to be transformed.
   * @param label  The label of input image.
   */

  void processImgFile(std::vector<std::string>& data, int* labels);

  /**
   * @brief Applies the transformation on one image Mat.
   *
   * @param img    The input image Mat to be transformed.
   * @param target target is used to save the transformed data.
   */
  void transform(cv::Mat& img, float* target);

  /**
   * @brief Save image Mat as file.
   *
   * @param filename The file name.
   * @param im       The image to be saved.
   */
  void imsave(std::string filename, cv::Mat& im) { cv::imwrite(filename, im); }

  /**
   * @brief Decode the image string, then calls transform() function.
   *
   * @param src  The input image string.
   * @param size The length of string.
   * @param trg  trg is used to save the transformed data.
   */
  void transfromString(const char* src, const int size, float* trg);
  void transfromFile(std::string imgFile, float* trg);

  /**
   * @brief Return the transformed data and its label.
   */
  void obtain(float* data, int* label);

  /**
   * @brief Scales the smaller edge to size
   *
   * @param im     Input image to be resized
   * @param size   length of the shorter edge
   * 
   * @return       image resized
   */
  cv::Mat scale(cv::Mat& im, int size);

  /**
   * @brief Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
   *
   * @param im     Input image to be resized
   * @param minSize   min length of the shorter edge
   * @param maxSize   max length of the shorter edge
   * 
   * @return       image resized
   */
  cv::Mat scale(cv::Mat& im, int minSize, int maxSize);

  /**
   * @brief Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
   *
   * @param im     Input image 
   * @param size   crop size
   * 
   * @return       image resized
   */
  cv::Mat RandomSizedCrop(cv::Mat& im, int size);


  /**
   * @brief color random jitter
   *
   * @param im  Input image
   * @param 
   * @return       status
   */
  int random_jitter(cv::Mat& img, float saturation_jitter_ratio, 
                float brightness_jitter_ratio,
                float contrast_jitter_ratio);
  void saturation_jitter(cv::Mat& cv_img, float jitter_range);
  void brightness_jitter(cv::Mat& cv_img, float jitter_range);
  void contrast_jitter(cv::Mat& cv_img, float jitter_range);
  void color_normalization(cv::Mat& cv_img, float* mean, float* std);

  /**
    * @ brief crop and transform mat to buf
    * @ param  im         Input image
    * @ param  crop_size  crop size
    * @ param  h_offset   offset of height
    * @ param  w_offset   offset of width
    * @ param  flip       flip or not
    *
    * @ param  dst        output image buf
    */
  void crop(cv::Mat& im, float* dst, int crop_h, int crop_w, int h_offset, int w_offset, bool flip);

  void preprocess(cv::Mat& im, float* dst);
  cv::Mat convertTotorch(cv::Mat& im);

  void lighting(cv::Mat& im, float alphastd, const cv::Mat& eigval, const cv::Mat& eigvec);



private:
  int isTest_;
  int isColor_;
  int cropHeight_;
  int cropWidth_;
  int imgSize_;
  int capacity_;
  int fetchCount_;
  int fetchId_;
  bool isEltMean_;
  bool isChannelMean_;
  int numThreads_;
  float scale_;
  int imgPixels_;
  float* meanValues_;
  float* stdValues_;
  
  /* for random jitter */
  float brightness_jitter_ratio_;
  float saturation_jitter_ratio_;
  float contrast_jitter_ratio_;

  /* for pca */
  cv::Mat eigval_;
  cv::Mat eigvec_;


  FILE* rng_;

  /**
   * Initialize the mean values.
   */
  void loadMean(float* values);

  /**
   * @brief Generates a random integer from Uniform({min, min + 1, ..., max}).
   * @param min The lower bound (inclusive) value of the random number.
   * @param max The upper bound (inclusive) value of the random number.
   *
   * @return
   * A uniformly random integer value from ({min, min + 1, ..., max}).
   */
  int Rand(int min, int max);
  float Rand(float min, float max);
  int Rand(int max);

  typedef std::pair<float*, int> DataType;
  typedef std::shared_ptr<DataType> DataTypePtr;
  std::vector<DataTypePtr> prefetch_;
  ThreadPool threadPool_;
  // std::unique_ptr<SyncThreadPool> syncThreadPool_;
  std::vector<std::future<DataTypePtr>> results_;
  BlockingQueue<DataTypePtr> prefetchFree_;
  BlockingQueue<DataTypePtr> prefetchFull_;
};  // class DataTransformer

#endif  // DATATRANSFORMER_H_
