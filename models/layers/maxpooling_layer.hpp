#pragma once

#include "layer.hpp"

class MaxPoolingLayer : Layer {
    static const int _defaultScale = 2;
    int _scale;

    Tensor _mask;

   public:
    MaxPoolingLayer(TensorSize inputSize, int scale = _defaultScale);

    Tensor forward(const Tensor& inputTensor);
    Tensor backward(const Tensor& dout, const Tensor& inputTensor);
};