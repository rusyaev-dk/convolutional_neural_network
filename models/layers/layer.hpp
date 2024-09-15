#pragma once

#include "../tensor/tensor.hpp"

class Layer {
   public:
    TensorSize inputSize;
    TensorSize outputSize;

    Layer(TensorSize inputSize, TensorSize outputSize);
    Layer(TensorSize inputSize);
};