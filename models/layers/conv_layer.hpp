#pragma once
#include <random>
#include <vector>

#include "../tensor/tensor.hpp"

/// Convolutional layer
class ConvLayer {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    TensorSize inputSize;
    TensorSize outputSize;

    std::vector<Tensor> filters;  // weights
    std::vector<double> bias;     // weights

    std::vector<Tensor> filtersGrad;
    std::vector<double> biasGrad;

    int padding;
    int convStep;

    int filtersCount;
    int filterSize;
    int filterDepth;

    void InitWeights();

   public:
    ConvLayer(TensorSize size, int fc, int fs, int padding, int convStep);

    Tensor Forward(const Tensor& inputTensor);
    Tensor Backward(const Tensor& dout, const Tensor& inputTensor);
    void UpdateWeights(double learningRate);
};