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

    std::vector<Tensor> filtersGradients;
    std::vector<double> biasGradients;

    int padding;
    int convStep;

    int filtersCount;
    int filterSize;
    int filterDepth;

    void _initWeights();

   public:
    ConvLayer(TensorSize size, int fc, int fs, int padding, int convStep);

    Tensor forward(const Tensor& inputTensor);
    Tensor backward(const Tensor& dout, const Tensor& inputTensor);
    void updateWeights(double learningRate);

    void setWeight(int filterIndex, int d, int i, int j, double weight);
    void setBias(int filterIndex, double bias);
};