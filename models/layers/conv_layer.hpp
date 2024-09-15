#pragma once
#include <random>
#include <vector>

#include "layer.hpp"

/// Convolutional layer
class ConvLayer : Layer {
    std::default_random_engine _generator;
    std::normal_distribution<double> _distribution;

    std::vector<Tensor> _filters;  // weights
    std::vector<double> _bias;     // weights

    std::vector<Tensor> _filtersGradients;
    std::vector<double> _biasGradients;

    int _padding;
    int _convStep;

    int _filtersCount;
    int _filterSize;
    int _filterDepth;

    void _initWeights();

   public:
    ConvLayer(TensorSize inputSize, int fc, int fs, int padding, int convStep);

    Tensor forward(const Tensor& inputTensor);
    Tensor backward(const Tensor& dout, const Tensor& inputTensor);
    void updateWeights(double learningRate);

    void setWeight(int filterIndex, int d, int i, int j, double weight);
    void setBias(int filterIndex, double bias);
};