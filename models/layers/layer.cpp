#include "layer.hpp"

Layer::Layer(TensorSize inputSize, TensorSize outputSize) {
    this->inputSize = inputSize;
    this->outputSize = outputSize;
}

Layer::Layer(TensorSize inputSize) {
    this->inputSize = inputSize;
    this->outputSize = inputSize;
}