#include "maxpooling_layer.hpp"

MaxPoolingLayer::MaxPoolingLayer(TensorSize inputSize,
                                 int scale = _defaultScale)
    : Layer(inputSize), _mask(Tensor(inputSize)) {
    this->outputSize.width = inputSize.width / scale;
    this->outputSize.height = inputSize.height / scale;
    this->outputSize.depth = inputSize.depth;

    this->_scale = scale;
}

Tensor MaxPoolingLayer::forward(const Tensor& inputTensor) {
    Tensor output(this->outputSize);

    for (int d = 0; d < this->inputSize.depth; d++) {
        for (int i = 0; i < this->inputSize.height; i++) {
            for (int j = 0; j < this->inputSize.width; j++) {
                double max = inputTensor(d, i, j);

                for (int subMatrixI = i; subMatrixI < i + _scale;
                     subMatrixI++) {
                    for (int subMatrixJ = j; subMatrixJ < j + _scale;
                         subMatrixJ++) {
                        double value = inputTensor(d, subMatrixI, subMatrixJ);

                        if (value > max) max = value;
                    }
                }

                output(d, i / _scale, j / _scale) = max;
            }
        }
    }

    return output;
}

Tensor MaxPoolingLayer::backward(const Tensor& dout,
                                 const Tensor& inputTensor) {}