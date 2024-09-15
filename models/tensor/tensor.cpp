#include "tensor.hpp"

void Tensor::_init(int width, int height, int depth) {
    this->size.width = width;
    this->size.height = height;
    this->size.depth = depth;

    this->dw = depth * width;
    this->values = std::vector<double>(width * height * depth, 0);
}

Tensor::Tensor(int width, int height, int depth) {
    this->_init(width, height, depth);
}

Tensor::Tensor(const TensorSize& size) {
    this->_init(size.width, size.height, size.depth);
}

double& Tensor::operator()(int depth, int i, int j) {
    return this->values[depth * this->dw + i * size.depth + j];
}

double Tensor::operator()(int depth, int i, int j) const {
    return this->values[depth * this->dw + i * size.depth + j];
}

TensorSize Tensor::getSize() const { return this->size; }
