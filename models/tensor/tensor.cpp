#include "tensor.hpp"

void Tensor::Init(int width, int height, int depth) {
  this->size.width = width;
  this->size.height = height;
  this->size.depth = depth;

  this->dw = depth * width;
  this->values = std::vector<double>(width * height * depth, 0);
}

Tensor::Tensor(int width, int height, int depth) {
  this->Init(width, height, depth);
}

Tensor::Tensor(const TensorSize& size) {
  this->Init(size.width, size.height, size.depth);
}

double& Tensor::operator()(int i, int j, int k) {
  return this->values[i * this->dw + j * size.depth + k];
}

double Tensor::operator()(int i, int j, int k) const {
  return this->values[i * this->dw + j * size.depth + k];
}

TensorSize Tensor::getSize() const { return this->size; }
