#pragma once
#include <iostream>
#include <vector>

struct TensorSize {
    int width;
    int height;
    int depth;
};

class Tensor {
    TensorSize size;

    std::vector<double> values;

    int dw;

    void _init(int width, int height, int depth);

   public:
    Tensor(int width, int height, int depth);
    Tensor(const TensorSize& size);

    double& operator()(int depth, int i, int j);
    double operator()(int depth, int i, int j) const;

    TensorSize getSize() const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    for (int i = 0; i < tensor.size.depth; i++) {
        for (int j = 0; j < tensor.size.height; j++) {
            for (int k = 0; k < tensor.size.width; k++)
                os << tensor.values[j * tensor.dw + k * tensor.size.depth + i]
                   << " ";
            os << std::endl;
        }

        os << std::endl;
    }

    return os;
}