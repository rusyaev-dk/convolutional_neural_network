#include "conv_layer.hpp"

ConvLayer::ConvLayer(TensorSize size, int fc, int fs, int padding, int convStep)
    : distribution(0.0, sqrt(2.0 / (fs * fs * size.depth))) {
    this->inputSize = size;

    this->outputSize.width = (size.width - fs + 2 * padding) / convStep + 1;
    this->outputSize.height = (size.height - fs + 2 * padding) / convStep + 1;
    this->outputSize.depth = fc;

    this->padding = padding;
    this->convStep = convStep;

    this->filtersCount = fc;
    this->filterSize = fs;
    this->filterDepth = size.depth;

    this->filters = std::vector<Tensor>(fc, Tensor(fs, fs, this->filterDepth));
    this->filtersGrad =
        std::vector<Tensor>(fc, Tensor(fs, fs, this->filterDepth));

    // добавляем fc нулей для весов смещения и их градиентов
    this->bias = std::vector<double>(fc, 0);
    this->biasGrad = std::vector<double>(fc, 0);

    InitWeights();
}

void ConvLayer::InitWeights() {
    // going through each of the filters
    for (int index = 0; index < this->filtersCount; index++) {
        for (int i = 0; i < this->filterSize; i++)
            for (int j = 0; j < this->filterSize; j++)
                for (int k = 0; k < this->filterDepth; k++)
                    this->filters[index](k, i, j) = distribution(
                        generator);  // generating a random number and write it
                                     // into the filter element

        this->bias[index] = 0.01;  // all bias are set to 0.01
    }
}

Tensor ConvLayer::Forward(const Tensor& inputTensor) {
    Tensor output(outputSize);

    for (int f = 0; f < this->filtersCount; f++) {
        }
}