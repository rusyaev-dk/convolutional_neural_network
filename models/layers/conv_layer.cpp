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
    this->filtersGradients =
        std::vector<Tensor>(fc, Tensor(fs, fs, this->filterDepth));

    // adding fc zeros fir bias weights и weights gradients
    this->bias = std::vector<double>(fc, 0);
    this->biasGradients = std::vector<double>(fc, 0);

    _initWeights();
}

void ConvLayer::_initWeights() {
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

Tensor ConvLayer::forward(const Tensor& inputTensor) {
    Tensor output(outputSize);

    for (int f = 0; f < this->filtersCount; f++) {
        for (int h = 0; h < this->outputSize.height; h++) {
            for (int w = 0; w < this->outputSize.width; w++) {
                double sum = this->bias[f];  // adding bias to result sum

                // going through filters
                for (int i = 0; i < this->filterSize; i++) {
                    for (int j = 0; j < this->filterSize; j++) {
                        int i_0 = this->convStep * h + i - this->padding;
                        int j_0 = this->convStep * h + j - this->padding;

                        // Since the elements are zero outside the limits of the
                        // input tensor, just ignore them
                        bool ignore =
                            (i_0 < 0 || i_0 >= this->inputSize.height) ||
                            (j_0 < 0 || j_0 >= this->inputSize.width);
                        if (ignore) continue;

                        // Go through the entire depth of the tensor and count
                        // the sum
                        for (int d = 0; d < this->filterDepth; d++) {
                            sum += inputTensor(d, i_0, j_0) *
                                   this->filters[f](d, i_0, j_0);
                        }
                    }
                }

                output(f, h, w) = sum;
            }
        }
    }

    return output;
}

Tensor ConvLayer::backward(const Tensor& dout, const Tensor& inputTensor) {
    TensorSize deltasSize;

    deltasSize.height = this->convStep * (this->outputSize.height - 1) + 1;
    deltasSize.width = this->convStep * (this->outputSize.width - 1) + 1;
    deltasSize.depth = this->outputSize.depth;

    Tensor deltas(deltasSize);

    // Counting deltas
    for (int d = 0; d < deltasSize.depth; d++) {
        for (int i = 0; i < deltasSize.height; i++) {
            for (int j = 0; j < deltasSize.width; j++) {
                deltas(d, i * this->convStep, j * this->convStep) =
                    dout(d, i, j);
            }
        }
    }

    // Counting gradients for weigths and bias
    for (int f = 0; f < this->filtersCount; f++) {
        for (int h = 0; h < this->outputSize.height; h++) {
            for (int w = 0; w < this->outputSize.width; w++) {
                double delta = deltas(f, h, w);  // Remember gradient value

                for (int i = 0; i < this->filterSize; i++) {
                    for (int j = 0; j < this->filterSize; j++) {
                        int i_0 = i + h - this->padding;
                        int j_0 = j + w - this->padding;

                        // ignore the elements that go beyond the borders
                        bool ignore =
                            (i_0 < 0 || i_0 >= this->inputSize.height) ||
                            (j_0 < 0 || j_0 >= this->inputSize.width);
                        if (ignore) continue;

                        for (int d = 0; d < this->filterDepth; d++) {
                            this->filtersGradients[f](d, i, j) +=
                                delta * inputTensor(d, i_0, j_0);
                        }
                    }
                }

                this->biasGradients[f] += delta;
            }
        }
    }

    int pad = this->filterSize - 1 - this->padding;
    Tensor inputTensorGradients(this->inputSize);

    for (int h = 0; h < this->inputSize.height; h++) {
        for (int w = 0; w < this->inputSize.width; w++) {
            for (int d = 0; d < this->filterDepth; d++) {
                double sum = 0;

                for (int i = 0; i < this->filterSize; i++) {
                    for (int j = 0; j < this->filterSize; j++) {
                        int i_0 = h + i - pad;
                        int j_0 = w + j - pad;

                        bool ignore = (i_0 < 0 || i_0 >= deltasSize.height) ||
                                      (j_0 < 0 || j_0 >= deltasSize.width);
                        if (ignore) continue;

                        for (int f = 0; f < this->filtersCount; f++) {
                            sum +=
                                this->filters[f](d, this->filtersCount - 1 - i,
                                                 this->filtersCount - 1 - j) *
                                deltas(f, i_0,
                                       j_0);  // add the product of the turned
                                              // filters to the deltas
                        }
                    }
                }

                inputTensorGradients(d, h, w) = sum;
            }
        }
    }

    return inputTensorGradients;
}

void ConvLayer::updateWeights(double alpha) {
    for (int f = 0; f < this->filtersCount; f++) {
        for (int i = 0; i < this->filterSize; i++) {
            for (int j = 0; j < this->filterSize; j++) {
                for (int d = 0; d < this->filterDepth; d++) {
                    this->filters[f](d, i, j) -=
                        alpha * this->filtersGradients[f](d, i, j);
                    this->filtersGradients[f](d, i, j) =
                        0;  // zero the filter gradient
                }
            }
        }
        this->bias[f] -= alpha * this->biasGradients[f];
        this->bias[f] = 0;
    }
}

void ConvLayer::setWeight(int filterIndex, int d, int i, int j, double weight) {
    this->filters[filterIndex](d, i, j) = weight;
}

void ConvLayer::setBias(int filterIndex, double bias) {
    this->bias[filterIndex] = bias;
}
