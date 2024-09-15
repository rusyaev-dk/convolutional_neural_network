#include "conv_layer.hpp"

ConvLayer::ConvLayer(TensorSize inputSize, int fc, int fs, int padding,
                     int convStep)
    : Layer(inputSize),
      _distribution(0.0, sqrt(2.0 / (fs * fs * inputSize.depth))) {
    this->inputSize = inputSize;

    this->outputSize.width =
        (inputSize.width - fs + 2 * padding) / convStep + 1;
    this->outputSize.height =
        (inputSize.height - fs + 2 * padding) / convStep + 1;
    this->outputSize.depth = fc;

    this->_padding = padding;
    this->_convStep = convStep;

    this->_filtersCount = fc;
    this->_filterSize = fs;
    this->_filterDepth = inputSize.depth;

    this->_filters =
        std::vector<Tensor>(fc, Tensor(fs, fs, this->_filterDepth));
    this->_filtersGradients =
        std::vector<Tensor>(fc, Tensor(fs, fs, this->_filterDepth));

    // Add fc zeros fir bias weights и weights gradients
    this->_bias = std::vector<double>(fc, 0);
    this->_biasGradients = std::vector<double>(fc, 0);

    _initWeights();
}

void ConvLayer::_initWeights() {
    // Go through each of the filters
    for (int index = 0; index < this->_filtersCount; index++) {
        for (int i = 0; i < this->_filterSize; i++)
            for (int j = 0; j < this->_filterSize; j++)
                for (int k = 0; k < this->_filterDepth; k++)
                    this->_filters[index](k, i, j) = _distribution(
                        _generator);  // Generate a random number and write it
                                      // into the filter element

        this->_bias[index] = 0.01;  // All bias are set to 0.01
    }
}

Tensor ConvLayer::forward(const Tensor& inputTensor) {
    Tensor output(outputSize);

    for (int f = 0; f < this->_filtersCount; f++) {
        for (int h = 0; h < this->outputSize.height; h++) {
            for (int w = 0; w < this->outputSize.width; w++) {
                double sum = this->_bias[f];  // adding bias to result sum

                // Go through filters
                for (int i = 0; i < this->_filterSize; i++) {
                    for (int j = 0; j < this->_filterSize; j++) {
                        int i_0 = this->_convStep * h + i - this->_padding;
                        int j_0 = this->_convStep * h + j - this->_padding;

                        // Since the elements are zero outside the limits of the
                        // input tensor, just ignore them
                        bool ignore =
                            (i_0 < 0 || i_0 >= this->inputSize.height) ||
                            (j_0 < 0 || j_0 >= this->inputSize.width);
                        if (ignore) continue;

                        // Go through the entire depth of the tensor and count
                        // the sum
                        for (int d = 0; d < this->_filterDepth; d++) {
                            sum += inputTensor(d, i_0, j_0) *
                                   this->_filters[f](d, i_0, j_0);
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

    deltasSize.height = this->_convStep * (this->outputSize.height - 1) + 1;
    deltasSize.width = this->_convStep * (this->outputSize.width - 1) + 1;
    deltasSize.depth = this->outputSize.depth;

    Tensor deltas(deltasSize);

    // Count deltas
    for (int d = 0; d < deltasSize.depth; d++) {
        for (int i = 0; i < deltasSize.height; i++) {
            for (int j = 0; j < deltasSize.width; j++) {
                deltas(d, i * this->_convStep, j * this->_convStep) =
                    dout(d, i, j);
            }
        }
    }

    // Count gradients for weigths and bias
    for (int f = 0; f < this->_filtersCount; f++) {
        for (int h = 0; h < this->outputSize.height; h++) {
            for (int w = 0; w < this->outputSize.width; w++) {
                double delta = deltas(f, h, w);  // Remember gradient value

                for (int i = 0; i < this->_filterSize; i++) {
                    for (int j = 0; j < this->_filterSize; j++) {
                        int i_0 = i + h - this->_padding;
                        int j_0 = j + w - this->_padding;

                        // Ignore the elements that go beyond the borders
                        bool ignore =
                            (i_0 < 0 || i_0 >= this->inputSize.height) ||
                            (j_0 < 0 || j_0 >= this->inputSize.width);
                        if (ignore) continue;

                        for (int d = 0; d < this->_filterDepth; d++) {
                            this->_filtersGradients[f](d, i, j) +=
                                delta * inputTensor(d, i_0, j_0);
                        }
                    }
                }

                this->_biasGradients[f] += delta;
            }
        }
    }

    int pad = this->_filterSize - 1 - this->_padding;
    Tensor inputTensorGradients(this->inputSize);

    for (int h = 0; h < this->inputSize.height; h++) {
        for (int w = 0; w < this->inputSize.width; w++) {
            for (int d = 0; d < this->_filterDepth; d++) {
                double sum = 0;

                for (int i = 0; i < this->_filterSize; i++) {
                    for (int j = 0; j < this->_filterSize; j++) {
                        int i_0 = h + i - pad;
                        int j_0 = w + j - pad;

                        bool ignore = (i_0 < 0 || i_0 >= deltasSize.height) ||
                                      (j_0 < 0 || j_0 >= deltasSize.width);
                        if (ignore) continue;

                        for (int f = 0; f < this->_filtersCount; f++) {
                            sum += this->_filters[f](
                                       d, this->_filtersCount - 1 - i,
                                       this->_filtersCount - 1 - j) *
                                   deltas(f, i_0,
                                          j_0);  // add the product of the
                                                 // turned filters to the deltas
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
    for (int f = 0; f < this->_filtersCount; f++) {
        for (int i = 0; i < this->_filterSize; i++) {
            for (int j = 0; j < this->_filterSize; j++) {
                for (int d = 0; d < this->_filterDepth; d++) {
                    this->_filters[f](d, i, j) -=
                        alpha * this->_filtersGradients[f](d, i, j);
                    this->_filtersGradients[f](d, i, j) =
                        0;  // zero the filter gradient
                }
            }
        }
        this->_bias[f] -= alpha * this->_biasGradients[f];
        this->_bias[f] = 0;
    }
}

void ConvLayer::setWeight(int filterIndex, int d, int i, int j, double weight) {
    this->_filters[filterIndex](d, i, j) = weight;
}

void ConvLayer::setBias(int filterIndex, double bias) {
    this->_bias[filterIndex] = bias;
}
