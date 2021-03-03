#ifndef MLP_H
#define MLP_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "matrix.hpp"
#include "graph.hpp"
using namespace ML;

/* activate method */
enum ActiveType {
    SIGMOID = 0,
    TANH,
    RELU,
    LINEAR
};
/* Optimize method */
enum OptType {
    OPT_NONE = 0,
    OPT_SGD,
    OPT_RMSPROP,
    OPT_ADAM
};
/* loss type */
enum LossType {
    MSE = 0,
    CROSS_ENTROPY
};
/* layer type */
enum LayerType {
  INPUT = 0,
  HIDDEN,
  OUTPUT
};

double sigmoid(double x)
{
    return exp(x) / (exp(x) + 1);
}
double relu(double x)
{
    return x > 0 ? x : 0;
}
double linear(double x)
{
    return x;
}
double dsigmoid(double y)
{
    return y * (1 - y);
}
double drelu(double y)
{
    return y > 0 ? 1 : 0;
}
double dtanh(double y)
{
    return 1 - y * y;
}
template <typename T>
Mat<T> LOG(Mat<T> X)
{
    return for_each(X, log);
}
template <typename T>
Mat<T> EXP(Mat<T> X)
{
    return for_each(X, exp);
}
template <typename T>
Mat<T> SQRT(Mat<T> X)
{
    return for_each(X, sqrt);
}
template <typename T>
Mat<T> SOFTMAX(Mat<T>& X)
{
    /* softmax works in multi-classify */
    T maxValue = max(X);
    Mat<T> delta = EXP(X - maxValue);
    float s = sum(delta);
    if (s != 0) {
        X = delta / s;
    }
    return X;
}

template <typename T, bool needTrain = true>
class Layer
{
public:
    std::map<int, Mat<T> > W;
    Mat<T> B;
    Mat<T> O;
    Mat<T> E;
    /* paramter */
    int layerDim;
    int inputDim;
    ActiveType activeType;
    LossType lossType;
    LayerType layerType;
    /* buffer for optimization */
    std::map<int, Mat<T> > dW;
    std::map<int, Mat<T> > Sw;
    std::map<int, Mat<T> > Vw;
    Mat<T> dB;
    Mat<T> Sb;
    Mat<T> Vb;
    double alpha1;
    double alpha2;
public:
    Layer():layerDim(0), inputDim(0), alpha1(1), alpha2(1){}
    virtual ~Layer(){}
    Layer(const Layer& layer)
    {
        W = layer.W;
        B = layer.B;
        O = layer.O;
        E = layer.E;
        /* paramter */
        layerDim = layer.layerDim;
        inputDim = layer.inputDim;
        activeType = layer.activeType;
        lossType = layer.lossType;
        layerType = layer.layerType;
        /* buffer for optimization */
        if (needTrain == true) {
            dW = layer.dW;
            Sw = layer.Sw;
            Vw = layer.Vw;
            dB = layer.dB;
            Sb = layer.Sb;
            Vb = layer.Vb;
            alpha1 = 1;
            alpha2 = 1;
        }
    }
    Layer& operator = (const Layer& layer)
    {
        if (this == &layer) {
            return *this;
        }
        W = layer.W;
        B = layer.B;
        O = layer.O;
        E = layer.E;
        /* paramter */
        layerDim = layer.layerDim;
        inputDim = layer.inputDim;
        activeType = layer.activeType;
        lossType = layer.lossType;
        layerType = layer.layerType;
        /* buffer for optimization */
        if (needTrain == true) {
            dW = layer.dW;
            Sw = layer.Sw;
            Vw = layer.Vw;
            dB = layer.dB;
            Sb = layer.Sb;
            Vb = layer.Vb;
            alpha1 = 1;
            alpha2 = 1;
        }
        return *this;
    }
    Layer(LayerType layerType,
          ActiveType activeType,
          LossType lossType,
          int layerDim)
    {
        this->layerDim = layerDim;
        this->lossType = lossType;
        this->activeType = activeType;
        this->layerType = layerType;
        this->alpha1 = 1;
        this->alpha2 = 1;
        B = Mat<T>(layerDim, 1);
        B.uniformRandom();
        O = Mat<T>(layerDim, 1);
        if (needTrain == true) {
            E = Mat<T>(layerDim, 1);
            dB = Mat<T>(layerDim, 1);
            Sb = Mat<T>(layerDim, 1);
            Vb = Mat<T>(layerDim, 1);
        }
    }

    Layer(LayerType layerType,
          ActiveType activeType,
          LossType lossType,
          int layerDim,
          int inputDim)
    {
        this->layerDim = layerDim;
        this->inputDim = inputDim;
        this->lossType = lossType;
        this->activeType = activeType;
        this->layerType = layerType;
        this->alpha1 = 1;
        this->alpha2 = 1;
        W[0] = Mat<T>(layerDim, inputDim);
        W[0].uniformRandom();
        B = Mat<T>(layerDim, 1);
        B.uniformRandom();
        O = Mat<T>(layerDim, 1);
        if (needTrain == true) {
            dW[0] = Mat<T>(layerDim, inputDim);
            Sw[0] = Mat<T>(layerDim, inputDim);
            Vw[0] = Mat<T>(layerDim, inputDim);
            E = Mat<T>(layerDim, 1);
            dB = Mat<T>(layerDim, 1);
            Sb = Mat<T>(layerDim, 1);
            Vb = Mat<T>(layerDim, 1);
        }
    }

    void connect(int from, int inputDim)
    {
        W[from] = Mat<T>(layerDim, inputDim);
        W[from].uniformRandom();
        /* buffer for optimization */
        if (needTrain == true) {
            E = Mat<T>(layerDim, 1);
            dW[from] = Mat<T>(layerDim, inputDim);
            Sw[from] = Mat<T>(layerDim, inputDim);
            Vw[from] = Mat<T>(layerDim, inputDim);
        }
    }
    Mat<T> Activate(const Mat<T> &x)
    {
        Mat<T> y;
        switch (activeType) {
            case SIGMOID:
                y = for_each(x, sigmoid);
                break;
            case RELU:
                y = for_each(x, relu);
                break;
            case TANH:
                y = for_each(x, tanh);
                break;
            case LINEAR:
                y = x;
                break;
            default:
                y = for_each(x, sigmoid);
                break;
        }
        return y;
    }

    Mat<T> dActivate(const Mat<T>& y)
    {
        Mat<T> dy;
        switch (activeType) {
            case SIGMOID:
                dy = for_each(y, dsigmoid);
                break;
            case RELU:
                dy = for_each(y, drelu);
                break;
            case TANH:
                dy = for_each(y, dtanh);
                break;
            case LINEAR:
                dy = y;
                dy.assign(1);
                break;
            default:
                dy = for_each(y, dsigmoid);
                break;
        }
        return dy;
    }
};

template <typename T, bool needTrain = true>
class MLP : public Graph<Layer<T> >
{
public:
   using DataType = T;
   using DAG = Graph<Layer<T> >;
   using InputParam = std::map<std::string, Mat<T> >;
   using InputParamVec = std::vector<InputParam>;
   using Target = std::vector<Mat<T> >;
public:
    MLP(){}
    ~MLP(){}
    void addLayer(const Layer<T> &layer, const std::string &layerName)
    {
        return DAG::insertVertex(layer, layerName);
    }

    void addLayer(LayerType layerType,
                  ActiveType activeType,
                  LossType lossType,
                  int layerDim,
                  const std::string &layerName)
    {
        return DAG::insertVertex(Layer<T, needTrain>(layerType, activeType, lossType, layerDim), layerName);
    }

    void addLayer(LayerType layerType,
                  ActiveType activeType,
                  LossType lossType,
                  int layerDim,
                  int inputDim,
                  const std::string &layerName)
    {
        return DAG::insertVertex(Layer<T, needTrain>(layerType, activeType, lossType, layerDim, inputDim), layerName);
    }

    void connectLayer(const std::string &fromName, const std::string &toName)
    {
        int from = DAG::findVertex(fromName);
        int to = DAG::findVertex(toName);
        if (from < 0 || to < 0) {
            std::cout<<"invalid name"<<std::endl;
            return;
        }
        Layer<T> &layer = DAG::getObject(to);
        Layer<T> &preLayer = DAG::getObject(from);
        layer.connect(from, preLayer.layerDim);
        return DAG::insertEdge(from, to);
    }

    void copyTo(MLP& dst)
    {
        for (int i = 0; i < DAG::vertexs.size(); i++) {
            dst.vertexs[i].object.W = DAG::vertexs[i].object.W;
            dst.vertexs[i].object.B = DAG::vertexs[i].object.B;
        }
        return;
    }

    void softUpdateTo(MLP &dst, double alpha)
    {
        for (int  current : DAG::topologySequence) {
            Layer<T> &layer = DAG::getObject(current);
            Layer<T> &dstLayer = dst.getObject(current);
            for (int  from : DAG::previous.at(current)) {
                dstLayer.W[from] = dstLayer.W[from] * (1 - alpha) + layer.W[from] * alpha;
            }
            dstLayer.B = dstLayer.B * (1 - alpha) + layer.B * alpha;
        }
        return;
    }

    void feedForward(const InputParam &x)
    {
        if (!DAG::isDAG()) {
            return;
        }
        for (int current : DAG::topologySequence) {
            Layer<T> &layer = DAG::getObject(current);
            if (layer.layerType == INPUT) {
                layer.O = layer.Activate(layer.W[0] * x.at(DAG::vertexs[current].name) + layer.B);
            } else {
                Mat<T> s(layer.O.rows, layer.O.cols);
                for (int from : DAG::previous[current]) {
                    Layer<T> &preLayer = DAG::getObject(from);    
                    s += layer.W[from] * preLayer.O;
                }
                s += layer.B;
                layer.O = layer.Activate(s);
                if (layer.lossType == CROSS_ENTROPY) {
                    layer.O = SOFTMAX(layer.O);
                }
            }
        }
        return;
    }

    void gradient(InputParam &x, Mat<T> &y)
    {
        if (!DAG::isDAG()) {
            return;
        }
        /* error backpropagate */
        for (int i = DAG::topologySequence.size() - 1; i >= 0; i--) {
            int current = DAG::topologySequence[i];
            Layer<T> &layer = DAG::getObject(current);
            if (layer.layerType == OUTPUT) {
                /* calculate loss */
                if (layer.lossType == CROSS_ENTROPY) {
                    layer.E = (y % LOG(layer.O)) * (-1);
                } else if (layer.lossType == MSE){
                    layer.E = layer.O - y;
                }
            } else {
                for (int to : DAG::nexts[current]) {
                    Layer<T> &nextLayer = DAG::getObject(to);
                    layer.E += nextLayer.W[current].Tr() * nextLayer.E;
                }
            }
        }

        /* calculate  gradient */
        for (int current : DAG::topologySequence) {
            Layer<T> &layer = DAG::getObject(current);
            if (layer.lossType == CROSS_ENTROPY) {
                Mat<T> dy = layer.O - y;
                layer.dW[0] += dy * layer.O.Tr();
                layer.dB += dy;
            } else if (layer.lossType == MSE) {
                Mat<T> dy = layer.E % layer.dActivate(layer.O);
                if (DAG::vertexs.at(current).indegree == 0) {
                    layer.dW[0] += dy * x[DAG::vertexs[current].name].Tr();
                } else {
                    for (int from : DAG::previous.at(current)) {
                        Layer<T> &preLayer = DAG::getObject(from);
                        layer.dW[from] += dy * preLayer.O.Tr();
                    }
                    layer.dB += dy;
                }
            }
        }
        return;
    }

    void SGD(double learningRate)
    {
        if (!DAG::isDAG()) {
            return;
        }
        for (int current : DAG::topologySequence) {
            Layer<T> &layer = DAG::getObject(current);
            for (int from : DAG::previous.at(current)) {
                layer.W[from] -= layer.dW[from] * learningRate;
                layer.dW[from].zero();
            }
            layer.B -= layer.dB * learningRate;
            layer.dB.zero();
        }
        return;
    }

    void RMSProp(double rho, double learningRate)
    {
        if (!DAG::isDAG()) {
            return;
        }
        for (int current : DAG::topologySequence) {
            Layer<T> &layer = DAG::getObject(current);
            for (int from : DAG::previous.at(current)) {
                layer.Sw[from] = layer.Sw[from] * rho + (layer.dW[from] % layer.dW[from]) * (1 - rho);
                layer.W[from] -= layer.dW[from] / (SQRT(layer.Sw[from]) + 1e-9) * learningRate;
                layer.dW[from].zero();
            }
            layer.Sb = layer.Sb * rho + (layer.dB % layer.dB) * (1 - rho);
            layer.B -= layer.dB / (SQRT(layer.Sb) + 1e-9)* learningRate;
            layer.dB.zero();
        }
        return;
    }

    void Adam(double alpha1, double alpha2, double learningRate)
    {
        if (!DAG::isDAG()) {
            return;
        }
        for (int current : DAG::topologySequence) {
            Layer<T> &layer = DAG::getObject(current);
            layer.alpha1 *= alpha1;
            layer.alpha2 *= alpha2;
            for (int from : DAG::previous.at(current)) {
                layer.Vw[from] = layer.Vw[from] * alpha1 + layer.dW[from] * (1 - alpha1);
                layer.Sw[from] = layer.Sw[from] * alpha2 + (layer.dW[from] % layer.dW[from]) * (1 - alpha2);
                Mat<T> Vwt = layer.Vw[from] / (1 - layer.alpha1);
                Mat<T> Swt = layer.Sw[from] / (1 - layer.alpha2);
                layer.W[from] -= Vwt / (SQRT(Swt) + 1e-9) * learningRate;
                layer.dW[from].zero();
            }
            layer.Vb = layer.Vb * alpha1 + layer.dB * (1 - alpha1);
            layer.Sb = layer.Sb * alpha2 + (layer.dB % layer.dB) * (1 - alpha2);
            Mat<T> Vbt = layer.Vb / (1 - layer.alpha1);
            Mat<T> Sbt = layer.Sb / (1 - layer.alpha2);
            layer.B -= Vbt / (SQRT(Sbt) + 1e-9) * learningRate;
            layer.dB.zero();
        }
        return;
    }

    void optimize(OptType optType, double learningRate)
    {
        switch (optType) {
        case OPT_SGD:
            SGD(learningRate);
            break;
        case OPT_RMSPROP:
            RMSProp(0.9, learningRate);
            break;
        case OPT_ADAM:
            Adam(0.9, 0.99, learningRate);
            break;
        default:
            RMSProp(0.9, learningRate);
            break;
        }
        return;
    }

    void show()
    {
        DAG::vertexs[DAG::topologySequence.size() - 1].object.O.show();
        return;
    }

    void load(const std::string& fileName)
    {
        std::ifstream file;
        file.open(fileName);
        if (!file.is_open()) {
            return;
        }
        for (int i : DAG::topologySequence) {
            Layer<T> &layer = DAG::getObject(i);
            for (int from : DAG::previous.at(i)) {
                layer.W[from].load(fileName);
            }
            layer.B.load(fileName);
        }
        file.close();
        return;
    }

    void save(const std::string& fileName)
    {
        std::ifstream file;
        file.open(fileName);
        if (!file.is_open()) {
            return;
        }
        for (int i : DAG::topologySequence) {
            Layer<T> &layer = DAG::getObject(i);
            for (int from : DAG::previous.at(i)) {
                layer.W[from].save(fileName);
            }
            layer.B.save(fileName);
        }
        file.close();
        return;
    }
};
#endif // MLP_H
