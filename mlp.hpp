#ifndef MLP_H
#define MLP_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <functional>
#include <map>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "matrix.hpp"
#include "graph.hpp"
using namespace ML;

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
/* activate method */
enum ActiveType {
    ACTIVE_SIGMOID = 0,
    ACTIVE_TANH,
    ACTIVE_RELU,
    ACTIVE_LINEAR
};

template <typename T>
class OptNull
{
public:
    OptNull(){}
    void copyFrom(const OptNull &){}
    void init(LayerType , int , int ){}
    void connect(int , int , int){}
};

template <typename T>
class OptParam
{
public:
    std::map<int, Mat<T> > dW;
    std::map<int, Mat<T> > Sw;
    std::map<int, Mat<T> > Vw;
    Mat<T> dB;
    Mat<T> Sb;
    Mat<T> Vb;
    Mat<T> E;
    double alpha1;
    double alpha2;
public:
     OptParam():alpha1(1), alpha2(1){}
     void copyFrom(const OptParam &param)
     {
         if (this == &param) {
             return;
         }
         dW = param.dW;
         Sw = param.Sw;
         Vw = param.Vw;
         dB = param.dB;
         Sb = param.Sb;
         Vb = param.Vb;
         E = param.E;
         alpha1 = param.alpha1;
         alpha2 = param.alpha2;
         return;
     }
     void init(LayerType layerType, int layerDim, int inputDim)
     {
         if (layerType == INPUT) {
             dW[0] = Mat<T>(layerDim, inputDim);
             Sw[0] = Mat<T>(layerDim, inputDim);
             Vw[0] = Mat<T>(layerDim, inputDim);
         }
         E = Mat<T>(layerDim, 1);
         dB = Mat<T>(layerDim, 1);
         Sb = Mat<T>(layerDim, 1);
         Vb = Mat<T>(layerDim, 1);
         return;
     }
     void connect(int from, int layerDim, int inputDim)
     {
         dW[from] = Mat<T>(layerDim, inputDim);
         Sw[from] = Mat<T>(layerDim, inputDim);
         Vw[from] = Mat<T>(layerDim, inputDim);
         return;
     }
};

template <typename T, bool onTrain>
class Layer:public std::conditional<onTrain, OptParam<T>, OptNull<T> >::type
{
public:
    using TOpt = typename std::conditional<onTrain, OptParam<T>, OptNull<T> >::type;
public:
    std::map<int, Mat<T> > W;
    Mat<T> B;
    Mat<T> O;
    /* paramter */
    int layerDim;
    int inputDim;
    ActiveType activeType;
    LossType lossType;
    LayerType layerType;
public:
    Layer():layerDim(0), inputDim(0){}
    virtual ~Layer(){}
    void copyFrom(const Layer& layer)
    {
        W = layer.W;
        B = layer.B;
        O = layer.O;
        /* paramter */
        layerDim = layer.layerDim;
        inputDim = layer.inputDim;
        activeType = layer.activeType;
        lossType = layer.lossType;
        layerType = layer.layerType;
        return TOpt::copyFrom(layer);
    }
    Layer(const Layer& layer)
    {
        copyFrom(layer);
    }
    Layer& operator = (const Layer& layer)
    {
        if (this == &layer) {
            return *this;
        }
        copyFrom(layer);
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
        B = Mat<T>(layerDim, 1, UNIFORM_RAND);
        O = Mat<T>(layerDim, 1);
        TOpt::init(layerType, layerDim, 1);
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
        W[0] = Mat<T>(layerDim, inputDim, UNIFORM_RAND);
        B = Mat<T>(layerDim, 1, UNIFORM_RAND);
        O = Mat<T>(layerDim, 1);
        TOpt::init(layerType, layerDim, inputDim);
    }

    void connect(int from, int inputDim)
    {
        W[from] = Mat<T>(layerDim, inputDim, UNIFORM_RAND);
        return TOpt::connect(from, layerDim, inputDim);
    }
    Mat<T> Activate(const Mat<T> &x)
    {
        Mat<T> y;
        switch (activeType) {
            case ACTIVE_SIGMOID:
                y = for_each(x, sigmoid);
                break;
            case ACTIVE_RELU:
                y = for_each(x, relu);
                break;
            case ACTIVE_TANH:
                y = for_each(x, tanh);
                break;
            case ACTIVE_LINEAR:
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
            case ACTIVE_SIGMOID:
                dy = for_each(y, dsigmoid);
                break;
            case ACTIVE_RELU:
                dy = for_each(y, drelu);
                break;
            case ACTIVE_TANH:
                dy = for_each(y, dtanh);
                break;
            case ACTIVE_LINEAR:
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

template <typename T, bool onTrain = true>
class MLP : public Graph<Layer<T, onTrain> >
{
public:
   using DataType = T;
   using DAG = Graph<Layer<T, onTrain> >;
   using Input = std::map<std::string, Mat<T> >;
   using InputVec = std::vector<Input>;
   using Target = std::vector<Mat<T> >;
   using Targets = std::map<std::string, Mat<T> >;
public:
    MLP(){}
    ~MLP(){}
    void addLayer(const Layer<T, onTrain> &layer, const std::string &layerName)
    {
        return DAG::insertVertex(layer, layerName);
    }

    void addLayer(LayerType layerType,
                  ActiveType activeType,
                  LossType lossType,
                  int layerDim,
                  const std::string &layerName)
    {
        return DAG::insertVertex(Layer<T, onTrain>(layerType, activeType, lossType, layerDim), layerName);
    }

    void addLayer(LayerType layerType,
                  ActiveType activeType,
                  LossType lossType,
                  int layerDim,
                  int inputDim,
                  const std::string &layerName)
    {
        return DAG::insertVertex(Layer<T, onTrain>(layerType, activeType, lossType, layerDim, inputDim), layerName);
    }

    void connectLayer(const std::string &fromName, const std::string &toName)
    {
        int from = DAG::findVertex(fromName);
        int to = DAG::findVertex(toName);
        if (from < 0 || to < 0) {
            std::cout<<"invalid name"<<std::endl;
            return;
        }
        auto &layer = DAG::getObject(to);
        auto &preLayer = DAG::getObject(from);
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
            auto &layer = DAG::getObject(current);
            auto &dstLayer = dst.getObject(current);
            for (int  from : DAG::previous.at(current)) {
                dstLayer.W[from] = dstLayer.W[from] * (1 - alpha) + layer.W[from] * alpha;
            }
            dstLayer.B = dstLayer.B * (1 - alpha) + layer.B * alpha;
        }
        return;
    }

    void feedForward(const Input &x)
    {
        if (!DAG::isDAG()) {
            return;
        }
        for (int current : DAG::topologySequence) {
            auto &layer = DAG::getObject(current);
            if (layer.layerType == INPUT) {
                layer.O = layer.Activate(layer.W[0] * x.at(DAG::vertexs[current].name) + layer.B);
            } else {
                Mat<T> s(layer.O.rows, layer.O.cols);
                for (int from : DAG::previous[current]) {
                    auto &preLayer = DAG::getObject(from);
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

    void gradient(Input &x, Mat<T> &y)
    {
        if (!DAG::isDAG()) {
            return;
        }
        /* error backpropagate */
        for (int i = DAG::topologySequence.size() - 1; i >= 0; i--) {
            int current = DAG::topologySequence[i];
            auto &layer = DAG::getObject(current);
            if (layer.layerType == OUTPUT) {
                /* calculate loss */
                if (layer.lossType == CROSS_ENTROPY) {
                    layer.E = (y % LOG(layer.O)) * (-1);
                } else if (layer.lossType == MSE){
                    layer.E = layer.O - y;
                }
            } else {
                for (int to : DAG::nexts[current]) {
                    auto &nextLayer = DAG::getObject(to);
                    layer.E += nextLayer.W[current].Tr() * nextLayer.E;
                }
            }
        }

        /* calculate  gradient */
        for (int current : DAG::topologySequence) {
            auto &layer = DAG::getObject(current);
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
                        auto &preLayer = DAG::getObject(from);
                        layer.dW[from] += dy * preLayer.O.Tr();
                    }

                }
                layer.dB += dy;
                layer.E.zero();
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
            auto &layer = DAG::getObject(current);
            if (layer.layerType == INPUT) {
                layer.W[0] -= layer.dW[0] * learningRate;
                layer.dW[0].zero();
            } else {
                for (int from : DAG::previous.at(current)) {
                    layer.W[from] -= layer.dW[from] * learningRate;
                    layer.dW[from].zero();
                }
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
            auto&layer = DAG::getObject(current);
            if (layer.layerType == INPUT) {
                layer.Sw[0] = layer.Sw[0] * rho + (layer.dW[0] % layer.dW[0]) * (1 - rho);
                layer.W[0] -= layer.dW[0] / (SQRT(layer.Sw[0]) + 1e-9) * learningRate;
                layer.dW[0].zero();
            } else {
                for (int from : DAG::previous.at(current)) {
                    layer.Sw[from] = layer.Sw[from] * rho + (layer.dW[from] % layer.dW[from]) * (1 - rho);
                    layer.W[from] -= layer.dW[from] / (SQRT(layer.Sw[from]) + 1e-9) * learningRate;
                    layer.dW[from].zero();
                }
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
            auto& layer = DAG::getObject(current);
            layer.alpha1 *= alpha1;
            layer.alpha2 *= alpha2;
            if (layer.layerType == INPUT) {
                layer.Vw[0] = layer.Vw[0] * alpha1 + layer.dW[0] * (1 - alpha1);
                layer.Sw[0] = layer.Sw[0] * alpha2 + (layer.dW[0] % layer.dW[0]) * (1 - alpha2);
                Mat<T> Vwt = layer.Vw[0] / (1 - layer.alpha1);
                Mat<T> Swt = layer.Sw[0] / (1 - layer.alpha2);
                layer.W[0] -= Vwt / (SQRT(Swt) + 1e-9) * learningRate;
                layer.dW[0].zero();
            } else {
                for (int from : DAG::previous.at(current)) {
                    layer.Vw[from] = layer.Vw[from] * alpha1 + layer.dW[from] * (1 - alpha1);
                    layer.Sw[from] = layer.Sw[from] * alpha2 + (layer.dW[from] % layer.dW[from]) * (1 - alpha2);
                    Mat<T> Vwt = layer.Vw[from] / (1 - layer.alpha1);
                    Mat<T> Swt = layer.Sw[from] / (1 - layer.alpha2);
                    layer.W[from] -= Vwt / (SQRT(Swt) + 1e-9) * learningRate;
                    layer.dW[from].zero();
                }
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
            auto &layer = DAG::getObject(i);
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
            auto &layer = DAG::getObject(i);
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
