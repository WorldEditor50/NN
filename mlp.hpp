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

template <typename T>
class NoneOpt
{
public:
    NoneOpt(){}
    NoneOpt(const NoneOpt &){}
    NoneOpt& operator=(const NoneOpt &){return *this;}
    NoneOpt(LayerType , int , int ){}
    void connect(int , int , int){}
    void _(const std::vector<int> &,
           LayerType ,
           T ,
           std::map<int, Mat<T> > &,
           Mat<T> &){}
};

template <typename T>
class SGD
{
public:
   std::map<int, Mat<T> > dW;
   Mat<T> dB;
   Mat<T> E;
public:
    SGD(){}
    SGD(const SGD &sgd)
    {
        dW = sgd.dW;
        dB = sgd.dB;
        E = sgd.E;
        return;
    }
    SGD& operator = (const SGD &sgd)
    {
        if (this == &sgd) {
            return *this;
        }
        dW = sgd.dW;
        dB = sgd.dB;
        E = sgd.E;
        return *this;
    }
    SGD(LayerType layerType, int layerDim, int inputDim)
    {
        if (layerType == INPUT) {
            dW[0] = Mat<T>(layerDim, inputDim);
        }
        E = Mat<T>(layerDim, 1);
        dB = Mat<T>(layerDim, 1);
    }
    void connect(int from, int layerDim, int inputDim)
    {
        dW[from] = Mat<T>(layerDim, inputDim);
        return;
    }
    void _(const std::vector<int> &previous,
           LayerType layerType,
           T learningRate,
           std::map<int, Mat<T> > &W,
           Mat<T> &B)
    {
        if (layerType == INPUT) {
            W[0] -= dW[0] * learningRate;
            dW[0].zero();
        } else {
            for (int from : previous) {
                W[from] -= dW[from] * learningRate;
                dW[from].zero();
            }
        }
        B -= dB * learningRate;
        dB.zero();
        return;
    }
};

template <typename T>
class RMSProp
{
public:
     static T rho;
public:
    std::map<int, Mat<T> > dW;
    Mat<T> dB;
    Mat<T> E;
    std::map<int, Mat<T> > Sw;
    Mat<T> Sb;
public:
    RMSProp(){}
    RMSProp(const RMSProp &rmsprop)
    {
        dW = rmsprop.dW;
        dB = rmsprop.dB;
        E = rmsprop.E;
        Sw = rmsprop.Sw;
        Sb = rmsprop.Sb;
        rho = rmsprop.rho;
        return;
    }
    RMSProp& operator=(const RMSProp &rmsprop)
    {
        if (this == &rmsprop) {
            return *this;
        }
        dW = rmsprop.dW;
        dB = rmsprop.dB;
        E = rmsprop.E;
        Sw = rmsprop.Sw;
        Sb = rmsprop.Sb;
        rho = rmsprop.rho;
        return *this;
    }
    RMSProp(LayerType layerType, int layerDim, int inputDim)
    {
        if (layerType == INPUT) {
            dW[0] = Mat<T>(layerDim, inputDim);
            Sw[0] = Mat<T>(layerDim, inputDim);
        }
        E = Mat<T>(layerDim, 1);
        dB = Mat<T>(layerDim, 1);
        Sb = Mat<T>(layerDim, 1);
        return;
    }
    void connect(int from, int layerDim, int inputDim)
    {
        dW[from] = Mat<T>(layerDim, inputDim);
        Sw[from] = Mat<T>(layerDim, inputDim);
        return;
    }
    void _(const std::vector<int> &previous,
           LayerType layerType,
           T learningRate,
           std::map<int, Mat<T> > &W,
           Mat<T> &B)
    {
        if (layerType == INPUT) {
            Sw[0] = Sw[0] * rho + (dW[0] % dW[0]) * (1 - rho);
            W[0] -= dW[0] / (SQRT(Sw[0]) + 1e-9) * learningRate;
            dW[0].zero();
        } else {
            for (int from : previous) {
                Sw[from] = Sw[from] * rho + (dW[from] % dW[from]) * (1 - rho);
                W[from] -= dW[from] / (SQRT(Sw[from]) + 1e-9) * learningRate;
                dW[from].zero();
            }
        }
        Sb = Sb * rho + (dB % dB) * (1 - rho);
        B -= dB / (SQRT(Sb) + 1e-9)* learningRate;
        dB.zero();
        return;
    }
};
template<typename T>
T RMSProp<T>::rho(0.9);

template <typename T>
class Adam
{
public:
    static T alpha1Factor;
    static T alpha2Factor;
public:
    std::map<int, Mat<T> > dW;
    Mat<T> dB;
    Mat<T> E;
    std::map<int, Mat<T> > Sw;
    Mat<T> Sb;
    std::map<int, Mat<T> > Vw;
    Mat<T> Vb;
    T alpha1;
    T alpha2;
public:
    Adam():alpha1(1), alpha2(1){}
    Adam(const Adam &adam)
    {
        dW = adam.dW;
        dB = adam.dB;
        E = adam.E;
        Sw = adam.Vw;
        Sb = adam.Vb;
        Vw = adam.Vw;
        Vb = adam.Vb;
        alpha1 = adam.alpha1;
        alpha2 = adam.alpha2;
        return;
    }
    Adam& operator=(const Adam &adam)
    {
        if (this == &adam) {
            return *this;
        }
        dW = adam.dW;
        dB = adam.dB;
        E = adam.E;
        Sw = adam.Vw;
        Sb = adam.Vb;
        Vw = adam.Vw;
        Vb = adam.Vb;
        alpha1 = adam.alpha1;
        alpha2 = adam.alpha2;
        return *this;
    }
    Adam(LayerType layerType, int layerDim, int inputDim):alpha1(1), alpha2(1)
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

    void _(const std::vector<int> &previous,
           LayerType layerType,
           T learningRate,
           std::map<int, Mat<T> > &W,
           Mat<T> &B)
    {
        alpha1 *= alpha1Factor;
        alpha2 *= alpha2Factor;
        if (layerType == INPUT) {
            Vw[0] = Vw[0] * alpha1Factor + dW[0] * (1 - alpha1Factor);
            Sw[0] = Sw[0] * alpha2Factor + (dW[0] % dW[0]) * (1 - alpha2Factor);
            Mat<T> Vwt = Vw[0] / (1 - alpha1);
            Mat<T> Swt = Sw[0] / (1 - alpha2);
            W[0] -= Vwt / (SQRT(Swt) + 1e-9) * learningRate;
            dW[0].zero();
        } else {
            for (int from : previous) {
                Vw[from] = Vw[from] * alpha1Factor + dW[from] * (1 - alpha1Factor);
                Sw[from] = Sw[from] * alpha2Factor + (dW[from] % dW[from]) * (1 - alpha2Factor);
                Mat<T> Vwt = Vw[from] / (1 - alpha1);
                Mat<T> Swt = Sw[from] / (1 - alpha2);
                W[from] -= Vwt / (SQRT(Swt) + 1e-9) * learningRate;
                dW[from].zero();
            }
        }
        Vb = Vb * alpha1Factor + dB * (1 - alpha1Factor);
        Sb = Sb * alpha2Factor + (dB % dB) * (1 - alpha2Factor);
        Mat<T> Vbt = Vb / (1 - alpha1);
        Mat<T> Sbt = Sb / (1 - alpha2);
        B -= Vbt / (SQRT(Sbt) + 1e-9) * learningRate;
        dB.zero();
        return;
    }
};
template<typename T>
T Adam<T>::alpha1Factor(0.9);
template<typename T>
T Adam<T>::alpha2Factor(0.99);

template <typename T, template<typename> class OptimizeF>
class Layer : public OptimizeF<T>
{
public:
    std::map<int, Mat<T> > W;
    Mat<T> B;
    Mat<T> O;
    /* paramter */
    int layerDim;
    int inputDim;
    LossType lossType;
    LayerType layerType;
public:
    Layer():layerDim(0), inputDim(0){}
    virtual ~Layer(){}
    Layer(const Layer& layer):
        OptimizeF<T>(layer),
        W(layer.W),
        B(layer.B),
        O(layer.O),
        layerDim(layer.layerDim),
        inputDim(layer.inputDim),
        lossType(layer.lossType),
        layerType(layer.layerType) {}
    Layer& operator = (const Layer& layer)
    {
        if (this == &layer) {
            return *this;
        }
        W = layer.W;
        B = layer.B;
        O = layer.O;
        /* paramter */
        layerDim = layer.layerDim;
        inputDim = layer.inputDim;
        lossType = layer.lossType;
        layerType = layer.layerType;
        OptimizeF<T>::operator=(layer);
        return *this;
    }
    Layer(LayerType layerType_, LossType lossType_, int layerDim_):
        OptimizeF<T>(layerType_, layerDim_, 1)
    {
        layerDim = layerDim_;
        lossType = lossType_;
        layerType = layerType_;
        B = Mat<T>(layerDim, 1, UNIFORM_RAND);
        O = Mat<T>(layerDim, 1);
    }

    Layer(LayerType layerType, LossType lossType, int layerDim, int inputDim) :
        OptimizeF<T>(layerType, layerDim, inputDim)
    {
        this->layerDim = layerDim;
        this->inputDim = inputDim;
        this->lossType = lossType;
        this->layerType = layerType;
        W[0] = Mat<T>(layerDim, inputDim, UNIFORM_RAND);
        B = Mat<T>(layerDim, 1, UNIFORM_RAND);
        O = Mat<T>(layerDim, 1);
    }

    void connect(int from, int inputDim_)
    {
        this->inputDim = inputDim_;
        W[from] = Mat<T>(layerDim, inputDim, UNIFORM_RAND);
        return OptimizeF<T>::connect(from, layerDim, inputDim);
    }
    void optimize(const std::vector<int> &previous, T learningRate)
    {
        return OptimizeF<T>::_(previous, layerType, learningRate, W, B);
    }
};


template <typename T,
          template<typename> class ActivateF = Relu,
          template<typename> class OptimizeF = SGD>
class MLP : public Graph<Layer<T, OptimizeF> >
{
public:
    struct LayerParam
    {
        LayerType layerType;
        LossType lossType;
        int layerDim;
        int inputDim;
        std::string layerName;
    };
    struct EdgeParam
    {
        std::string fromName;
        std::string toName;
    };

    using DataType = T;
    using TLayer = Layer<T, OptimizeF>;
    using LayerParams = std::vector<LayerParam>;
    using GraphParams = std::vector<EdgeParam>;
    using DAG = Graph<Layer<T, OptimizeF> >;
    using Input = std::map<std::string, Mat<T> >;
    using InputVec = std::vector<Input>;
    using Target = std::vector<Mat<T> >;
    using Targets = std::map<std::string, Mat<T> >;
    using Flat = MLP<T, ActivateF, NoneOpt>;
public:
    MLP(){}
    ~MLP(){}
    MLP(const MLP& mlp):DAG(mlp){}
    MLP& operator = (const MLP& mlp)
    {
        if (this == &mlp) {
            return *this;
        }
        DAG::operator=(mlp);
        return *this;
    }
    MLP(const LayerParams& layerParam, const GraphParams& graphParam)
    {
        for (int i = 0; i < layerParam.size(); i++) {
            if (layerParam[i].layerType == INPUT) {
                DAG::insertVertex(TLayer(layerParam[i].layerType,
                                         layerParam[i].lossType,
                                         layerParam[i].layerDim,
                                         layerParam[i].inputDim),
                                  layerParam[i].layerName);
            } else {
                DAG::insertVertex(TLayer(layerParam[i].layerType,
                                         layerParam[i].lossType,
                                         layerParam[i].layerDim),
                                  layerParam[i].layerName);
            }
        }
        for (int i = 0; i < graphParam.size(); i++) {
             connectLayer(graphParam[i].fromName, graphParam[i].toName);
        }
        DAG::generate();
    }

    void addLayer(LayerType layerType,
                  LossType lossType,
                  int layerDim,
                  const std::string &layerName)
    {
        return DAG::insertVertex(TLayer(layerType, lossType, layerDim), layerName);
    }

    void addLayer(LayerType layerType,
                  LossType lossType,
                  int layerDim,
                  int inputDim,
                  const std::string &layerName)
    {
        return DAG::insertVertex(TLayer(layerType, lossType, layerDim, inputDim), layerName);
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

    Flat clone()
    {
        Flat dst;
        /* copy vertexs */
        for (auto& x : DAG::vertexs) {
            auto& layer = x.object;
            if (layer.layerType == INPUT) {
                dst.addLayer(layer.layerType, layer.lossType, layer.layerDim, layer.inputDim, x.name);
            } else {
                dst.addLayer(layer.layerType, layer.lossType, layer.layerDim ,x.name);
            }
        }
        /* copy edges */
        for (int current : DAG::topologySequence) {
            auto &layer = dst.getObject(current);
            if (layer.layerType != INPUT) {
                for (int  from : DAG::previous.at(current)) {
                    auto &preLayer = DAG::getObject(from);
                    layer.connect(from, preLayer.layerDim);
                }
            }
        }
        dst.edges = DAG::edges;
        dst.topologySequence = DAG::topologySequence;
        dst.traversalSequence = DAG::traversalSequence;
        dst.previous = DAG::previous;
        dst.nexts = DAG::nexts;
        /* copy data */
        copyTo(dst);
        return dst;
    }

    void copyTo(Flat& dst)
    {
        for (int i = 0; i < DAG::vertexs.size(); i++) {
            dst.vertexs[i].object.W = DAG::vertexs[i].object.W;
            dst.vertexs[i].object.B = DAG::vertexs[i].object.B;
        }
        return;
    }

    void softUpdateTo(Flat& dst, double alpha)
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
                layer.O = ActivateF<T>::_(layer.W[0] * x.at(DAG::vertexs[current].name) + layer.B);
            } else {
                Mat<T> s(layer.O.rows, layer.O.cols);
                for (int from : DAG::previous[current]) {
                    auto &preLayer = DAG::getObject(from);
                    s += layer.W[from] * preLayer.O;
                }
                s += layer.B;
                layer.O = ActivateF<T>::_(s);
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
                Mat<T> dy = layer.E % ActivateF<T>::d(layer.O);
                if (layer.layerType == INPUT) {
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
    void optimize(double learningRate)
    {
        if (!DAG::isDAG()) {
            return;
        }
        for (int current : DAG::topologySequence) {
            DAG::getObject(current).optimize(DAG::previous.at(current), learningRate);
        }
        return;
    }
    void show()
    {
        DAG::vertexs[DAG::topologySequence.size() - 1].object.O.show();
        return;
    }
};
#endif // MLP_H
