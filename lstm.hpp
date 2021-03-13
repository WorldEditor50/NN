#ifndef LSTM_HPP
#define LSTM_HPP
#include "matrix.hpp"
using namespace ML;

template <int cellDim, int inputDim, int outputDim>
class Cell
{
public:
    using DataType = double;
public:
    /* forget gate */
    Mat<DataType> Wf;
    Mat<DataType> Uf;
    Mat<DataType> Bf;
    /* input gate */
    Mat<DataType> Wi;
    Mat<DataType> Ui;
    Mat<DataType> Bi;
    Mat<DataType> Wa;
    Mat<DataType> Ua;
    Mat<DataType> Ba;
    /* output gate */
    Mat<DataType> Wo;
    Mat<DataType> Uo;
    Mat<DataType> Bo;
    /* buffer */
    /* forget gate */
    Mat<DataType> dWf;
    Mat<DataType> dUf;
    Mat<DataType> dBf;
    /* input gate */
    Mat<DataType> dWi;
    Mat<DataType> dUi;
    Mat<DataType> dBi;
    Mat<DataType> dWa;
    Mat<DataType> dUa;
    Mat<DataType> dBa;
    /* output gate */
    Mat<DataType> dWo;
    Mat<DataType> dUo;
    Mat<DataType> dBo;
    /* cell state */
    /* forget gate */
    Mat<DataType> f;
    /* input gate */
    Mat<DataType> i;
    Mat<DataType> a;
    /* output gate */
    Mat<DataType> o;
    /* cell state */
    Mat<DataType> s;
    /* hiddien output */
    Mat<DataType> h;
    /* gradient */
    Mat<DataType> ds;
    Mat<DataType> dh;
    Mat<DataType> dx;
    Mat<DataType> x;
    /* predict */
    Mat<DataType> yp;
    Mat<DataType> Wp;
    Mat<DataType> Bp;
public:
    Cell()
    {
        /* forget gate */
        Wf = Mat<DataType>(cellDim, inputDim, UNIFORM_RAND);
        Uf = Mat<DataType>(cellDim, outputDim, UNIFORM_RAND);
        Bf = Mat<DataType>(cellDim, 1, UNIFORM_RAND);
        /* input gate */
        Wi = Mat<DataType>(cellDim, inputDim, UNIFORM_RAND);
        Ui = Mat<DataType>(cellDim, outputDim, UNIFORM_RAND);
        Bi = Mat<DataType>(cellDim, 1, UNIFORM_RAND);
        Wa = Mat<DataType>(cellDim, inputDim, UNIFORM_RAND);
        Ua = Mat<DataType>(cellDim, outputDim, UNIFORM_RAND);
        Ba = Mat<DataType>(cellDim, 1, UNIFORM_RAND);
        /* output gate */
        Wo = Mat<DataType>(cellDim, inputDim, UNIFORM_RAND);
        Uo = Mat<DataType>(cellDim, outputDim, UNIFORM_RAND);
        Bo = Mat<DataType>(cellDim, 1, UNIFORM_RAND);
        /* gradient */
        /* forget gate */
        dWf = Mat<DataType>(cellDim, inputDim);
        dUf = Mat<DataType>(cellDim, outputDim);
        dBf = Mat<DataType>(cellDim, 1);
        /* input gate */
        dWi = Mat<DataType>(cellDim, inputDim);
        dUi = Mat<DataType>(cellDim, outputDim);
        dBi = Mat<DataType>(cellDim, 1);
        dWa = Mat<DataType>(cellDim, inputDim);
        dUa = Mat<DataType>(cellDim, outputDim);
        dBa = Mat<DataType>(cellDim, 1);
        /* output gate */
        dWo = Mat<DataType>(cellDim, inputDim);
        dUo = Mat<DataType>(cellDim, outputDim);
        dBo = Mat<DataType>(cellDim, 1);
        /* cell state */
        f = Mat<DataType>(cellDim, 1);
        i = Mat<DataType>(cellDim, 1);
        a = Mat<DataType>(cellDim, 1);
        o = Mat<DataType>(cellDim, 1);
        s = Mat<DataType>(cellDim, 1);
        h = Mat<DataType>(cellDim, 1);
        ds = Mat<DataType>(cellDim, 1);
        dh = Mat<DataType>(cellDim, 1);
        dx = Mat<DataType>(cellDim, 1);
        /* predict */
        yp = Mat<DataType>(cellDim, 1);
        Wp = Mat<DataType>(cellDim, outputDim);
        Bp = Mat<DataType>(cellDim, 1);
    }
    void zero()
    {
        dWf.zero();
        dUf.zero();
        dBf.zero();
        dWi.zero();
        dUi.zero();
        dBi.zero();
        dWa.zero();
        dUa.zero();
        dBa.zero();
        dWo.zero();
        dUo.zero();
        dBo.zero();
    }

    void SGD(double learningRate)
    {
        Wf -= dWf * learningRate;
        Uf -= dUf * learningRate;
        Bf -= dBf * learningRate;
        Wi -= dWi * learningRate;
        Ui -= dUi * learningRate;
        Bi -= dBi * learningRate;
        Wa -= dWa * learningRate;
        Ua -= dUa * learningRate;
        Ba -= dBa * learningRate;
        Wo -= dWo * learningRate;
        Uo -= dUo * learningRate;
        Bo -= dBo * learningRate;
        zero();
        return;
    }
    void forward(const Mat<DataType> &x,
                 const Mat<DataType> &h_,
                 const Mat<DataType> &s_)
    {
        /* forget gate */
        f = SIGMOID(Wf * x + Uf * h_ + Bf);
        /* input gate */
        i = SIGMOID(Wi * x + Ui * h_ + Bi);
        a = TANH(Wa * x + Ua * h_ + Ba);
        /* cell state */
        s = f % s_ +  i % a;
        /* output gate */
        o = SIGMOID(Wo * x + Uo * h_ + Bo);
        h = o % TANH(s);
        /* predict */
        yp = SOFTMAX(Wp * h + Bp);
    }

    void gradient(const Mat<DataType> &deltaH,
                  const Mat<DataType> &deltaS)
    {

    }
};

template <int cellDim, int inputDim, int outputDim>
class LSTM
{
public:
    using CELL = Cell<cellDim, inputDim, outputDim>;
    using SequenceIO = std::vector<Mat<double> >;
    CELL cell;
    std::vector<CELL> buffer;
public:
    void forward(const SequenceIO &seq)
    {
        for (auto& x : seq) {
            cell.forward(x);
        }
        return;
    }

    void backward(const SequenceIO &x, const SequenceIO &y)
    {
        return;
    }
};

#endif // LSTM_HPP
