#ifndef LSTM_HPP
#define LSTM_HPP
#include "matrix.hpp"
using namespace ML;

namespace lstm {

template <typename DataType, int inputDim, int hiddenDim, int outputDim>
class CellParam
{
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
    /* output */
    Mat<DataType> Wp;
    Mat<DataType> Bp;
public:
    CellParam()
    {
        /* forget gate */
        Wf = Mat<DataType>(hiddenDim, inputDim);
        Uf = Mat<DataType>(hiddenDim, hiddenDim);
        Bf = Mat<DataType>(hiddenDim, 1);
        /* input gate */
        Wi = Mat<DataType>(hiddenDim, inputDim);
        Ui = Mat<DataType>(hiddenDim, hiddenDim);
        Bi = Mat<DataType>(hiddenDim, 1);
        Wa = Mat<DataType>(hiddenDim, inputDim);
        Ua = Mat<DataType>(hiddenDim, hiddenDim);
        Ba = Mat<DataType>(hiddenDim, 1);
        /* output gate */
        Wo = Mat<DataType>(hiddenDim, inputDim);
        Uo = Mat<DataType>(hiddenDim, hiddenDim);
        Bo = Mat<DataType>(hiddenDim, 1);
        /* output */
        Wp = Mat<DataType>(outputDim, hiddenDim);
        Bp = Mat<DataType>(outputDim, 1);
    }
    void zero()
    {
        Wf.zero();
        Uf.zero();
        Bf.zero();
        Wi.zero();
        Ui.zero();
        Bi.zero();
        Wa.zero();
        Ua.zero();
        Ba.zero();
        Wo.zero();
        Uo.zero();
        Bo.zero();
    }
    void random()
    {
        Wf.uniformRandom();
        Uf.uniformRandom();
        Bf.uniformRandom();
        Wi.uniformRandom();
        Ui.uniformRandom();
        Bi.uniformRandom();
        Wa.uniformRandom();
        Ua.uniformRandom();
        Ba.uniformRandom();
        Wo.uniformRandom();
        Uo.uniformRandom();
        Bo.uniformRandom();
    }
};

template <typename DataType, int hiddenDim, int outputDim>
class CellState
{
public:
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
    /* output */
    Mat<DataType> y;
public:
    CellState()
    {
        f = Mat<DataType>(hiddenDim, 1);
        i = Mat<DataType>(hiddenDim, 1);
        a = Mat<DataType>(hiddenDim, 1);
        o = Mat<DataType>(hiddenDim, 1);
        s = Mat<DataType>(hiddenDim, 1);
        h = Mat<DataType>(hiddenDim, 1);
        y = Mat<DataType>(outputDim, 1);
    }

    void copyFrom(const CellState &state)
    {
        f = state.f;
        i = state.i;
        a = state.a;
        o = state.o;
        s = state.s;
        h = state.h;
        y = state.y;
    }

    CellState(const CellState &state)
    {
        copyFrom(state);
    }

    CellState& operator = (const CellState &state)
    {
        if (this == &state) {
            return *this;
        }
        copyFrom(state);
        return *this;
    }
};

template <int inputDim, int hiddenDim, int outputDim>
class LSTM
{
public:
    using DataType = double;
    using Param = CellParam<DataType, inputDim, hiddenDim, outputDim>;
    using State = CellState<DataType, hiddenDim, outputDim>;
    using SequenceIO = std::vector<Mat<DataType> >;
public:
     Param param;
     State state;
     Param dParam;
     Param Sp;
     State delta;
     State delta_future;
     std::vector<State> states;
public:
    LSTM()
    {
        param.random();
    }

    Mat<DataType>& feedForward(const Mat<DataType> &x)
    {
        Mat<DataType> &h_ = state.h;
        Mat<DataType> &s_ = state.s;
        /* forget gate */
        state.f = SIGMOID(param.Wf * x + param.Uf * h_ + param.Bf);
        /* input gate */
        state.i = SIGMOID(param.Wi * x + param.Ui * h_ + param.Bi);
        state.a = TANH(param.Wa * x + param.Ua * h_ + param.Ba);
        /* cell state */
        state.s = state.f % s_ +  state.i % state.a;
        /* output gate */
        state.o = SIGMOID(param.Wo * x + param.Uo * h_ + param.Bo);
        state.h = state.o % TANH(state.s);
        /* predict */
        state.y = SIGMOID(param.Wp * state.h + param.Bp);
        return state.y;
    }

    void forward(const SequenceIO &seq)
    {
        for (auto &x : seq) {
            feedForward(x);
            /* save state */
            states.push_back(state);
        }
        return;
    }

    void gradient(SequenceIO &x, const SequenceIO &y)
    {
        /* backward */
        for (int i = states.size() - 2; i >= 0; i--) {
            /* loss */
            delta.y = states[i].y - y[i];
            delta.h += param.Wp.Tr() * delta.y;
            delta.h += param.Ui.Tr() * delta_future.i;
            delta.h += param.Ua.Tr() * delta_future.a;
            delta.h += param.Uf.Tr() * delta_future.f;
            delta.h += param.Uo.Tr() * delta_future.o;

            delta.o = delta.h % TANH(states[i + 1].s) % DSIGMOID(states[i].o);
            delta.s = delta.h % states[i].o % DTANH(states[i + 1].s) +
                    delta_future.s % states[i + 1].f;
            delta.f = delta.s % states[i].s % DSIGMOID(states[i].f);
            delta.i = delta.s % states[i].a % DSIGMOID(states[i].i);
            delta.a = delta.s % states[i].i % DSIGMOID(states[i].a);

            /* gradient */
            dParam.Wi += delta.i * x[i].Tr();
            dParam.Wa += delta.a * x[i].Tr();
            dParam.Wf += delta.f * x[i].Tr();
            dParam.Wo += delta.o * x[i].Tr();

            dParam.Ui += delta.i * states[i].h.Tr();
            dParam.Ua += delta.a * states[i].h.Tr();
            dParam.Uf += delta.f * states[i].h.Tr();
            dParam.Uo += delta.o * states[i].h.Tr();

            dParam.Bi += delta.i;
            dParam.Ba += delta.a;
            dParam.Bf += delta.f;
            dParam.Bo += delta.o;

            dParam.Wp += delta.y % DSIGMOID(states[i].y) * states[i + 1].h.Tr();
            dParam.Bp += delta.y % DSIGMOID(states[i].y);
            /* save */
            delta_future = delta;
        }
        states.clear();
        return;
    }

    void SGD(double learningRate)
    {
        param.Wf -= dParam.Wf * learningRate;
        param.Uf -= dParam.Uf * learningRate;
        param.Bf -= dParam.Bf * learningRate;

        param.Wi -= dParam.Wi * learningRate;
        param.Ui -= dParam.Ui * learningRate;
        param.Bi -= dParam.Bi * learningRate;

        param.Wa -= dParam.Wa * learningRate;
        param.Ua -= dParam.Ua * learningRate;
        param.Ba -= dParam.Ba * learningRate;

        param.Wo -= dParam.Wo * learningRate;
        param.Uo -= dParam.Uo * learningRate;
        param.Bo -= dParam.Bo * learningRate;

        param.Wp -= dParam.Wp * learningRate;
        param.Bp -= dParam.Bp * learningRate;
        dParam.zero();
        return;
    }

    void RMSProp(double rho, double learningRate)
    {
        Sp.Wi = Sp.Wi * rho + (dParam.Wi % dParam.Wi) * (1 - rho);
        Sp.Wa = Sp.Wa * rho + (dParam.Wa % dParam.Wa) * (1 - rho);
        Sp.Wf = Sp.Wf * rho + (dParam.Wf % dParam.Wf) * (1 - rho);
        Sp.Wo = Sp.Wo * rho + (dParam.Wo % dParam.Wo) * (1 - rho);
        Sp.Wp = Sp.Wp * rho + (dParam.Wp % dParam.Wp) * (1 - rho);

        Sp.Ui = Sp.Ui * rho + (dParam.Ui % dParam.Ui) * (1 - rho);
        Sp.Ua = Sp.Ua * rho + (dParam.Ua % dParam.Ua) * (1 - rho);
        Sp.Uf = Sp.Uf * rho + (dParam.Uf % dParam.Uf) * (1 - rho);
        Sp.Uo = Sp.Uo * rho + (dParam.Uo % dParam.Uo) * (1 - rho);

        Sp.Bi = Sp.Bi * rho + (dParam.Bi % dParam.Bi) * (1 - rho);
        Sp.Ba = Sp.Ba * rho + (dParam.Ba % dParam.Ba) * (1 - rho);
        Sp.Bf = Sp.Bf * rho + (dParam.Bf % dParam.Bf) * (1 - rho);
        Sp.Bo = Sp.Bo * rho + (dParam.Bo % dParam.Bo) * (1 - rho);
        Sp.Bp = Sp.Bp * rho + (dParam.Bp % dParam.Bp) * (1 - rho);

        param.Wi -= dParam.Wi / (SQRT(Sp.Wi) + 1e-9) * learningRate;
        param.Wa -= dParam.Wa / (SQRT(Sp.Wa) + 1e-9) * learningRate;
        param.Wf -= dParam.Wf / (SQRT(Sp.Wf) + 1e-9) * learningRate;
        param.Wo -= dParam.Wo / (SQRT(Sp.Wo) + 1e-9) * learningRate;
        param.Wp -= dParam.Wp / (SQRT(Sp.Wp) + 1e-9) * learningRate;

        param.Ui -= dParam.Ui / (SQRT(Sp.Ui) + 1e-9) * learningRate;
        param.Ua -= dParam.Ua / (SQRT(Sp.Ua) + 1e-9) * learningRate;
        param.Uf -= dParam.Uf / (SQRT(Sp.Uf) + 1e-9) * learningRate;
        param.Uo -= dParam.Uo / (SQRT(Sp.Uo) + 1e-9) * learningRate;

        param.Bi -= dParam.Bi / (SQRT(Sp.Bi) + 1e-9) * learningRate;
        param.Ba -= dParam.Ba / (SQRT(Sp.Ba) + 1e-9) * learningRate;
        param.Bf -= dParam.Bf / (SQRT(Sp.Bf) + 1e-9) * learningRate;
        param.Bo -= dParam.Bo / (SQRT(Sp.Bo) + 1e-9) * learningRate;
        param.Bp -= dParam.Bp / (SQRT(Sp.Bp) + 1e-9) * learningRate;

        dParam.zero();
        return;
    }
};

}
#endif // LSTM_HPP
