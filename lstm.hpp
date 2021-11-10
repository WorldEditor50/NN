#ifndef LSTM_HPP
#define LSTM_HPP
#include "matrix.hpp"
namespace ML {
using T = double;
template <int inputDim, int hiddenDim, int outputDim>
class CellParam
{
public:
    /* forget gate */
    Mat<T> Wf;
    Mat<T> Uf;
    Mat<T> Bf;
    /* input gate */
    Mat<T> Wi;
    Mat<T> Ui;
    Mat<T> Bi;
    Mat<T> Wg;
    Mat<T> Ug;
    Mat<T> Bg;
    /* output gate */
    Mat<T> Wo;
    Mat<T> Uo;
    Mat<T> Bo;
    /* output */
    Mat<T> Wp;
    Mat<T> Bp;
public:
    CellParam()
    {
        /* forget gate */
        Wf = Mat<T>(hiddenDim, inputDim);
        Uf = Mat<T>(hiddenDim, hiddenDim);
        Bf = Mat<T>(hiddenDim, 1);
        /* input gate */
        Wi = Mat<T>(hiddenDim, inputDim);
        Ui = Mat<T>(hiddenDim, hiddenDim);
        Bi = Mat<T>(hiddenDim, 1);
        Wg = Mat<T>(hiddenDim, inputDim);
        Ug = Mat<T>(hiddenDim, hiddenDim);
        Bg = Mat<T>(hiddenDim, 1);
        /* output gate */
        Wo = Mat<T>(hiddenDim, inputDim);
        Uo = Mat<T>(hiddenDim, hiddenDim);
        Bo = Mat<T>(hiddenDim, 1);
        /* output */
        Wp = Mat<T>(outputDim, hiddenDim);
        Bp = Mat<T>(outputDim, 1);
    }
    void zero()
    {
        Wf.zero();
        Uf.zero();
        Bf.zero();
        Wi.zero();
        Ui.zero();
        Bi.zero();
        Wg.zero();
        Ug.zero();
        Bg.zero();
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
        Wg.uniformRandom();
        Ug.uniformRandom();
        Bg.uniformRandom();
        Wo.uniformRandom();
        Uo.uniformRandom();
        Bo.uniformRandom();
    }
};

template <int hiddenDim, int outputDim>
class CellState
{
public:
    /* forget gate */
    Mat<T> f;
    /* input gate */
    Mat<T> i;
    Mat<T> g;
    /* output gate */
    Mat<T> o;
    /* cell state */
    Mat<T> c;
    /* hiddien output */
    Mat<T> h;
    /* output */
    Mat<T> y;
public:
    CellState()
    {
        f = Mat<T>(hiddenDim, 1);
        i = Mat<T>(hiddenDim, 1);
        g = Mat<T>(hiddenDim, 1);
        o = Mat<T>(hiddenDim, 1);
        c = Mat<T>(hiddenDim, 1);
        h = Mat<T>(hiddenDim, 1);
        y = Mat<T>(outputDim, 1);
    }

    CellState(const CellState &state):
        f(state.f),i(state.i),g(state.g),
        o(state.o),c(state.c),h(state.h),y(state.y){}

    CellState& operator = (const CellState &state)
    {
        if (this == &state) {
            return *this;
        }
        f = state.f;
        i = state.i;
        g = state.g;
        o = state.o;
        c = state.c;
        h = state.h;
        y = state.y;
        return *this;
    }
    void clear()
    {
        f.zero();
        i.zero();
        g.zero();
        o.zero();
        c.zero();
        h.zero();
        y.zero();
    }
};

template <int inputDim, int hiddenDim, int outputDim>
class LSTM
{
public:
    using Param = CellParam<inputDim, hiddenDim, outputDim>;
    using State = CellState<hiddenDim, outputDim>;
public:
     Param P;
     Param dP;
     Param Sp;
     State state;
     State delta;
     State delta_;
     std::vector<State> states;
public:
    LSTM()
    {
        P.random();
    }

    Mat<T>& feedForward(const Mat<T> &x)
    {
        /*
                                                            y
                                                            |
                                                           h(t)
                                              c(t)          |
                c(t-1) -->--x-----------------+----------------->--- c(t)
                            |                 |             |
                            |                 |            tanh
                            |                 |             |
                            |          -------x      -------x
                         f  |        i |      | g    | o    |
                            |          |      |      |      |
                         sigmoid    sigmoid  tanh  sigmoid  |
                            |          |      |      |      |
                h(t-1) -->----------------------------      ---->--- h(t)
                            |
                            x(t)

            i = sigmoid(Wii*x + bii + Uhi*h + bhi);
            f = sigmoid(Wif*x + bif + Uhf*h + bhf);
            g = tanh(Wig*x + big + Uhg*h + bhg);
            o = sigmoid(Wio*x + bio + Uho*h + bho);
            f = sigmoid(Wif*x + bif + Uhf*h + bhf);
            c' = f*c + i*g
            h' = o*tanh(c')
            y = sigmoid(W*h' + b)
        */
        /* input gate */
        state.i = Sigmoid<T>::_(P.Wi * x + P.Ui * state.h + P.Bi);
        /* forget gate */
        state.f = Sigmoid<T>::_(P.Wf * x + P.Uf * state.h + P.Bf);
        /* output gate */
        state.o = Sigmoid<T>::_(P.Wo * x + P.Uo * state.h + P.Bo);
        state.g = Tanh<T>::_(P.Wg * x + P.Ug * state.h + P.Bg);
        /* cell state */
        state.c = state.f % state.c +  state.i % state.g;
        state.h = state.o % Tanh<T>::_(state.c);
        /* predict */
        state.y = Sigmoid<T>::_(P.Wp * state.h + P.Bp);
        return state.y;
    }

    void forward(const std::vector<Mat<T> > &seq)
    {
        state.clear();
        states.push_back(state);
        for (auto &x : seq) {  
            feedForward(x);
            states.push_back(state);
        }
        return;
    }

    void gradient(const std::vector<Mat<T> > &x, const std::vector<Mat<T> > &y)
    {
        delta.clear();
        delta_.clear();
        for (int t = states.size() - 2; t >= 1; t--) {
            /* loss */
            delta.y = (states[t].y - y[t]) * 2;
            /* backward */
            delta.h += P.Wp.Tr() * delta.y;
            delta.h += P.Ui.Tr() * delta_.i;
            delta.h += P.Ug.Tr() * delta_.g;
            delta.h += P.Uf.Tr() * delta_.f;
            delta.h += P.Uo.Tr() * delta_.o;

            delta.o = delta.h % Tanh<T>::_(states[t].c) % Sigmoid<T>::d(states[t].o);
            delta.c = delta.h % states[t].o % Tanh<T>::d(states[t].c) +
                    delta_.c % states[t + 1].f;
            delta.f = delta.c % states[t - 1].c % Sigmoid<T>::d(states[t].f);
            delta.i = delta.c % states[t].g % Sigmoid<T>::d(states[t].i);
            delta.g = delta.c % states[t].i % Tanh<T>::d(states[t].g);

            /* gradient */
            dP.Wi += delta.i * x[t].Tr();
            dP.Wg += delta.g * x[t].Tr();
            dP.Wf += delta.f * x[t].Tr();
            dP.Wo += delta.o * x[t].Tr();

            dP.Ui += delta.i * states[t - 1].h.Tr();
            dP.Ug += delta.g * states[t - 1].h.Tr();
            dP.Uf += delta.f * states[t - 1].h.Tr();
            dP.Uo += delta.o * states[t - 1].h.Tr();

            dP.Bi += delta.i;
            dP.Bg += delta.g;
            dP.Bf += delta.f;
            dP.Bo += delta.o;

            dP.Wp += (delta.y % Sigmoid<T>::d(states[t].y)) * states[t].h.Tr();
            dP.Bp += delta.y % Sigmoid<T>::d(states[t].y);
            /* save */
            delta_ = delta;
        }
        states.clear();
        return;
    }

    void SGD(double learningRate)
    {
        P.Wf -= dP.Wf * learningRate;
        P.Uf -= dP.Uf * learningRate;
        P.Bf -= dP.Bf * learningRate;

        P.Wi -= dP.Wi * learningRate;
        P.Ui -= dP.Ui * learningRate;
        P.Bi -= dP.Bi * learningRate;

        P.Wg -= dP.Wg * learningRate;
        P.Ug -= dP.Ug * learningRate;
        P.Bg -= dP.Bg * learningRate;

        P.Wo -= dP.Wo * learningRate;
        P.Uo -= dP.Uo * learningRate;
        P.Bo -= dP.Bo * learningRate;

        P.Wp -= dP.Wp * learningRate;
        P.Bp -= dP.Bp * learningRate;
        dP.zero();
        return;
    }

    void RMSProp(double rho, double learningRate)
    {
        Sp.Wi = Sp.Wi * rho + (dP.Wi % dP.Wi) * (1 - rho);
        Sp.Wg = Sp.Wg * rho + (dP.Wg % dP.Wg) * (1 - rho);
        Sp.Wf = Sp.Wf * rho + (dP.Wf % dP.Wf) * (1 - rho);
        Sp.Wo = Sp.Wo * rho + (dP.Wo % dP.Wo) * (1 - rho);
        Sp.Wp = Sp.Wp * rho + (dP.Wp % dP.Wp) * (1 - rho);

        Sp.Ui = Sp.Ui * rho + (dP.Ui % dP.Ui) * (1 - rho);
        Sp.Ug = Sp.Ug * rho + (dP.Ug % dP.Ug) * (1 - rho);
        Sp.Uf = Sp.Uf * rho + (dP.Uf % dP.Uf) * (1 - rho);
        Sp.Uo = Sp.Uo * rho + (dP.Uo % dP.Uo) * (1 - rho);

        Sp.Bi = Sp.Bi * rho + (dP.Bi % dP.Bi) * (1 - rho);
        Sp.Bg = Sp.Bg * rho + (dP.Bg % dP.Bg) * (1 - rho);
        Sp.Bf = Sp.Bf * rho + (dP.Bf % dP.Bf) * (1 - rho);
        Sp.Bo = Sp.Bo * rho + (dP.Bo % dP.Bo) * (1 - rho);
        Sp.Bp = Sp.Bp * rho + (dP.Bp % dP.Bp) * (1 - rho);

        P.Wi -= dP.Wi / (SQRT(Sp.Wi) + 1e-9) * learningRate;
        P.Wg -= dP.Wg / (SQRT(Sp.Wg) + 1e-9) * learningRate;
        P.Wf -= dP.Wf / (SQRT(Sp.Wf) + 1e-9) * learningRate;
        P.Wo -= dP.Wo / (SQRT(Sp.Wo) + 1e-9) * learningRate;
        P.Wp -= dP.Wp / (SQRT(Sp.Wp) + 1e-9) * learningRate;

        P.Ui -= dP.Ui / (SQRT(Sp.Ui) + 1e-9) * learningRate;
        P.Ug -= dP.Ug / (SQRT(Sp.Ug) + 1e-9) * learningRate;
        P.Uf -= dP.Uf / (SQRT(Sp.Uf) + 1e-9) * learningRate;
        P.Uo -= dP.Uo / (SQRT(Sp.Uo) + 1e-9) * learningRate;

        P.Bi -= dP.Bi / (SQRT(Sp.Bi) + 1e-9) * learningRate;
        P.Bg -= dP.Bg / (SQRT(Sp.Bg) + 1e-9) * learningRate;
        P.Bf -= dP.Bf / (SQRT(Sp.Bf) + 1e-9) * learningRate;
        P.Bo -= dP.Bo / (SQRT(Sp.Bo) + 1e-9) * learningRate;
        P.Bp -= dP.Bp / (SQRT(Sp.Bp) + 1e-9) * learningRate;

        dP.zero();
        return;
    }
};

}
#endif // LSTM_HPP
