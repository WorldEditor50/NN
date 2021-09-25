#ifndef EXPRESSION_HPP
#define EXPRESSION_HPP
#include <cmath>

namespace Exp {

using T = float;
class Constance
{
public:
    Constance():c(0){}
    explicit Constance(const T c_):c(c_){}
    inline T operator()(T x) const {return c;}
private:
    T c;
};

class Varibale
{
public:
    inline T operator()(T x) const {return x;}
};
/* expression */
template <typename TExpr>
class Expr
{
public:
    Expr(){}
    explicit Expr(const TExpr &expr_):expr(expr_){}
    explicit Expr(const T &c):expr(Constance(c)){}
    inline T operator()(T x) const {return expr(x);}
private:
    TExpr expr;
};
using Const = Expr<Constance>;
using Var = Expr<Varibale>;
template<typename TLeft, typename TRight, template<typename> class TFunctor>
class BinaryOperator
{
public:
    BinaryOperator(){}
    explicit BinaryOperator(const TLeft &left_, const TRight &right_):
        left(left_), right(right_){}
    inline T operator()(T x) const
    {
        return TFunctor<T>::apply(left(x), right(x));
    }
private:
    TLeft left;
    TRight right;
};

template<typename TRight, template<typename> class TFunctor>
class UnaryOperator
{
public:
    explicit UnaryOperator(const TRight &right_):right(right_){}
    inline T operator()(T x) const
    {
        return TFunctor<T>::apply(right(x));
    }
private:
    TRight right;
};
/* operator */
template <typename T>
class Plus
{
public:
    inline static T apply(T x1, T x2) {return x1 + x2;};
};

template <typename T>
class Minus
{
public:
    inline static T apply(T x1, T x2) {return x1 - x2;};
};

template <typename T>
class Multi
{
public:
    inline static T apply(T x1, T x2) {return x1 * x2;};
};

template <typename T>
class Divide
{
public:
    inline static T apply(T x1, T x2) {return x1 / x2;};
};

template <typename T>
class Negative
{
public:
    inline static T apply(T x) {return -x;};
};
/* function */
template <typename T>
class Pow
{
public:
    inline static T apply(T x, T n) {return pow(x, n);};
};
template <typename T>
class Sqrt
{
public:
    inline static T apply(T x) {return sqrt(x);};
};
template <typename T>
class Exp
{
public:
    inline static T apply(T x) {return exp(x);};
};
template <typename T>
class Tanh
{
public:
    inline static T apply(T x) {return tanh(x);};
};
template <typename T>
class Sigmoid
{
public:
    inline static T apply(T x) {return exp(x) / (1 + exp(x));};
};
template <typename T>
class Relu
{
public:
    inline static T apply(T x) {return x > 0 ? x : 0;};
};
/* global operator */
/* plus */
template<typename TLeft, typename TRight>
Expr<BinaryOperator<Expr<TLeft>, Expr<TRight>, Plus> > operator + (const Expr<TLeft> &left, const Expr<TRight> &right)
{
    using PlusOperator = BinaryOperator<Expr<TLeft>, Expr<TRight>, Plus>;
    return Expr<PlusOperator>(PlusOperator(left, right));
}
/* minus */
template<typename TLeft, typename TRight>
inline Expr<BinaryOperator<Expr<TLeft>, Expr<TRight>, Minus> > operator - (const Expr<TLeft> &left, const Expr<TRight> &right)
{
    using MinusOperator = BinaryOperator<Expr<TLeft>, Expr<TRight>, Minus>;
    return Expr<MinusOperator>(MinusOperator(left, right));
}
/* multi */
template<typename TLeft, typename TRight>
inline Expr<BinaryOperator<Expr<TLeft>, Expr<TRight>, Multi> > operator * (const Expr<TLeft> &left, const Expr<TRight> &right)
{
    using MultiOperator = BinaryOperator<Expr<TLeft>, Expr<TRight>, Multi>;
    return Expr<MultiOperator>(MultiOperator(left, right));
}
/* divide */
template<typename TLeft, typename TRight>
inline Expr<BinaryOperator<Expr<TLeft>, Expr<TRight>, Divide> > operator / (const Expr<TLeft> &left, const Expr<TRight> &right)
{
    using DivideOperator = BinaryOperator<Expr<TLeft>, Expr<TRight>, Divide>;
    return Expr<DivideOperator >(DivideOperator(left, right));
}

/* negative */
template<typename TRight>
inline Expr<UnaryOperator<Expr<TRight>, Negative> > operator - (const Expr<TRight> &right)
{
    using NegativeOperator = UnaryOperator<Expr<TRight>, Negative>;
    return Expr<NegativeOperator>(NegativeOperator(right));
}

template<typename TExpr>
inline Expr<UnaryOperator<Expr<TExpr>, Sqrt> > SQRT(const Expr<TExpr> &expr)
{
    using SqrTFunctor = UnaryOperator<Expr<TExpr>, Sqrt>;
    return Expr<SqrTFunctor >(SqrTFunctor(expr));
}

template<typename TExpr>
inline Expr<UnaryOperator<Expr<TExpr>, Exp> > EXP(const Expr<TExpr> &expr)
{
    using ExpOperator = UnaryOperator<Expr<TExpr>, Exp>;
    return Expr<ExpOperator>(ExpOperator(expr));
}

template<typename TExpr>
inline Expr<UnaryOperator<Expr<TExpr>, Tanh> > TANH(const Expr<TExpr> &expr)
{
    using TanhOperator = UnaryOperator<Expr<TExpr>, Tanh>;
    return Expr<TanhOperator>(TanhOperator(expr));
}

template<typename TExpr>
inline Expr<UnaryOperator<Expr<TExpr>, Sigmoid> > SIGMOID(const Expr<TExpr> &expr)
{
    using SigmoidOperator = UnaryOperator<Expr<TExpr>, Sigmoid>;
    return Expr<SigmoidOperator>(SigmoidOperator(expr));
}

template<typename TExpr>
Expr<UnaryOperator<Expr<TExpr>, Relu> > RELU(const Expr<TExpr> &expr)
{
    using ReluOperator = UnaryOperator<Expr<TExpr>, Relu>;
    return Expr<ReluOperator>(ReluOperator(expr));
}

}
#endif // EXPRESSION_HPP
