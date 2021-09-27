#ifndef VECTOREXPR_HPP
#define VECTOREXPR_HPP
#include "allocator.hpp"
#include <iostream>
#include <tuple>
#include <cmath>

namespace VectorExpr {

using T = double;
/* scalar */
template<typename T>
class Scalar
{
public:
    T const &s;
public:
    Scalar(T const &s_):s(s_){}
    inline T operator[](size_t) const {return s;}
    inline size_t size() const {return 0;}
};

template <typename TExprImpl>
class Expr
{
public:
    inline const TExprImpl& impl() const {return static_cast<const TExprImpl&>(*this);}
    inline size_t size() const {return static_cast<const TExprImpl&>(*this).size();}
};

class Vector : public Expr<Vector>
{
private:
    static Allocator<T> allocator;
    T *ptr;
    size_t size_;
public:
    inline T operator[](size_t i) const {return ptr[i];}
    inline size_t size() const {return size_;}
    Vector():ptr(nullptr), size_(0){}
    explicit Vector(size_t N):ptr(allocator.allocate(N)), size_(N){}
    explicit Vector(size_t N, T x):ptr(allocator.allocate(N)), size_(N)
    {
        for (int i = 0; i < size_; i++) {
            ptr[i] = x;
        }
    }
    ~Vector()
    {
        if (ptr != nullptr) {
            allocator.deallocate(size_, ptr);
        }
    }
    Vector(const Vector &r):ptr(allocator.allocate(r.size_)), size_(r.size_)
    {
        for (int i = 0; i < size_; i++) {
            ptr[i] = r.ptr[i];
        }
    }
    Vector(Vector &&r):ptr(r.ptr),size_(r.size_)
    {
        r.ptr = nullptr;
        r.size_ = 0;
    }
    template<typename TExpr>
    Vector(const Expr<TExpr> &r)
    {
        const TExpr& expr = r.impl();
        ptr = allocator.allocate(expr.size());
        size_ = expr.size();
        for (int i = 0; i < size_; i++) {
            ptr[i] = expr[i];
        }
    }
    Vector& operator = (const Vector &r)
    {
        if (this == &r) {
            return *this;
        }
        if (size_ != r.size_) {
            allocator.deallocate(size_, ptr);
            ptr = allocator.allocate(r.size_);
            size_ = r.size_;
        }
        for (int i = 0; i < size_; i++) {
            ptr[i] = r.ptr[i];
        }
        return *this;
    }
    Vector& operator = (Vector &&r)
    {
        if (this == &r) {
            return *this;
        }
        ptr = r.ptr;
        size_ = r.size_;
        r.ptr = nullptr;
        r.size_ = 0;
        return *this;
    }

    template<typename TExpr>
    Vector& operator = (const Expr<TExpr> &r)
    {
        const TExpr& vec = r.impl();
        ptr = allocator.allocate(vec.size_);
        size_ = vec.size_;
        for (int i = 0; i < size_; i++) {
            ptr[i] = r.ptr[i];
        }
    }

    void show()
    {
        for (int i = 0; i < size_; i++) {
            std::cout<<ptr[i]<<" ";
        }
        std::cout<<std::endl;
        return;
    }

};
Allocator<T> Vector::allocator;
/* trait */
template<typename TRef>
class Trait
{
public:
    using ExpRef = const TRef&;
};

template<typename TRef>
class Trait<Scalar<TRef> >
{
public:
    using ExpRef = Scalar<TRef>;
};

/* vector operator */
template<typename TOperator, typename TLeft, typename TRight>
class BinaryOperator : public Expr<BinaryOperator<TOperator, TLeft, TRight> >
{
public:
    explicit BinaryOperator(const Expr<TLeft> &left_, const Expr<TRight> &right_):
        left(left_.impl()), right(right_.impl()){}
    inline T operator[](size_t i) const {return TOperator::apply(left[i], right[i]);}
    inline size_t size() const {return left.size();}
private:
    const TLeft &left;
    const TRight &right;
};
template<typename TOperator, typename TRight>
class UnaryOperator : public Expr<UnaryOperator<TOperator, TRight> >
{
public:
    explicit UnaryOperator(const Expr<TRight> &right_):right(right_.impl()){}
    inline T operator[](size_t i) const {return TOperator::apply(right[i]);}
    inline size_t size() const {return right.size();}
private:
    const TRight &right;
};
template<typename TOperator, typename ...TArgs>
class MultiOperator : public Expr<MultiOperator<TOperator, TArgs...> >
{
public:
    explicit MultiOperator(const Expr<TArgs>& ...args_):args(args_.impl()...){}
private:
    std::tuple<const TArgs&...> args;
};
/* basic operation */
class Plus
{
public:
    inline static T apply(T x1, T x2) {return x1 + x2;};
};

class Minus
{
public:
    inline static T apply(T x1, T x2) {return x1 - x2;};
};

class Multi
{
public:
    inline static T apply(T x1, T x2) {return x1 * x2;};
};

class Divide
{
public:
    inline static T apply(T x1, T x2) {return x1 / x2;};
};

class Negative
{
public:
    inline static T apply(T x) {return -x;};
};
/* function */
class Pow
{
public:
    inline static T apply(T x, T n) {return pow(x, n);};
};

class Sqrt
{
public:
    inline static T apply(T x) {return sqrt(x);};
};

class Exp
{
public:
    inline static T apply(T x) {return exp(x);};
};

class Tanh
{
public:
    inline static T apply(T x) {return tanh(x);};
};

class Sigmoid
{
public:
    inline static T apply(T x) {return exp(x) / (1 + exp(x));};
};

class Relu
{
public:
    inline static T apply(T x) {return x > 0 ? x : 0;};
};

template<typename TLeft, typename TRight>
inline BinaryOperator<Plus, TLeft, TRight>
operator + (const Expr<TLeft> &left_, const Expr<TRight> &right_)
{
    return BinaryOperator<Plus, TLeft, TRight>(left_, right_);
}

template<typename TLeft, typename TRight>
inline BinaryOperator<Minus, TLeft, TRight>
operator - (const Expr<TLeft> &left_, const Expr<TRight> &right_)
{
    return BinaryOperator<Minus, TLeft, TRight>(left_, right_);
}

template<typename TLeft, typename TRight>
inline BinaryOperator<Multi, TLeft, TRight>
operator * (const Expr<TLeft> &left_, const Expr<TRight> &right_)
{
    return BinaryOperator<Multi, TLeft, TRight>(left_, right_);
}

template<typename TLeft, typename TRight>
inline BinaryOperator<Divide, TLeft, TRight>
operator / (const Expr<TLeft> &left_, const Expr<TRight> &right_)
{
    return BinaryOperator<Divide, TLeft, TRight>(left_, right_);
}


T dot(const Vector &x1, const Vector &x2)
{
    T s = 0;
    auto z = x1 * x2;
    for (size_t i = 0; i < x1.size(); i++) {
        s += z[i];
    }
    return s;
}

/* negative */
template<typename TRight>
inline UnaryOperator<Negative, TRight>
operator - (const Expr<TRight> &right)
{
    return UnaryOperator<Negative, TRight>(right);
}

template<typename TExpr>
inline UnaryOperator<Sqrt, TExpr>
SQRT(const Expr<TExpr> &expr)
{
    return UnaryOperator<Sqrt, TExpr>(expr);
}

template<typename TExpr>
inline UnaryOperator<Exp, TExpr>
EXP(const Expr<TExpr> &expr)
{
    return UnaryOperator<Exp, TExpr>(expr);
}

template<typename TExpr>
inline UnaryOperator<Tanh, TExpr>
TANH(const Expr<TExpr> &expr)
{
    return UnaryOperator<Tanh, TExpr>(expr);
}

template<typename TExpr>
inline UnaryOperator<Sigmoid, TExpr>
SIGMOID(const Expr<TExpr> &expr)
{
    return UnaryOperator<Sigmoid, TExpr>(expr);
}

template<typename TExpr>
inline UnaryOperator<Relu, TExpr>
RELU(const Expr<TExpr> &expr)
{
    return UnaryOperator<Relu, TExpr>(expr);
}


}
#endif // VECTOREXPR_HPP
