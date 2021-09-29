#ifndef VECTOREXPR_HPP
#define VECTOREXPR_HPP
#include "allocator.hpp"
#include <iostream>
#include <cmath>
#include <type_traits>

namespace VectorExpr {

using T = double;

template <typename TExprImpl>
class Expr
{
public:
    inline const TExprImpl& impl() const {return static_cast<const TExprImpl&>(*this);}
    inline size_t size() const {return static_cast<const TExprImpl&>(*this).size();}
};

/* scalar */
class Scalar : public Expr<Scalar>
{
protected:
    T s;
public:
    Scalar(const T &s_):s(s_){}
    Scalar(const Scalar &r):s(r.s){}
    inline T operator[](size_t) const {return s;}
    inline size_t size() const {return 0;}
};
/* Vector */
class Vector : public Expr<Vector>
{
protected:
    static Allocator<T> allocator;
    T *ptr;
    size_t size_;
public:
    inline T operator[](size_t i) const {return ptr[i];}
    inline T& at(size_t i) const {return ptr[i];}
    inline size_t size() const {return size_;}
    Vector():ptr(nullptr), size_(0){}
    explicit Vector(size_t N):ptr(allocator.allocate(N)), size_(N){}
    explicit Vector(size_t N, T x):ptr(allocator.allocate(N)), size_(N)
    {
        for (size_t i = 0; i < size_; i++) {
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
        for (size_t i = 0; i < size_; i++) {
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
        for (size_t i = 0; i < size_; i++) {
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
        for (size_t i = 0; i < size_; i++) {
            ptr[i] = r.ptr[i];
        }
        return *this;
    }
    Vector& operator = (Vector &&r)
    {
        if (this == &r) {
            return *this;
        }
        allocator.deallocate(size_, ptr);
        ptr = r.ptr;
        size_ = r.size_;
        r.ptr = nullptr;
        r.size_ = 0;
        return *this;
    }

    template<typename TExpr>
    Vector& operator = (const Expr<TExpr> &r)
    {
        const TExpr& expr = r.impl();
        allocator.deallocate(size_, ptr);
        ptr = allocator.allocate(expr.size());
        size_ = expr.size();
        for (size_t i = 0; i < size_; i++) {
            ptr[i] = expr[i];
        }
        return *this;
    }

    void show()
    {
        for (size_t i = 0; i < size_; i++) {
            std::cout<<ptr[i]<<" ";
        }
        std::cout<<std::endl;
        return;
    }

};
Allocator<T> Vector::allocator;
/* trait */
template<typename TExpr>
class Trait
{
public:
    using Type = typename std::conditional<std::is_same<Scalar, TExpr>::value,
                                           Scalar,
                                           const TExpr&>::type;
};

/* vector operator */
template<typename TOperator, typename TLeft, typename TRight>
class BinaryOperator : public Expr<BinaryOperator<TOperator, TLeft, TRight> >
{
public:
    explicit BinaryOperator(const Expr<TLeft> &left_, const Expr<TRight> &right_):
        left(left_.impl()), right(right_.impl()){}
    inline T operator[](size_t i) const {return TOperator::apply(left[i], right[i]);}
    inline T& at(size_t i) const {return TOperator::apply(left[i], right[i]);}
    inline size_t size() const {return left.size();}
protected:
    typename Trait<TLeft>::Type left;
    typename Trait<TRight>::Type right;
};
template<typename TOperator, typename TRight>
class UnaryOperator : public Expr<UnaryOperator<TOperator, TRight> >
{
public:
    explicit UnaryOperator(const Expr<TRight> &right_):right(right_.impl()){}
    inline T operator[](size_t i) const {return TOperator::apply(right[i]);}
    inline T& at(size_t i) const {return TOperator::apply(right[i]);}
    inline size_t size() const {return right.size();}
protected:
    const TRight &right;
};

/* basic operation */
struct Plus {
    inline static T apply(T x1, T x2) {return x1 + x2;};
};

struct Minus {
    inline static T apply(T x1, T x2) {return x1 - x2;};
};

struct Multi {
    inline static T apply(T x1, T x2) {return x1 * x2;};
};

struct Divide {
    inline static T apply(T x1, T x2) {return x1 / x2;};
};

struct Negative {
    inline static T apply(T x) {return -x;};
};
/* function */
struct Pow {
    inline static T apply(T x, T n) {return pow(x, n);};
};

struct Sqrt {
    inline static T apply(T x) {return sqrt(x);};
};

struct Exp {
    inline static T apply(T x) {return exp(x);};
};

struct Tanh {
    inline static T apply(T x) {return tanh(x);};
};

struct Sigmoid {
    inline static T apply(T x) {return exp(x) / (1 + exp(x));};
};

struct Relu {
    inline static T apply(T x) {return x > 0 ? x : 0;};
};

template<typename TLeft, typename TRight>
inline BinaryOperator<Plus, TLeft, TRight>
operator + (const Expr<TLeft> &left_, const Expr<TRight> &right_)
{
    return BinaryOperator<Plus, TLeft, TRight>(left_, right_);
}
template<typename TLeft>
inline BinaryOperator<Plus, TLeft, Scalar>
operator + (const Expr<TLeft> &left_, T right_)
{
    return BinaryOperator<Plus, TLeft, Scalar>(left_, Scalar(right_));
}

template<typename TLeft, typename TRight>
inline BinaryOperator<Minus, TLeft, TRight>
operator - (const Expr<TLeft> &left_, const Expr<TRight> &right_)
{
    return BinaryOperator<Minus, TLeft, TRight>(left_, right_);
}

template<typename TLeft>
inline BinaryOperator<Minus, TLeft, Scalar>
operator - (const Expr<TLeft> &left_, T right_)
{
    return BinaryOperator<Minus, TLeft, Scalar>(left_, Scalar(right_));
}

template<typename TLeft, typename TRight>
inline BinaryOperator<Multi, TLeft, TRight>
operator * (const Expr<TLeft> &left_, const Expr<TRight> &right_)
{
    return BinaryOperator<Multi, TLeft, TRight>(left_, right_);
}

template<typename TLeft>
inline BinaryOperator<Multi, TLeft, Scalar>
operator * (const Expr<TLeft> &left_, T right_)
{
    return BinaryOperator<Multi, TLeft, Scalar>(left_, Scalar(right_));
}

template<typename TLeft, typename TRight>
inline BinaryOperator<Divide, TLeft, TRight>
operator / (const Expr<TLeft> &left_, const Expr<TRight> &right_)
{
    return BinaryOperator<Divide, TLeft, TRight>(left_, right_);
}

template<typename TLeft>
inline BinaryOperator<Divide, TLeft, Scalar>
operator / (const Expr<TLeft> &left_, T right_)
{
    return BinaryOperator<Divide, TLeft, Scalar>(left_, Scalar(right_));
}
/* evaluate */
template<typename TOperator, size_t N>
struct evaluate
{
    inline static void _(const Vector &x)
    {
        TOperator::apply(x[N]);
        return evaluate<TOperator, N - 1>::_(x);
    }
    inline static void __(const Vector &x1, const Vector &x2)
    {
        TOperator::apply(x1[N], x2[N]);
        return evaluate<TOperator, N - 1>::__(x1, x2);
    }
};
template<typename TOperator>
struct evaluate<TOperator, 0>
{
    inline static void _(const Vector &x){TOperator::apply(x[0]);}
    inline static void __(const Vector &x1, const Vector &x2)
    {
        return TOperator::apply(x1[0], x2[0]);
    }
};
template<size_t N>
struct Dot
{
    inline static T _(const Vector &x1, const Vector &x2)
    {
        return x1[N] * x2[N] + Dot<N - 1>::_(x1, x2);
    }
};

template<>
struct Dot<0>
{
    inline static T _(const Vector &x1, const Vector &x2){return x1[0] * x2[0];}
};

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
