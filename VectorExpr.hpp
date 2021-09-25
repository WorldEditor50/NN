#ifndef VECTOREXPR_HPP
#define VECTOREXPR_HPP
#include "allocator.hpp"
#include <iostream>

namespace VectorExp {

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

template <typename TExpr>
class VectorExpr
{
public:
    inline T operator[](size_t i) const {return static_cast<TExpr const &>(*this)[i];}
    inline TExpr const & value() const {return static_cast<TExpr const &>(*this);}
};

class Vector : public VectorExpr<Vector>
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
    template <typename TRight>
    Vector(VectorExpr<TRight> const &expr)
    {
        TRight const &r = expr.value();
        ptr = allocator.allocate(r.size_);
        size_ = r.size_;
        for (int i = 0; i < size_; i++) {
            ptr[i] = r.ptr[i];
        }
    }
    template <typename TRight>
    Vector& operator = (VectorExpr<TRight> const &expr)
    {
        TRight const & r = expr.value();
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
    template <typename TRight>
    Vector& operator = (VectorExpr<TRight> &&expr)
    {
        TRight const & r = expr.value();
        if (this == &r) {
            return *this;
        }
        ptr = r.ptr;
        size_ = r.size_;
        r.ptr = nullptr;
        r.size_ = 0;
        return *this;
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
template<typename T>
class VectorTrait
{
public:
    using ExpRef = T const&;
};

template<typename T>
class VectorTrait<Scalar<T> >
{
public:
    using ExpRef = Scalar<T>;
};

/* vector operation */
template<typename TLeft, typename TRight>
class VectorPlus : public VectorExpr<VectorPlus<TLeft, TRight> >
{
public:
    explicit VectorPlus(VectorExpr<TLeft> const &left_, VectorExpr<TRight> const &right_):left(left_), right(right_){}
    inline T operator[](size_t i) const {return left[i] + right[i];}
private:
    TLeft const &left;
    TRight const &right;
};

template<typename TLeft, typename TRight>
class VectorMinus : public VectorExpr<VectorMinus<TLeft, TRight> >
{
public:
    explicit VectorMinus(VectorExpr<TLeft> const &left_, VectorExpr<TRight> const &right_):left(left_), right(right_){}
    inline T operator[](size_t i) const {return left[i] - right[i];}
private:
    TLeft const &left;
    TRight const & right;
};

template<typename TLeft, typename TRight>
class VectorMulti : public VectorExpr<VectorMulti<TLeft, TRight> >
{
public:
    explicit VectorMulti(VectorExpr<TLeft> const &left_, VectorExpr<TRight> const &right_):left(left_), right(right_){}
    inline T operator[](size_t i) const {return left[i] * right[i];}
private:
    TLeft const &left;
    TRight const & right;
};

template<typename TLeft, typename TRight>
class VectorDivide : public VectorExpr<VectorDivide<TLeft, TRight> >
{
public:
    explicit VectorDivide(VectorExpr<TLeft> const &left_, VectorExpr<TRight> const &right_):left(left_), right(right_){}
    inline T operator[](size_t i) const {return left[i] / right[i];}
private:
    TLeft const &left;
    TRight const & right;
};

template<typename TLeft, typename TRight>
VectorPlus<TLeft, TRight> operator + (VectorExpr<TLeft> const &left_, VectorExpr<TRight> const &right_)
{
    return VectorPlus<TLeft, TRight>(left_, right_);
}

template<typename TLeft, typename TRight>
VectorMinus<TLeft, TRight> operator - (VectorExpr<TLeft> const &left_, VectorExpr<TRight> const &right_)
{
    return VectorMinus<TLeft, TRight>(left_, right_);
}

template<typename TLeft, typename TRight>
VectorMulti<TLeft, TRight> operator * (VectorExpr<TLeft> const &left_, VectorExpr<TRight> const &right_)
{
    return VectorMulti<TLeft, TRight>(left_, right_);
}

template<typename TLeft, typename TRight>
VectorDivide<TLeft, TRight> operator / (VectorExpr<TLeft> const &left_, VectorExpr<TRight> const &right_)
{
    return VectorDivide<TLeft, TRight>(left_, right_);
}

/* scale operation */
template<typename TLeft>
class VectorPlusScale : public VectorExpr<VectorPlusScale<TLeft> >
{
public:
    explicit VectorPlusScale(const VectorExpr<TLeft> &left_, T right_):left(left_), right(right_){}
    inline T operator[](size_t i) {return left[i] + right;}
private:
    TLeft& left;
    T right;
};
template<typename TRight>
class ScalePlusVector : public VectorExpr<ScalePlusVector<TRight> >
{
public:
    explicit ScalePlusVector(T left_, const VectorExpr<TRight> &right_):left(left_), right(right_){}
    inline T operator[](size_t i) {return left + right[i];}
private:
    T left;
    TRight& right;
};

template<typename TLeft>
class VectorMinusScale : public VectorExpr<VectorMinusScale<TLeft> >
{
public:
    explicit VectorMinusScale(const VectorExpr<TLeft> &left_, T right_):left(left_), right(right_){}
    inline T operator[](size_t i) {return left[i] - right;}
private:
    TLeft& left;
    T right;
};
template<typename TRight>
class ScaleMinusVector : public VectorExpr<ScaleMinusVector<TRight> >
{
public:
    explicit ScaleMinusVector(T left_, const VectorExpr<TRight> &right_):left(left_), right(right_){}
    inline T operator[](size_t i) {return left - right[i];}
private:
    T left;
    TRight& right;
};

template<typename TLeft>
class VectorMultiScale : public VectorExpr<VectorMultiScale<TLeft> >
{
public:
    explicit VectorMultiScale(const VectorExpr<TLeft> &left_, T right_):left(left_), right(right_){}
    inline T operator[](size_t i) {return left[i] * right;}
private:
    TLeft& left;
    T right;
};
template<typename TRight>
class ScaleMultiVector: public VectorExpr<ScaleMultiVector<TRight> >
{
public:
    explicit ScaleMultiVector(T left_, const VectorExpr<TRight> &right_):left(left_), right(right_){}
    inline T operator[](size_t i) {return left * right[i];}
private:
    T left;
    TRight& right;
};

template<typename TLeft>
class VectorDivideScale : public VectorExpr<VectorDivideScale<TLeft> >
{
public:
    explicit VectorDivideScale(const VectorExpr<TLeft> &left_, T right_):left(left_), right(right_){}
    inline T operator[](size_t i) {return left[i] / right;}
private:
    TLeft& left;
    T right;
};

template<typename TLeft>
VectorPlusScale<TLeft> operator + (const VectorExpr<TLeft> &left_, T right_)
{
    return VectorPlusScale<TLeft>(left_, right_);
}

template<typename TRight>
ScalePlusVector<TRight> operator + (T left_, const VectorExpr<TRight> &right_)
{
    return ScalePlusVector<TRight>(left_, right_);
}

template<typename TLeft>
VectorMinusScale<TLeft> operator - (const VectorExpr<TLeft> &left_, T right_)
{
    return VectorMinusScale<TLeft>(left_, right_);
}

template<typename TRight>
ScaleMinusVector<TRight> operator - (T left_, const VectorExpr<TRight> &right_)
{
    return ScaleMinusVector<TRight>(left_, right_);
}


template<typename TLeft>
VectorMultiScale<TLeft> operator * (const VectorExpr<TLeft> &left_, T right_)
{
    return VectorMultiScale<TLeft>(left_, right_);
}

template<typename TRight>
ScaleMultiVector<TRight> operator * (T left_, const VectorExpr<TRight> &right_)
{
    return ScaleMultiVector<TRight>(left_, right_);
}

template<typename TLeft>
VectorDivideScale<TLeft> operator / (const VectorExpr<TLeft> &left_, T right_)
{
    return VectorDivideScale<TLeft>(left_, right_);
}

}
#endif // VECTOREXPR_HPP
