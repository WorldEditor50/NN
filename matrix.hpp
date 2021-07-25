#ifndef MATRIx_H
#define MATRIx_H
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <functional>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <memory>
namespace ML {

enum MatType{
    ZERO = 0,
    IDENTITY,
    UNIFORM_RAND
};
template<typename T>
class Mat
{
public:
    int rows;
    int cols;
    std::vector<std::vector<T> > data;
public:
	Mat():rows(0), cols(0){}
    ~Mat(){}
    inline bool isShapeEqual(const Mat<T>& x)const{return (rows == x.rows && cols == x.cols);}
    inline bool isNull() const {return rows == 0 || cols == 0;}
    inline bool isSquare()const {return rows == cols;}
    inline T& at(int row, int col) {return data[row][col];}
    std::vector<T>& operator[](int i){return data[i];}
    Mat<T>& create(int rows, int cols)
    {
        this->rows = rows;
        this->cols = cols;
        this->data = std::vector<std::vector<T> >(rows);
        for (int i = 0; i < rows; i++) {
            data[i] = std::vector<T>(cols, 0);
        }
        return *this;
    }
    void expand(std::function<void(int, int)> func)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                func(i, j);
            }
        }
        return;
    }
    void assign(const Mat<T>& x)
    {
        for (int i = 0; i < rows; i++) {
            data[i] = x.data[i];
        }
        return;
    }

    void assign(T x)
    {
        expand([this, x](int i, int j){data[i][j] = x;});
        return;
    }
    void identity()
    {
        if (!isSquare()) {
            return;
        }
        expand([this](int i, int j){data[i][j] = T(i == j);});
        return;
    }

    void random(int minValue, int maxValue)
    {
        expand([this, minValue, maxValue](int i, int j){
            data[i][j] = T(minValue + rand() % (maxValue - minValue));
        });
        return;
    }

    void uniformRandom()
    {
        expand([this](int i, int j){
            data[i][j] = T(rand() % 10000 - rand() % 10000) / 10000;
        });
        return;
    }

    Mat(int rows, int cols, MatType type = ZERO)
    {
        create(rows, cols);
        switch (type) {
        case ZERO:
            assign(0);
            break;
        case IDENTITY:
            identity();
            break;
        case UNIFORM_RAND:
            uniformRandom();
            break;
        default:
            break;
        }

    }

    Mat(const Mat<T>& x)
    {
        if (this == &x) {
            return;
        }
        create(x.rows, x.cols);
        assign(x);
    }

    Mat<T> operator = (const Mat<T>& x)
    {
        if (this == &x) {
            return *this;
        }
        if (isNull()) {
            create(x.rows, x.cols);
        }
        if (!isShapeEqual(x)) {
            std::cout<<"= size is not matched"<<std::endl;
            return *this;
        }
        assign(x);
        return *this;
    }

    void zero(){ assign(0);}

    std::vector<T> column(int col)
    {
        std::vector<T> columnT;
        expand([this, &columnT, col](int i, int j){
            if (j == col) {
                columnT.push_back(data[i][j]);
            }
        });
        return columnT;
    }

    std::vector<T> toVector()
    {
        std::vector<T> x;
        expand([this, &x](int i, int j){x.push_back(data[i][j]);});
        return x;
    }

    void show()
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout<<data[i][j]<<" ";
            }
            std::cout<<std::endl;
        }
        return;
    }

    Mat<T> operator + (const Mat<T>& x)
    {
        if (!isShapeEqual(x)) {
            std::cout<<"+ size is not matched"<<std::endl;
            return *this;
        }
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] + x.data[i][j];
            }
        }
        return y;
    }

    Mat<T> operator - (const Mat<T>& x)
    {
        if (!isShapeEqual(x)) {
            std::cout<<"- size is not matched"<<std::endl;
            std::cout<<"this row:"<<rows<<"  x row:"<<x.rows<<std::endl;
            std::cout<<"this col:"<<cols<<"  x col:"<<x.cols<<std::endl;
            return *this;
        }
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] - x.data[i][j];
            }
        }
        return y;
    }

    Mat<T> operator * (const Mat<T>& x)
    {
        if (cols != x.rows) {
            std::cout<<"* size is not matched"<<std::endl;
            return *this;
        }
        /* (m, p) x (p, n) = (m, n) */
        int m = rows;
        int p = cols;
        int n = x.cols;
        Mat<T> y(m, n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    y.data[i][j] += data[i][k] * x.data[k][j];
                }
            }
        }
        return y;
    }

    Mat<T> operator / (const Mat<T>& x)
    {
        if (!isShapeEqual(x)) {
            std::cout<<"/ size is not matched"<<std::endl;
            return *this;
        }
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                 y.data[i][j] = data[i][j] / x.data[i][j];
            }
        }
        return y;
    }

    Mat<T> operator % (const Mat<T>& x)
    {
        if (!isShapeEqual(x)) {
            std::cout<<"% size is not matched"<<std::endl;
            return *this;
        }
        Mat<T> y(rows, cols);
        for (int i = 0; i < y.rows; i++) {
            for (int j = 0; j < y.cols; j++) {
                y.data[i][j] = data[i][j] * x.data[i][j];
            }
        }
        return y;
    }

    Mat<T>& operator += (const Mat<T>& x)
    {
        if (!isShapeEqual(x)) {
            std::cout<<"+= size is not matched"<<std::endl;
            return *this;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += x.data[i][j];
            }
        }
        return *this;
    }

    Mat<T>& operator -= (const Mat<T>& x)
    {
        if (!isShapeEqual(x)) {
            std::cout<<"-= size is not matched"<<std::endl;
            return *this;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] -= x.data[i][j];
            }
        }
        return *this;
    }

    Mat<T>& operator /= (const Mat<T>& x)
    {
        if (!isShapeEqual(x)) {
            std::cout<<"/= size is not matched"<<std::endl;
            return *this;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] /= x.data[i][j];
            }
        }
        return *this;
    }

    Mat<T>& operator %= (const Mat<T>& x)
    {
        if (!isShapeEqual(x)) {
            std::cout<<"%= size is not matched"<<std::endl;
            return *this;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= x.data[i][j];
            }
        }
        return *this;
    }

    Mat<T> operator + (T x)
    {
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] + x;
            }
        }
        return y;
    }

    Mat<T> operator - (T x)
    {
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] - x;
            }
        }
        return y;
    }

    Mat<T> operator * (T x)
    {
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] * x;
            }
        }
        return y;
    }

    Mat<T> operator / (T x)
    {
        Mat<T> y(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[i][j] = data[i][j] / x;
            }
        }
        return y;
    }

    Mat<T>& operator += (T x)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += x;
            }
        }
        return *this;
    }

    Mat<T>& operator -= (T x)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] -= x;
            }
        }
        return *this;
    }

    Mat<T>& operator *= (T x)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= x;
            }
        }
        return *this;
    }

    Mat<T>& operator /= (T x)
    {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] /= x;
            }
        }
        return *this;
    }

    Mat<T> Tr()
    {
        Mat<T> y(cols,rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y.data[j][i] = data[i][j];
            }
        }
        return y;
    }

    Mat<T> subset(int fromRow, int fromCol, int rowOffset, int colOffset)
    {
        Mat<T> y(rowOffset, colOffset);
        int r = (fromRow + rowOffset > rows)?rows:(fromRow + rowOffset);
        int c = (fromCol + colOffset > cols)?cols:(fromCol + colOffset);
        for (int i = fromRow; i < r; i++) {
            for (int j = fromCol; j < c; j++) {
                y.data[i - fromRow][j - fromCol] = data[i][j];
            }
        }
        return y;
    }

    void set(int fromRow, int fromCol, const Mat<T> &x)
    {
        int r = (fromRow + x.rows > rows)?rows:(fromRow + x.rows);
        int c = (fromCol + x.cols > cols)?cols:(fromCol + x.cols);
        for (int i = fromRow; i < r; i++) {
            for (int j = fromCol; j < c; j++) {
                data[i][j] = x.data[i - fromRow][j - fromCol];
            }
        }
        return;
    }
    void save(const std::string& fileName)
    {
        std::ofstream file;
        file.open(fileName, std::ofstream::app);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file << data[i][j]<<" ";
            }
            file << std::endl;
        }
        return;
    }

    void load(const std::string& fileName)
    {
        std::ifstream file;
        file.open(fileName);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file >> data[i][j];
            }
        }
        return;
    }
};
template<typename T>
Mat<T> for_each(const Mat<T>& x, std::function<double(double)> func)
{
    Mat<T> y(x.rows, x.cols);
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            y.data[i][j] = func(x.data[i][j]);
        }
    }
    return y;
}

template<typename T>
T sum(Mat<T>& x)
{
    T s = 0;
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            s += x.data[i][j];
        }
    }
    return s;
}

template<typename T>
T max(Mat<T>& x)
{
    T maxT = x.data[0][0];
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            if (maxT < x.data[i][j]) {
                maxT = x.data[i][j];
            }
        }
    }
    return maxT;
}

template<typename T>
T min(Mat<T>& x)
{
    T minT = x.data[0][0];
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            if (minT > x.data[i][j]) {
                minT = x.data[i][j];
            }
        }
    }
    return minT;
}
template <typename T>
Mat<T> Kronecker(const Mat<T> &x1, const Mat<T> &x2)
{
    int rows = x1.rows * x2.rows;
    int cols = x1.cols * x2.cols;
    Mat<T> y(rows, cols);
    for (int i = 0; i < x1.rows; i++) {
        for (int j = 0; j < x1.cols; j++) {
            for (int h = 0; h < x2.rows; h++) {
                for (int k = 0; k < x2.cols; k++) {
                    y.data[i * x2.rows + h][j * x2.cols + k] = x1.data[i][j] * x2.data[h][k];
                }
            }
        }
    }
    return y;
}
inline double sigmoid(double x){return exp(x) / (exp(x) + 1);}
inline double relu(double x){return x > 0 ? x : 0;}
inline double linear(double x){return x;}
inline double dsigmoid(double y){return y * (1 - y);}
inline double drelu(double y){return y > 0 ? 1 : 0;}
inline double dtanh(double y){return 1 - y * y;}
inline double dlinear(double){return 1;}
template <typename T>
Mat<T> LOG(const Mat<T> &x){return for_each(x, static_cast<double(*)(double)>(log));}
template <typename T>
Mat<T> EXP(const Mat<T> &x){return for_each(x, static_cast<double(*)(double)>(exp));}
template <typename T>
Mat<T> SQRT(const Mat<T> &x){return for_each(x, static_cast<double(*)(double)>(sqrt));}

template <typename T>
class Sigmoid {
public:
    static Mat<T> _(const Mat<T> &x){return for_each(x, sigmoid);}
    static Mat<T> d(const Mat<T> &y){return for_each(y, dsigmoid);}
};
template <typename T>
class Relu {
public:
    static Mat<T> _(const Mat<T> &x){return for_each(x, relu);}
    static Mat<T> d(const Mat<T> &y){return for_each(y, drelu);}
};
template <typename T>
class Tanh {
public:
    static Mat<T> _(const Mat<T> &x){return for_each(x, static_cast<double(*)(double)>(tanh));}
    static Mat<T> d(const Mat<T> &y){return for_each(y, dtanh);}
};
template <typename T>
class Linear {
public:
    static Mat<T> _(const Mat<T> &x){return x;}
    static Mat<T> d(const Mat<T> &x){Mat<T> y(x); y.assign(1); return y;}
};

template <typename T>
Mat<T> SOFTMAX(Mat<T>& x)
{
    T maxValue = max(x);
    Mat<T> delta = EXP(x - maxValue);
    T s = sum(delta);
    return delta / (s + 1e-9);
}
}
#endif // MATRIx_H
