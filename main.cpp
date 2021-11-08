#include "mlp.hpp"
#include "lstm.hpp"
#include "expression.hpp"
#include "Vector.hpp"
#include "VectorExpr.hpp"
#include <chrono>

using namespace ML;

void test_DAG()
{
    std::cout<<"mlp"<<std::endl;
    MLP<double, Sigmoid, Adam> mlp;
    /* add layer */
    std::cout<<"add layer"<<std::endl;
    mlp.addLayer(INPUT, MSE, 4, 4, "input1");
    mlp.addLayer(INPUT, MSE, 4, 4, "input2");
    mlp.addLayer(INPUT, MSE, 4, 4, "input3");
    mlp.addLayer(INPUT, MSE, 4, 4, "input4");
    mlp.addLayer(HIDDEN, MSE, 8, "hidden1");
    mlp.addLayer(HIDDEN, MSE, 8, "hidden2");
    mlp.addLayer(HIDDEN, MSE, 8, "hidden3");
    mlp.addLayer(OUTPUT, MSE, 4, "output");
    /* connection */
    std::cout<<"connect"<<std::endl;
    mlp.connectLayer("input1", "hidden1");
    mlp.connectLayer("input2", "hidden1");
    mlp.connectLayer("input3", "hidden2");
    mlp.connectLayer("input4", "hidden2");
    mlp.connectLayer("input2", "hidden3");
    mlp.connectLayer("input3", "hidden3");
    mlp.connectLayer("hidden1", "hidden3");
    mlp.connectLayer("hidden2", "hidden3");
    mlp.connectLayer("hidden3", "output");
    if (!mlp.generate()) {
        std::cout<<"failed to genarate graph";
        return;
    }
    /* topology */
    std::cout<<"topology"<<std::endl;
    mlp.showTopology();

    /* feed forward */
    std::cout<<"feed forward"<<std::endl;
    MLP<double,Sigmoid, Adam>::Input x;
    x["input1"] = Mat<double>(4, 1);
    x["input2"] = Mat<double>(4, 1);
    x["input3"] = Mat<double>(4, 1);
    x["input4"] = Mat<double>(4, 1);
    mlp.feedForward(x);
    mlp.show();
    Mat<double> y(4, 1);
    std::cout<<"gradient"<<std::endl;
    mlp.gradient(x, y);
    std::cout<<"Adam"<<std::endl;
    mlp.optimize(0.01);
    mlp.show();
    return;
}

void test_xor()
{
    /*
                  +-->-hidden1-->--+
                  |                |
        input -->-|----->----------|-->-- output
                  |                |
                  +-->-hidden2-->--+
    */
    std::cout<<"mlp"<<std::endl;
    using BPNN = MLP<float, Relu, Adam>;
    BPNN bp(BPNN::LayerParams {
                {INPUT, MSE, 4, 2, "input"},
                {HIDDEN, MSE, 4, 1, "hidden1"},
                {HIDDEN, MSE, 4, 1, "hidden2"},
                {OUTPUT, MSE, 1, 1, "output"}
            },
            BPNN::GraphParams {
                {"input", "hidden1"},
                {"input", "hidden2"},
                {"input", "output"},
                {"hidden1", "output"},
                {"hidden2", "output"}
            });
    std::cout<<"topology:"<<std::endl;
    bp.showTopology();
    bp.show();
    /* train */
    std::cout<<"training:"<<std::endl;
    BPNN::InputVec x(4);
    for (int i = 0; i < 4; i++) {
        x[i]["input"] = Mat<BPNN::DataType>(2, 1);
    }
    x[0]["input"][0][0] = 0;
    x[0]["input"][1][0] = 0;
    x[1]["input"][0][0] = 1;
    x[1]["input"][1][0] = 0;
    x[2]["input"][0][0] = 0;
    x[2]["input"][1][0] = 1;
    x[3]["input"][0][0] = 1;
    x[3]["input"][1][0] = 1;
    BPNN::Target y(4);
    for (int i = 0; i < 4; i ++) {
        y[i] = Mat<BPNN::DataType>(1, 1);
    }
    y[0][0][0] = 0;
    y[1][0][0] = 1;
    y[2][0][0] = 1;
    y[3][0][0] = 0;
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 4; j++) {
            int k = rand() % 4;
            bp.feedForward(x[k]);
            bp.gradient(x[k], y[k]);
        }
        bp.optimize(0.0005);
    }
    /* classify */
    std::cout<<"classify"<<std::endl;
    for (int i = 0; i < 4; i++) {
        bp.feedForward(x[i]);
        bp.show();
    }
    /* clone */
    std::cout<<"clone:"<<std::endl;
    BPNN::Flat predictNet = bp.clone();
    predictNet.showTopology();
    for (int i = 0; i < 4; i++) {
        predictNet.feedForward(x[i]);
        predictNet.show();
    }
    return;
}

template <typename TExpr>
void evaluate(const Exp::Expr<TExpr> &f)
{
    for (float x = 0; x < 100; x += 1.0) {
        std::cout<<f(x)<<" ";
    }
    std::cout<<std::endl;
    return;
}
struct Show {
    inline static void apply(double x)
    {
        std::cout<<x<<" ";
    }
};
void testVectorExpr()
{
    //test_xor();
    /* expression */
//    Exp::Var x;
//    evaluate(x * x - x + Exp::Const(2));
    const size_t N = 10;
    Vector<double> x1(N, 5);
    Vector<double> x2(N, 7);
    auto start = std::chrono::system_clock::now();
    //auto x3 = x1/x2  + x2/x1 + x1 * x1 - x2*x2 + x1 * x2;
    auto x3 = x1*5 + x2/7 + 12;
    x3.show();
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double t1 = (double(duration.count()) *
                         std::chrono::microseconds::period::num /
                         std::chrono::microseconds::period::den);

    VectorExpr::Vector u(N, 2);
    VectorExpr::Vector v(N, 3);
    double s = VectorExpr::Dot<N - 1>::_(u, v);
    std::cout<<"dot product:"<<s<<std::endl;
    auto start2 = std::chrono::system_clock::now();
    //VectorExpr::Vector z = u/v + v/u + u*u - v*v + u*v;
    VectorExpr::Vector z = u*2 + v/3 + 8;
    VectorExpr::evaluate<Show, N - 1>::_(z);
    auto end2 =  std::chrono::system_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    double t2 = (double(duration2.count()) *
                         std::chrono::microseconds::period::num /
                         std::chrono::microseconds::period::den);
    std::cout<<std::endl;
    std::cout<<"vector cost:"<<t1<<"s"<<std::endl;
    std::cout<<"vecotor expression cost:"<<t2<<"s"<<std::endl;
}

void test_lstm()
{
    auto zeta = [](double x, double y) ->double {
        return sin(x*x + y*y);
    };
    using Lstm = LSTM<2, 4, 1>;
    Lstm lstm;
    std::vector<Mat<double> > data;
    std::vector<Mat<double> > target;
    for (int i = 0; i < 1000; i++) {
        Mat<double> p(2, 1);
        double x = double(rand() % 1000) / 1000;
        double y = double(rand() % 1000) / 1000;
        double z = zeta(x, y);
        p[0][0] = x;
        p[0][1] = y;
        Mat<double> q(1, 1);
        q[0][0] = z;
        data.push_back(p);
        target.push_back(q);
    }
    auto sample = [&](std::vector<Mat<double> > &batchData,
            std::vector<Mat<double> > &batchTarget, int batchSize){
        for (int i = 0; i < batchSize; i++) {
            int k = rand() % data.size();
            batchData.push_back(data[k]);
            batchTarget.push_back(target[k]);
        }
    };
    for (int i = 0; i < 1000; i++) {
        std::vector<Mat<double> > batchData;
        std::vector<Mat<double> > batchTarget;
        sample(batchData, batchTarget, 32);
        lstm.forward(batchData);
        lstm.gradient(batchData, batchTarget);
        lstm.RMSProp(0.9, 0.01);
    }
    for (int i = 0; i < 10; i++) {
        Mat<double> p(2, 1);
        double x = double(rand() % 1000) / 1000;
        double y = double(rand() % 1000) / 1000;
        double z = zeta(x, y);
        p[0][0] = x;
        p[0][1] = y;
        std::cout<<"x = "<<x<<" y = "<<y<<" z = "<<z<<std::endl;
        std::cout<<"predict:"<<std::endl;
        lstm.feedForward(p).show();
    }
    return;
}
int main()
{
    srand((unsigned int)time(nullptr));
    test_lstm();
    return 0;
}
