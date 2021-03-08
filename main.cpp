#include "graph.hpp"
#include "mlp.hpp"

void test_DAG()
{
    std::cout<<"mlp"<<std::endl;
    MLP<double> mlp;
    /* add layer */
    std::cout<<"add layer"<<std::endl;
    mlp.addLayer(INPUT, SIGMOID, MSE, 4, 4, "input1");
    mlp.addLayer(INPUT, SIGMOID, MSE, 4, 4, "input2");
    mlp.addLayer(INPUT, SIGMOID, MSE, 4, 4, "input3");
    mlp.addLayer(INPUT, SIGMOID, MSE, 4, 4, "input4");
    mlp.addLayer(HIDDEN, SIGMOID, MSE, 8, "hidden1");
    mlp.addLayer(HIDDEN, SIGMOID, MSE, 8, "hidden2");
    mlp.addLayer(HIDDEN, SIGMOID, MSE, 8, "hidden3");
    mlp.addLayer(OUTPUT, SIGMOID, MSE, 4, "output");
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
    MLP<double>::InputParam x;
    x["input1"] = Mat<double>(4, 1);
    x["input2"] = Mat<double>(4, 1);
    x["input3"] = Mat<double>(4, 1);
    x["input4"] = Mat<double>(4, 1);
    mlp.feedForward(x);
    mlp.show();
    Mat<double> y(4, 1);
    std::cout<<"gradient"<<std::endl;
    mlp.gradient(x, y);
    std::cout<<"SGD"<<std::endl;
    mlp.SGD(0.01);
    mlp.show();
    std::cout<<"RMSProp"<<std::endl;
    mlp.RMSProp(0.9, 0.01);
    mlp.show();
    std::cout<<"Adam"<<std::endl;
    mlp.Adam(0.9, 0.99, 0.01);
    mlp.show();
    return;
}

void test_xor()
{
    std::cout<<"mlp"<<std::endl;
    using BPNN = MLP<double, true>;
    BPNN bp;
    /* add layer */
    std::cout<<"add layer"<<std::endl;
    bp.addLayer(INPUT, SIGMOID, MSE, 4, 2, "input");
    bp.addLayer(HIDDEN, SIGMOID, MSE, 4, "hidden1");
    bp.addLayer(HIDDEN, SIGMOID, MSE, 4, "hidden2");
    bp.addLayer(OUTPUT, SIGMOID, MSE, 1, "output");
    /* connect */
    std::cout<<"connect"<<std::endl;
    bp.connectLayer("input", "hidden1");
    bp.connectLayer("input", "hidden2");
    bp.connectLayer("input", "output");
    bp.connectLayer("hidden1", "output");
    bp.connectLayer("hidden2", "output");
    if (!bp.generate()) {
        std::cout<<"failed mlp is not DAG";
        return;
    }
    std::cout<<"topology:"<<std::endl;
    bp.showTopology();
    bp.show();
    /* train */
    std::cout<<"training:"<<std::endl;
    BPNN::InputParamVec x(4);
    for (int i = 0; i < 4; i++) {
        x[i]["input"] = Mat<double>(2, 1);
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
        y[i] = Mat<double>(1, 1);
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
        bp.RMSProp(0.9, 0.01);
    }
    /* classify */
    std::cout<<"classify"<<std::endl;
    for (int i = 0; i < 4; i++) {
        bp.feedForward(x[i]);
        bp.show();
    }
    return;
}

int main(int argc, char *argv[])
{
    srand((unsigned int)time(nullptr));
    test_xor();
    return 0;
}
