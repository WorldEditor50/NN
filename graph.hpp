#ifndef GRAPH_H
#define GRAPH_H
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <map>

class Edge
{
public:
    int from;
    int to;
    double weight;
    bool enable;
public:
    Edge():from(0), to(0), weight(0), enable(false){};
    ~Edge(){};
    Edge(int from, int to, double weight)
    {
        this->from = from;
        this->to = to;
        this->weight = weight;
        this->enable = true;
    }
};
template<typename T>
class Vertex
{
public:
    bool visited;
    int indegree;
    std::string name;
    T object;
public:
    Vertex():visited(false), indegree(0), name(""){}
    ~Vertex(){}
    Vertex(const T & obj, const std::string &vertexName):
         visited(false), indegree(0), name(vertexName), object(obj){}
    Vertex(const Vertex & v)
    {
        this->visited = v.visited;
        this->indegree = v.indegree;
        this->name = v.name;
        this->object = v.object;
    }
};
template<typename T>
class Graph
{
public:
    using DataType = T;
public:
    std::vector<Vertex<T> > vertexs;
    std::vector<Edge> edges;
    std::vector<int> topologySequence;
    std::vector<int> traversalSequence;
    std::map<int, std::vector<int> > previous;
    std::map<int, std::vector<int> > nexts;
public:
    Graph(){}
    ~Graph(){}
    inline bool isDAG(){return (vertexs.size() == topologySequence.size());}
    inline bool isEmpty(){return vertexs.size() < 2;}
    inline T& getObject(int index){return vertexs.at(index).object;}
    void copy(const Graph<T> &graph)
    {
        vertexs = graph.vertexs;
        edges = graph.edges;
        topologySequence = graph.topologySequence;
        traversalSequence = graph.traversalSequence;
        previous = graph.previous;
        nexts = graph.nexts;
    }
    Graph(const Graph<T> &graph)
    {
        copy(graph);
    }

    Graph<T>& operator = (const Graph<T> &graph)
    {
        if (this == &graph) {
            return *this;
        }
        copy(graph);
        return *this;
    }

    int findVertex(const std::string &name)
    {
        int index = -1;
        for (int i = 0; i < vertexs.size(); i++) {
            if (vertexs[i].name == name) {
                index = i;
                break;
            }
        }
        return index;
    }

    void insertVertex(const T& obj, const std::string &vertexName)
    {
        vertexs.push_back(Vertex<T>(obj, vertexName));
        return;
    }

    void insertEdge(int from, int to, double weight = 1)
    {
        if (vertexs.size() < 2) {
            return;
        }
        if (from >= vertexs.size() || to >= vertexs.size()) {
            std::cout<<"index out of range"<<std::endl;
            return;
        }
        edges.push_back(Edge(from, to, weight));
        vertexs[to].indegree++;
        return;
    }

    void insertEdge(const std::string &fromName, const std::string &toName, double weight = 1)
    {
        if (vertexs.size() < 2) {
            return;
        }
        int from = findVertex(fromName);
        int to = findVertex(toName);
        if (from < 0 || to < 0) {
            std::cout<<"invalid vertex name"<<std::endl;
            return;
        }
        if (from >= vertexs.size() || to >= vertexs.size()) {
            std::cout<<"index out of range"<<std::endl;
            return;
        }
        edges.push_back(Edge(from, to, weight));
        vertexs[to].indegree++;
        return;
    }

    std::vector<int> findNext(int index)
    {
        std::vector<int> nextIndex;
        for (auto x : edges) {
            if (x.from == index) {
                nextIndex.push_back(x.to);
            }
        }
        return nextIndex;
    }

    std::vector<int> findPrevious(int index)
    {
        std::vector<int> preIndex;
        for (auto x : edges) {
            if (x.to == index) {
                preIndex.push_back(x.from);
            }
        }
        //std::cout<<preIndex.size()<<std::endl;
        return preIndex;
    }

    bool generate()
    {
        for (int i = 0; i < vertexs.size(); i++) {
            nexts[i] = findNext(i);
            previous[i] = findPrevious(i);
        }
        return toposort();
    }

    void BFS(int index)
    {
        if (index > vertexs.size()) {
            return;
        }
        /* clear */
        clearVisit();
        /* visit */
        std::queue<int> indexQueue;
        indexQueue.push(index);
        vertexs[index].visited = true;
        traversalSequence.push_back(index);
        while (!indexQueue.empty()) {
            int index = indexQueue.front();
            indexQueue.pop();
            for (auto it = edges.begin(); it != edges.end(); it++) {
                if (it->from == index && vertexs[it->to].visited == false) {
                    /* visit */
                    vertexs[it->to].visited = true;
                    indexQueue.push(it->to);
                    traversalSequence.push_back(it->to);
                }
            }
        }
        return;
    }

    void DFS(int index)
    {
        if (index > vertexs.size()) {
            return;
        }
        /* clear */
        clearVisit();
        /* visit */
        std::stack<int> indexStack;
        vertexs[index].visited = true;
        traversalSequence.push_back(index);
        indexStack.push(index);
        indexStack.push(index);
        while (!indexStack.empty()) {
            int index = indexStack.top();
            indexStack.pop();
            for (auto it = edges.begin(); it != edges.end(); it++) {
                if (it->from == index && vertexs[it->to].visited == false) {
                    /* visit */
                    vertexs[it->to].visited = true;
                    traversalSequence.push_back(it->to);
                    indexStack.push(it->to);
                    /* transit */
                    index = it->to;
                    it = edges.begin();
                }
            }
        }
        return;
    }

    void RDFS(int index)
    {
        traversalSequence.push_back(index);
        vertexs[index].visited = true;
        for (auto it = edges.begin(); it != edges.end(); it++) {
            if (it->from == index) {
                if (vertexs[it->to].visited == false) {
                    RDFS(it->to);
                }
            }
        }
        return;
    }

    bool toposort()
    {
        /* clear */
        topologySequence.clear();
        /* clone index */
        std::vector<int> indegrees(vertexs.size());
        /* indegree */
        std::queue<int> indegreeQueue;
        for (int i = 0; i < vertexs.size(); i++) {
            if (vertexs[i].indegree == 0) {
                indegreeQueue.push(i);
                topologySequence.push_back(i);
            }
            indegrees[i] = vertexs[i].indegree;
        }
        /* sort */
        while (!indegreeQueue.empty()) {
            int index = indegreeQueue.front();
            indegreeQueue.pop();
            for (auto it = edges.begin(); it != edges.end(); it++) {
                if (it->from == index) {
                    indegrees[it->to]--;
                    if (indegrees[it->to] == 0) {
                        indegreeQueue.push(it->to);
                        topologySequence.push_back(it->to);
                    }
                }
            }
        }
        return topologySequence.size() == vertexs.size();
    }

    void clearVisit()
    {
        for (auto x : vertexs) {
            x.visited = false;
        }
        traversalSequence.clear();
        return;
    }
    void showTopology()
    {
        for (auto i : topologySequence) {
            std::cout<<"index: "<<i<<"  name: "<<vertexs[i].name<<std::endl;
        }
        return;
    }
};

#endif// GRAPH_H
