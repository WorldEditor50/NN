## C++ Note

### 1. 构造函数

- **尽量不要要在构造函数内进行数据初始化，否则影响内存分配**

- 复杂结构的类必须实现拷贝构造、赋值构造、移动构造函数

- 使用初始化列表初始化成员数据

- 如果存在父类，则在初始化列表调用父类构造函数

- 存在类内存在内存分配最好定义移动构造函数

  ```c++
  class Foo
  {
  public:
      int *ptr;
      int size_;
  public:
      Foo():ptr(nullptr), size_(0){}
      /* 使用初始化列表初始化成员数据 */
      Foo(const Foo& r):ptr(new int[r.size_]), size_(r.size_){}
      Foo& operator=(const Foo& r)
      {
          if (this == &r) {
              return *this;
          }
          if (ptr == nullptr) {
              ptr = new int[r.size_];
              size_ = r.size_;
          }
          int len = size_ > r.size_ ? size_ : r.size_;
          for (int i = 0; i < len; i++) {
              ptr = r.ptr[i];
          }
          return *this;
      }
      /* 移动构造 */
      Foo(Foo&& r):ptr(r.ptr), size_(r.size_)
      {
          r.ptr = nullptr;
          r.size_ = 0;
      }
      /* 赋值移动构造 */
      Foo& operator=(Foo&& r)
      {
          if (this == &r) {
              return *this;
          }
          ptr = r.ptr;
          size_ = r.size_;
          r.ptr = nullptr;
          r.size_ = size_;
          return *this;
      }
  };
  class Boo : public Foo
  {
  public:
      Boo(){}
      /* 调用父类拷贝构造函数 */
      Boo(const Boo &r):Foo(r){}
      Boo& operator=(const Boo& r)
      {
          if (this == &r) {
              return *this;
          }
          /* 调用父类赋值构造函数 */
  		Foo::operaor=(r);
          return *this;
      }
  }
  ```

  

  


### 2. 类型转换

- **static_cast**

  - 选择重载成员函数

    ```c++
    class Foo
    {
    public:
    	Foo(){}
        void set(int x){}
        void set(int x, int y){}
    };
    void test(std::function<int(int)> func){}
    int main()
    {
        test(static_cast<int Foo::(*)(int)>(Foo::set));
        return 0;
    }
    ```

    

- **dynamic_cast**



- **reinterpret_cast**



- **const_cast**

  



### 3. inline的作用

- **内联**

  ## 内联函数与一般函数区别

  1）内联含函数比一般函数在前面多一个inline修饰符。

  2）内联函数是直接复制“镶嵌”到主函数中去的，就是将内联函数的代码直接放在内联函数的位置上，这与一般函数不同，主函数在调用一般函数的时候，是指令跳转到被调用函数的入口地址，执行完被调用函数后，指令再跳转回主函数上继续执行后面的代码；而由于内联函数是将函数的代码直接放在了函数的位置上，所以没有指令跳转，指令按顺序执行。

  3）一般函数的代码段只有一份，放在内存中的某个位置上，当程序调用它时，指令就跳转过来；当下一次程序调用它是，指令又跳转过来；而内联函数是程序中调用几次内联函数，内联函数的代码就会复制几份放在对应的位置上

  4）内联函数一般在头文件中定义，而一般函数在头文件中声明，在cpp中定义。

  ```c++
  /*
  有inline关键字优化的函数类似于宏，函数调用时可以减少栈空间的使用，但函数是否会被内联有编译器决定
  */
  #define ADD(a, b) (a + b)
  template<typename T>
  inline T add(T a, T b) { return a + b;}
  ```

  

- **避免多重定义**

  ```c++
     /*
     	如果不加inline关键字，多个地方调用append会引起多重定义的编译错误，有inline关键字修饰的
      函数允许存在多个同名实例而不会报多重定义，因为内联函数的汇编代码会被嵌入调用处所在的函数
      */
  	template <typename T>
      inline void append(T t, std::string &dst) { dst += std::to_string(t);}
      inline void append(const std::string& t, std::string &dst) { dst += t;}
      inline void append(const char* t, std::string &dst) { dst += t;}
      template <typename ...T>
      inline std::string append(const T& ...t)
      {
          std::string dst;
          int argv[] = {(append(t, dst), 0)...};
          return dst;
      }
  ```

  

### 4. 表达式模板

#### 4.1 简述

表达式模板是一种使用类模板保存运算过程计算图的数值优化方法，优点如下：

- 所有的计算操作会集中统一处理
- 减少中间变量的使用
- 重载运算符简化代码表达
- 惰性计算

#### 4.2 实现

- 表达式封装

  ```c++
  template<typename TExprImpl>
  class Expr
  {
  public:
      TExprImpl& impl() {return static_cast<TExprImpl&>(*this);}
  };
  ```

  

- 元运算模板

- 运算符重载

- 表达式兼容

- 内存管理

  