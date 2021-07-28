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

  