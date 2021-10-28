# include <iostream>
# include <string>
# include <cstring>
# include <vector>
# include <new>
# include <memory>
# include <map>
# include <unordered_set>
# include <ranges>

using namespace std;
/*
std::shared_ptr<int> fun()
{
    std::shared_ptr<int>(new int (3));
    return res;
}

void fun(int* ptr)
{
    std::cout << "call deleter fun" <<endl;
    delete ptr;
}

int main()
{
    std::shared_ptr<int> x = fun();
    std::shared_ptr<int> x(new int(3), fun);
}
*/

int main()
{
    /*int* ptr = new int(3);
    int* ptr1 = new int[5];
    int a = 4;
    std::cout << sizeof(ptr) <<endl; //8
    std::cout << sizeof(ptr1) <<endl; //8
    std::cout << sizeof(a) <<endl; //4

    allocator<int> al;
    int* ptr2 = al.allocate(3); //只返回内存，不会构造
    
    std::vector<int> a{1};
    std::vector<int> b{1, 0, 0};
    int v = -1;

    cout << (a < b) <<endl;
    //cout << v;
    
    std::map<int, bool> m{{3,true}, {4,false},{1,true}};
    for (auto p : m){
        cout << p.first << ' ' <<p.second <<endl; 
    }
    std::unordered_set<int> s{3, 1, 5, 4, 1};
    for (auto p : s){
        cout << p << endl;
    }

    int y = 10;
    int z = 3;
    auto x = [y+z] (int val) mutable{
        ++y;
        return val > y;
        };
    std::cout << x(5) <<std::endl;
    auto lam = []<typename T>(T val) {return val +1;};*/
    std::vector<int> x{1,2,3, 4, 5};
    auto it = std::ranges::find(x, 3);
    std::cout << *it << std::endl;


    return 0;

}













/*int main()
{
    
    int* y = new int(2);
    cout << *y <<endl;
    delete y;
    int* y = new(std::nothrow) int[5];
    delete[] y;
    char ch[sizeof(int)];
    int* y = new (ch) int(4);

    int* x = 0;
    delete x;

    int* ptr = new int[5];
    int* ptr2 = (ptr + 1)
    delete[] ptr2; //error

    std::shared_ptr<int> x (new int(3));





}*/