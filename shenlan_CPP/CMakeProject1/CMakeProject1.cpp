// CMakeProject1.cpp: 定义应用程序的入口点。
//

#include "CMakeProject1.h"

using namespace std;

string jia(string str1, string str2, int x) {
    int len = max(str1.size(), str2.size());
    string result;
    reverse(str1.begin(), str1.end());
    reverse(str2.begin(), str2.end());
    int jinwei = 0;
    for (int i = 0; i < len; ++i) {
        int str1_i = i < str1.size() ? str1[i] - '0' : 0;
        int str2_i = i < str2.size() ? str2[i] - '0' : 0;
        int val = (str1_i + str2_i + jinwei) % x;
        jinwei = (str1_i + str2_i + jinwei) / x;
        result.insert(result.begin(), val + '0');
    }
    if (jinwei == 1) result.insert(result.begin(), '1');
    if (result[0] == '0') result[0] = ' ';
    return result;
}
string jian(string str1, string str2, int x) {
    bool pf = false;
    if (str1 < str2)
    {
        string mid = str1;
        str1 = str2;
        str2 = mid;
        pf = true;
    }
    int len = max(str1.size(), str2.size());
    string result;
    reverse(str1.begin(), str1.end());
    reverse(str2.begin(), str2.end());
    int jiewei_ed = 0;
    for (int i = 0; i < len; ++i) {
        int jiewei = 0;
        int str1_i = i < str1.size() ? str1[i] - '0' : 0;
        int str2_i = i < str2.size() ? str2[i] - '0' : 0;
        if (str1_i < str2_i) jiewei = 1;
        str1_i -= jiewei_ed;
        int val = str1_i + (jiewei * x) - str2_i;
        result.insert(result.begin(), val + '0');
        (jiewei == 1) ? (jiewei_ed = 1) : (jiewei_ed = 0);
    }
    if (result[0] == '0') result[0] = ' ';
    if (pf) result.insert(result.begin(), '-');
    //cout << pf << endl;
    return result;
}

bool panduan(string x, int y)
{
    int len = x.size();
    for (int i = 0; i < len; ++i)
    {
        int temp = (int)x[i];
        int thre_1 = y - 10 + 97;  //小写字母
        int thre_2 = y - 10 + 65;  //大写字母
        if (i == 0){
            if (x[i] == '+' || x[i] == '-' || (temp >= 48 && temp <= 57)) continue;
            else if (y > 10){  
                if ((temp >= 48 && temp <= 57) || (temp >= 97 && temp <= thre_1) || (temp >= 65 && temp <= thre_2)) continue;
                else ;
                }
            else {
                return 1;
                break;
            }
        }
        else{
            if (y <= 10) {
                if (temp >= 48 && temp <= 57) continue;
                else {
                    return 1;
                    break;
                }
            }
            else if (y > 10) {
                if (temp >= 48 && temp <= 57) continue;
                else if (temp >= 97 && temp <= thre_1) continue;
                else if (temp >= 65 && temp <= thre_2) continue;
                else {
                    return 1;
                    break;
                }
            }
        }
        
    }
    return 0;
}

int main()
{
    int jinzhi;
    cout << "Please enter input-calculation base(2~36):   " << endl;
    cin >> jinzhi;
    while (cin.fail() || jinzhi < 2 || jinzhi > 36)//检验输入是否正确
    {
        cin.clear();
        cin.sync();
        while (cin.get() != '\n') {
            continue;
        }
        cout << "Please enter an integer number in the range [2,36] :  " << endl;
        cin >> jinzhi;
    }
    string a;
    cout << "please input a long number a:  " << endl;
    cin >> a;
    while (panduan(a, jinzhi))//检验输入是否正确
    {
        cin.clear();
        cin.sync();
        while (cin.get() != '\n') {
            continue;
        }
        cout << "Error! Please input again :   " << endl;
        cin >> a;
    }
    string b;
    cout << "please input a long number b:  " << endl;
    cin >> b;
    while (panduan(b, jinzhi))//检验输入是否正确
    {
        cin.clear();
        cin.sync();
        while (cin.get() != '\n') {
            continue;
        }
        cout << "Error! Please input again :   " << endl;
        cin >> b;
    }
    string re_;
    //cout << a << "  " << b << endl;
    if (a[0] == '-'){
        a[0] = '0';
        if (b[0] == '-'){
            b[0] = '0';   
            re_ = jia(a, b, jinzhi);
            re_.insert(re_.begin(), '-');
        }
        else if (b[0] == '+'){
            b[0] = '0';
            re_ = jian(b, a, jinzhi);
        }
        else {
            b.insert(b.begin(), '0');
            re_ = jian(b, a, jinzhi);
        }      
    }
    else if (a[0] == '+'){
        a[0] = '0';
        if (b[0] == '-'){
            b[0] = '0';
            re_ = jian(a, b, jinzhi);
        }
        else if (b[0] == '+'){
            b[0] = '0';
            re_ = jia(a, b, jinzhi);
        }
        else{
            b.insert(b.begin(), '0');
            re_ = jia(a, b, jinzhi);
        }
    }
    else{
        a.insert(a.begin(), '0');
        if (b[0] == '-'){
            b[0] = '0';
            re_ = jian(a, b, jinzhi);
        }
        else if (b[0] == '+'){
            b[0] = '0';
            re_ = jia(a, b, jinzhi);
        }
        else re_ = jia(a, b, jinzhi);
    }
    int len = re_.size();
    for (int i = 0; i < len; ++i){
        //int temp = (int)re_[i];
        if (re_[i] == '+' || re_[i] == '-'|| re_[i] == ' ') continue;
        else if (re_[i] == '0') re_[i] = ' ';
        else break;
    }
    cout << "the answer is: " << endl;
    cout << re_ << endl;
    system("pause");
    return 0;
}