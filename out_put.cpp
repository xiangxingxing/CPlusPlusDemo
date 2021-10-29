//
// Created by DEV on 2021/10/27.
//

#include "iostream"
#include "iomanip"

#include "out_put.h"

using namespace std;

// :: 为范围解析运算符，在此之前必须使用类名
void out_put::BasicUserInputDemo() {
    std::cout << "Hello, World!" << std::endl;
    char name[50];
    cout << "请输入名称:  ";
    cin >> name;
    cout << "名称为:" << name <<endl;
};

void out_put::BasicOutputFormatDemo() {
    cout<<setiosflags(ios::left|ios::showpoint);//设左对齐，以一般实数方式显示
    cout.precision(5);//5位有效数字
    cout<<123.45678910<<endl;//123.46

    cout.width(10);          // 设置显示域宽10
    cout.fill('*');          // 在显示区域空白处用*填充
    cout<<resetiosflags(ios::left);  // 清除状态左对齐
    cout<<setiosflags(ios::right);   // 设置右对齐
    cout<<123.45678910<<endl;//****123.46

    cout<<setiosflags(ios::left|ios::fixed);    // 设左对齐，以固定小数位显示
    cout.precision(3);    // 设置实数显示三位小数
    cout<<999.456789<<endl;//999.457

    cout<<resetiosflags(ios::left|ios::fixed);  //清除状态左对齐和定点格式
    cout<<setiosflags(ios::left|ios::scientific);    //设置左对齐，以科学技术法显示
    cout.precision(3);   //设置保留三位小数
    cout<<123.45678910<<endl;//1.235e+02
}

//初始化静态成员变量
int out_put::m_total = -1;

void out_put::StringDemo() {
//    string s1;
//    cout<<"s1 = "<<s1<<endl;
//    string s2 = "c plus plus";
//    cout<<"s2 = "<<s2<<endl;
//    string s3 = s2;
//    cout<<"s2 == s3 => "<<(s2 == s3)<<endl;
//    string s4(5, 's');
//    cout<<"s4 = "<<s4<<endl;
//
//    string s5 = "http://c.biancheng.net";
//    int len = s5.length();//返回的真实长度，而不是+1
//    cout<<len<<endl;

    //拼接
    string s1 = "first ";
    string s2 = "second ";
    string s3 = "third ";
    char s4[] = "fourth ";
    char ch = '@';
    string s5 = s1 + s2;
    string s6 = s1 + s3;
    string s7 = s1 + s4;
    string s8 = s1 + ch;

    cout<<"s1 + s2 = "<<s5<<endl<<"s1 + s3 = "<<s6<<endl<<"s1 + s4 = "<<s7<<endl<<"s1 + ch = "<<s8<<endl;

//    string path = "D:\\demo.txt";
//    FILE *fp = fopen(path.c_str(), "rt");//string to C_str
}

void out_put::StringOperationDemo() {
    string s1 = "first second third";
    string s2 = "second";
    auto index = s1.rfind(s2, 6);
    if(index < s1.length()){
        cout<<"Found at index : "<<index << endl;
    }
    else{
        cout<<"Not found"<<endl;
    }

    string s3 = "center";
    //s1和s3共同具有的字符在字符串中首次出现的位置
    int pos = s1.find_first_of(s3);
    if(index < s1.length()){
        cout<<"Found at index : "<<pos << endl;
    }
    else{
        cout<<"Not found"<<endl;
    }
}
