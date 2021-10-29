#include "out_put.h"
#include "foo/Student.h"
#include "templ/Point.h"
#include "templ/List.h"
#include "company/worker_manager.h"
#include "company/employee.h"
#include "company/manager.h"
#include "company/worker.h"
#include "company/boss.h"

#include <iostream>
#include <fstream>

using namespace std;

static void GetSetDemo();
static void ConstObjDemo();
static void FriendDemo();
static void TemplateDemo();
static void ClassTemplDemo();
static void ListDemo();
static void FileTextDemo();
static void FileBinaryDemo();

[[noreturn]] static void CompanyDemo();
static void CompanyTest();

int main() {
    //out_put::BasicUserInputDemo();
    //out_put::BasicOutputFormatDemo();
    //out_put::StringDemo();
    //out_put::StringOperationDemo();
    //GetSetDemo();
    //ConstObjDemo();
    //FriendDemo();

    //TemplateDemo();
    //ClassTemplDemo();

    //ListDemo();
    //FileTextDemo();
    //FileBinaryDemo();

    CompanyDemo();
    //CompanyTest();

    //system("ls");

    return 0;
}

void GetSetDemo(){
    (new Student("小米", 15, 90)) -> show();
    (new Student("小红", 16, 80)) -> show();
    (new Student("小蓝", 17, 91)) -> show();
    (new Student("小绿", 18, 92)) -> show();

    int total = Student::GetTotal();
    float points = Student::GetPoints();
    cout<<"当前共有"<<total<<"名学生，总成绩是"<<points<<"，平均分是"<<points/total<<endl;
}

void ConstObjDemo(){
    //只能调用const成员函数
    //常对象
    const Student stu("小明", 15, 90.8);
    cout<<stu.GetName()<<"的年龄是"<<stu.GetAge()<<"，成绩是"<<stu.GetScore()<<endl;
    //常对象指针
    const Student *pstu = new Student("李刚", 16, 99);
    //pstu->show(); //show()不是const方法，无法调用
    cout<<pstu->GetName()<<"的年龄是"<<pstu->GetAge()<<"，成绩是"<<pstu->GetScore()<<endl;
}

//友元
void FriendDemo(){
    Student stu("小明", 18, 88);
    DisPlay(&stu);
    auto *pstu = new Student("李刚", 16, 99);
    DisPlay(pstu);
}

void TemplateDemo(){
    out_put output;
    int a = 99;
    int b = -100;
    cout<<"before swap, a = "<<a<<", b = "<<b<<endl;
    output.Swap(a, b);
    cout<<"after swap, a = "<<a<<", b = "<<b<<endl;
}

void ClassTemplDemo(){
    //对象变量
    Point<int, int> p1(10, 20);
    cout<<"x="<<p1.GetX()<<", y="<<p1.GetY()<<endl;

    Point<int, string> p2(10, "东经180度");
    cout<<"x="<<p2.GetX()<<", y="<<p2.GetY()<<endl;

    auto *p3 = new Point<string, string>("东经180度", "北纬210度");
    cout<<"x="<<p3->GetX()<<", y="<<p3->GetY()<<endl;
}

void ListDemo(){
    List<int> l;
    for (int i = 1; i < 5; ++i) {
        l.push_bask(i);
    }

    for (int i = 0; i < l.length(); ++i) {
        cout<<l[i]<<"  ";
    }
}

void FileTextDemo(){
    //读文件
    ifstream in_file;
    in_file.open("/Users/dev/Documents/test.json", ios::in);
//    if (!in_file)
    if(!in_file.is_open())
    {
        cout << "test.txt doesn't exist" << endl;
        return;
    }

    //写文件
    ofstream o_file;
    o_file.open("/Users/dev/Documents/output.txt", ios::app);
    if (!o_file)
    {
        cout << "error 1" << endl;
        return;
    }

    //第一种
//    char buf[1024];
//    while (in_file >> buf){
//        o_file << buf;
//    }
    //第二种
//    string buf;
//    while(getline(in_file, buf)){
//        cout<<buf<<endl;
//    }

    //第三种
    char buf[1024] = {0};
    while(in_file.getline(buf, sizeof(buf))){
        cout<<buf<<endl;
    }

    o_file.close();
    in_file.close();
}

void FileBinaryDemo(){
    //write
    char file_name[] = "/Users/dev/Documents/output.txt";
//    ofstream ofs(file_name, ios::out | ios::binary);
//    Student stu("levi", 18, 100);
//    ofs.write((const char *)&stu, sizeof(Student));
//
//    ofs.close();

    //read
    ifstream ifs(file_name, ios::in | ios::binary);
    //打开文件时需判断文件是否打开成功
    if(!ifs.is_open()){
        cout<<"open file failed."<<endl;
        return;
    }

    Student stu("", 0, 0);
    ifs.read((char *)&stu, sizeof(Student));
    //stu.show();
    DisPlay(&stu);//调用友元函数
    ifs.close();
}

[[noreturn]] void CompanyDemo(){
    WorkerManager manager;
    int choice = 0;
    while (true){
        manager.Show_Menu();
        cout << "请输入您的选择："<<endl;
        cin >> choice;

        switch (choice) {
            case 0: //0.退出管理程序
                manager.ExitSystem();
                break;
            case 1: //1.增加职工信息
                manager.AddWorkers();
                break;
            case 2: //2.显示职工信息
                manager.ShowAllWorkers();
                break;
            case 3: //3.删除离职职工
                manager.DeleteWorker();
                break;
            case 4: //4.修改职工信息
                manager.ModifyWorker();
                break;
            case 5: //5.查找职工信息
                manager.FindWorker();
                break;
            case 6: //6.按照编号排序
                manager.SortWorkers();
                break;
            case 7: //7.清空所有文档
                manager.CleanFile();
                break;

            default:
                //https://stackoverflow.com/questions/27616522/cannot-use-systemcls-in-xcode-in-mac
                system("clear");
                break;
        }
    }
}

void CompanyTest(){
    Worker *worker = nullptr;
    worker = new Employee(1, "张三", 1);
    worker->ShowInfo();

    worker = new Manager(2, "李四", 2);
    worker->ShowInfo();

    worker = new Boss(3, "王五", 3);
    worker->ShowInfo();
}
