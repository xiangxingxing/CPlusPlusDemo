//
// Created by DEV on 2021/10/29.
//

#include "worker_manager.h"
#include "employee.h"
#include "manager.h"
#include "boss.h"

#include <fstream>

WorkerManager::WorkerManager() {

    kFileName = "/Users/dev/Documents/company.txt";

    ifstream ifs(kFileName, ios::in);
    //文件不存在情况
    if(!ifs.is_open()){
        cout << "文件不存在" <<endl;
        this->m_emp_num_ = 0;
        this->m_emp_array_ = nullptr;
        this->m_file_empty_ = true;
        ifs.close();
        return;
    }

    //文件存在，且没有记录
    char ch;
    ifs >> ch;
    if(ifs.eof()){
        cout << "文件存在但记录空" <<endl;
        this->m_emp_num_ = 0;
        this->m_emp_array_ = nullptr;
        this->m_file_empty_ = true;
        ifs.close();
        return;
    }

    this->m_emp_num_ = this->GetWorkerCount();
    this->m_emp_array_ = new Worker*[m_emp_num_];
    this->InitWorkers();
    m_file_empty_ = false;
}

void WorkerManager::Show_Menu() {
    cout << "*********************************************"<<endl;
    cout << "********   欢迎使用职工管理系统   *************"<<endl;
    cout << "***********  0.退出管理程序  *****************"<<endl;
    cout << "***********  1.增加职工信息  *****************"<<endl;
    cout << "***********  2.显示职工信息  *****************"<<endl;
    cout << "***********  3.删除离职职工  *****************"<<endl;
    cout << "***********  4.修改职工信息  *****************"<<endl;
    cout << "***********  5.查找职工信息  *****************"<<endl;
    cout << "***********  6.按照编号排序  *****************"<<endl;
    cout << "***********  7.清空所有文档  *****************"<<endl;
    cout << "*********************************************"<<endl;
    cout << endl;
}

WorkerManager::~WorkerManager() {
    if(m_emp_array_){
        for (int i = 0; i < m_emp_num_; ++i) {
            if(this->m_emp_array_[i]){
                free(this->m_emp_array_[i]);
            }
        }

        m_emp_num_ = 0;
        m_file_empty_ = true;
        delete[] this->m_emp_array_;
        m_emp_array_ = nullptr;
    }
}

void WorkerManager::ExitSystem() {
    cout << "欢迎下次使用" << endl;
    exit(0);
}

void WorkerManager::AddWorkers() {
    cout << "请输入增加职工的数量：" << endl;
    int count = 0;
    cin >> count;
    while(cin.fail()) {
        cout << "输入不合法！请重新输入：" << std::endl;
        cin.clear();
        cin.ignore(256,'\n');
        cin >> count;
    }
    if(count > 0){
        int new_size = this->m_emp_num_ + count;
        Worker** new_space = new Worker*[new_size];

        if(this->m_emp_array_ != nullptr){
            for (int i = 0; i < m_emp_num_; ++i) {
                new_space[i] = m_emp_array_[i];
            }
        }

        for (int i = 0; i < count; ++i) {
            int id;
            string name;
            int type_id;

            cout << "请输入第 " << i + 1 << " 个新职工编号： " << endl;
            cin >> id;
            while(cin.fail()) {
                cout << "输入不合法编号！请重新输入：" << std::endl;
                cin.clear();
                cin.ignore(256,'\n');
                cin >> id;
            }

            cout << "请输入第 " << i + 1 << " 个新职工姓名： " << endl;
            cin >> name;

            cout << "请选择新职工的岗位： " << endl;
            cout << "1.普通职工" << endl;
            cout << "2.经理" << endl;
            cout << "3.老板" << endl;
            cin >> type_id;

            Worker * worker = nullptr;
            switch (type_id) {
                case 1:
                    worker = new Employee(id, name, 1);
                    break;
                case 2:
                    worker = new Manager(id, name, 2);
                    break;
                case 3:
                    worker = new Boss(id, name, 3);
                    break;
            }

            new_space[m_emp_num_ + i] = worker;
        }

        //释放原有空间
        delete[] this->m_emp_array_;

        this->m_emp_array_ = new_space;
        this->m_emp_num_ = new_size;
        this->m_file_empty_ = false;

        this->SaveInfo();
        cout << "成功添加 " << count << " 名新职工！" << endl;
    }else{
        cout << "输入有误" << endl;
    }
    //按任意键后 清屏后回到上级目录
//    cout << "按任意键返回上级..."<<endl;
//    char f;
//    cin >> f;
    system("clear");
}

void WorkerManager::ShowAllWorkers(){
    if(this->m_file_empty_){
        cout << "文件不存在或记录为空" << endl;
    }
    else{
        for (int i = 0; i < m_emp_num_; ++i) {
            Worker * worker = m_emp_array_[i];
            worker->ShowInfo();
        }
    }
}

void WorkerManager::SaveInfo() {
    ofstream ofs(kFileName, ios::out);

    for (int i = 0; i < this->m_emp_num_; ++i) {
        ofs << this->m_emp_array_[i]->m_id_ << " "
            << this->m_emp_array_[i]->m_name_ << " "
            << this->m_emp_array_[i]->m_dept_id_<< endl;
    }

    ofs.close();
}

int WorkerManager::GetWorkerCount() {
    ifstream ifs(kFileName, ios::in);
    if(!ifs.is_open()){
        cout<<"open file failed."<<endl;
        return -1;
    }
    int id;
    string name;
    int dept_id;
    int count = 0;
    while (ifs >> id && ifs >> name && ifs >> dept_id){
        count++;
    }

//    char buf[1024] = {0};
//    while(ifs.getline(buf, sizeof(buf))){
//        cout<<buf<<endl;
//    }

    ifs.close();
    return count;
}

void WorkerManager::InitWorkers(){
    ifstream ifs(kFileName, ios::in);
    if(!ifs.is_open()){
        cout<<"open file failed."<<endl;
    }
    int id;
    string name;
    int dept_id;
    int index = 0;
    Worker * worker = nullptr;
    while (ifs >> id && ifs >> name && ifs >> dept_id){
        switch (dept_id) {
            case 1:
                worker = new Employee(id, name, dept_id);
                break;
            case 2:
                worker = new Manager(id, name, dept_id);
                break;
            default:
                worker = new Boss(id, name, dept_id);
                break;
        }
        this->m_emp_array_[index] = worker;
        index++;
    }

    ifs.close();
}


void WorkerManager::DeleteWorker() {
    if(this->m_file_empty_){
        cout << "文件不存在或记录为空" << endl;
    }
    cout << "请输入离职员工编号 " << endl;
    int id = 0;
    cin >> id;
    if(this->ExistWorker(id) == -1){
        cout << "编号" << id << "员工不存在，删除失败"<<endl;
    }else{
        for (int i = id; i < this->m_emp_num_ - 1; ++i) {
            this->m_emp_array_[i] = this-> m_emp_array_[i + 1];
        }

        this->m_emp_num_--;
        this->SaveInfo();
        cout << "删除成功!" << endl;
    }

    system("clear");
}


int WorkerManager::ExistWorker(int id) {
    int index = -1;
    for (int i = 0; i < m_emp_num_; ++i) {
        if(this->m_emp_array_[i]->m_id_ ==  id){
            index = i;
            break;
        }
    }

    return index;
}

void WorkerManager::ModifyWorker() {
    if(this->m_file_empty_){
        cout << "文件不存在或记录为空" << endl;
    }
    else{
        cout << "请输入修改职工的编号:" << endl;
        int id;
        cin >> id;

        int index = this->ExistWorker(id);
        if(index != -1){
            //查找到编号员工
            free(this->m_emp_array_[index]);
            int new_id = 0;
            string new_name = "";
            int new_dept_id = 0;

            cout << "查到：  " << id << "号员工，请输入新职工号：  " << endl;
            cin >> new_id;

            cout << "请输入新姓名: " << endl;
            cin >> new_name;

            cout << "请输入岗位 " << endl;
            cout << "1.普通职工" << endl;
            cout << "2.经理" << endl;
            cout << "3.老板" << endl;
            cin >> new_dept_id;

            Worker * worker = nullptr;
            switch (new_dept_id) {
                case 1:
                    worker = new Employee(new_id, new_name, new_dept_id);
                    break;
                case 2:
                    worker = new Manager(new_id, new_name, new_dept_id);
                    break;
                default:
                    worker = new Boss(new_id, new_name, new_dept_id);
                    break;
            }

            m_emp_array_[index] = worker;
            cout << "修改成功！" << endl;
            worker->ShowInfo();

            //保存至文件中
            this->SaveInfo();
        }else{
            cout << "修改失败，查无此人" << endl;
        }
    }

    system("clear");
}

void WorkerManager::FindWorker(){
    if(this->m_file_empty_){
        cout << "文件不存在或记录为空" << endl;
    }
    else{
        cout << "请输入查找方式: " << endl;
        cout << "1.按职工编号查找: " << endl;
        cout << "2.按姓名查找: " << endl;
        int select = 0;
        cin >> select;

        if(select == 1){
            int id;
            cout << "请输入查找的职工编号： " << endl;
            cin >> id;

            int index = this->ExistWorker(id);
            if(index != -1){
                cout << "查找成功！该职工信息如下： " << endl;
                this->m_emp_array_[index]->ShowInfo();
            }else{
                cout << "查找失败，查无此人" << endl;
            }
        }else if(select == 2){
            string name;
            cout << "请输入查找的职工姓名： " << endl;
            cin >> name;
            bool flag = false;
            for (int i = 0; i < m_emp_num_; ++i) {
                if(m_emp_array_[i]->m_name_ == name){
                    flag = true;

                    this->m_emp_array_[i]->ShowInfo();
                }
            }
            if(!flag){
                cout << "查找失败，查无此人" << endl;
            }else{
                cout << "查询结束！" << endl;
            }
        }else{
            cout << "输入选项有误" << endl;
        }
    }

    system("clear");
}

void WorkerManager::SortWorkers() {
    if(this->m_file_empty_){
        cout << "文件不存在或记录为空" << endl;
    }else{
        cout << "请输入排序方式: " << endl;
        cout << "1.按职工编号升序: " << endl;
        cout << "2.按职工编号降序: " << endl;
        int select = 0;
        cout << "请输入排序方式： " << endl;
        cin >> select;

        for (int i = 0; i < m_emp_num_; ++i) {
            int temp_index = i;
            for (int j = i + 1; j < m_emp_num_; ++j) {
                auto first = m_emp_array_[temp_index];
                auto second = m_emp_array_[j];
                if(select == 1){
                    if(first->m_id_ > second->m_id_){
                        temp_index = j;
                    }
                }else if(select == 2){
                    if(first->m_id_ < second->m_id_){
                        temp_index = j;
                    }
                }
            }

            if(temp_index != i){
                auto temp = m_emp_array_[temp_index];
                m_emp_array_[temp_index] = m_emp_array_[i];
                m_emp_array_[i] = temp;
            }
        }
    }
    cout << "排序成功，排序后的结果为： " << endl;
    this->SaveInfo();
    this->ShowAllWorkers();

    system("clear");
}

void WorkerManager::CleanFile() {
    cout << "确认清空? " << endl;
    cout << "1.确认: " << endl;
    cout << "2.返回: " << endl;
    int select = 0;
    cin >> select;

    if(select == 1){
        //打开模式 ios::trunc 如果存在，删除文件并重新构建
        ofstream ofs(kFileName, ios::trunc);
        ofs.close();

        if(!m_file_empty_){
            //删除堆区的每个职工对象
            for (int i = 0; i < m_emp_num_; ++i) {
                if(this->m_emp_array_[i]){
                    free(this->m_emp_array_[i]);
                    m_emp_array_[i] = nullptr;
                }
            }

            //删除堆区数组指针
            delete[] this->m_emp_array_;
            m_emp_array_ = nullptr;
            m_emp_num_ = 0;
            m_file_empty_ = true;
        }

        cout << "清空成功！ " << endl;
    }
    system("clear");
}
