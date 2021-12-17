//
// Created by DEV on 2021/10/29.
//

#ifndef CPPRACTICE_WORKER_MANAGER_H
#define CPPRACTICE_WORKER_MANAGER_H

#include <iostream>
#include "worker.h"

using namespace std;

/*
 * 职工管理类
 * */
class WorkerManager{
public:
    int m_emp_num_;
    Worker ** m_emp_array_;
    bool m_file_empty_;
    const char * kFileName;

public:
    WorkerManager();
    ~WorkerManager();
    void SaveInfo();
    int GetWorkerCount();
    void InitWorkers();

    void Show_Menu();
    void ExitSystem();
    void AddWorkers();
    void ShowAllWorkers();
    void DeleteWorker();
    void ModifyWorker();
    void FindWorker();
    void SortWorkers();
    void CleanFile();

    int ExistWorker(int id);

};

#endif //CPPRACTICE_WORKER_MANAGER_H