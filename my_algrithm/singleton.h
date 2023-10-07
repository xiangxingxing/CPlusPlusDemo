//
// Created by xiangxx on 2023/8/25.
//

#ifndef CPPRACTICE_SINGLETON_H
#define CPPRACTICE_SINGLETON_H

#include <mutex>

class Singleton{
private:
	// 私有构造函数，保证外部不能实例化
	Singleton(){};

	// 静态指向实例的指针
	static Singleton* instance;

	// 静态互斥量，用于锁
	static std::mutex mtx;

public:
	static Singleton* getInstance();
};

Singleton* Singleton::instance = nullptr;

std::mutex Singleton::mtx;

Singleton* Singleton::getInstance()
{
	if(instance == nullptr){
		std::lock_guard<std::mutex> locker(mtx);
		if(instance == nullptr){
			instance = new Singleton();
		}
	}

	return instance;
}

#endif //CPPRACTICE_SINGLETON_H
