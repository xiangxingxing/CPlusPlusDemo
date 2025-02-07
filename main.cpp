#include "out_put.h"
#include "foo/Student.h"
#include "templ/Point.h"
#include "templ/List.h"
#include "company/worker_manager.h"
#include "company/employee.h"
#include "company/manager.h"
#include "company/worker.h"
#include "company/boss.h"
#include "stlib/std_manager.h"
#include "model/person.h"
#include "model/salary_manager.h"
#include "model/sales.h"

#include <iostream>
#include <fstream>
#include "vector"
#include "map"
//#include "json/json.h"
#include "my_algrithm/sort/quick_sorter.h"
#include "my_algrithm/sort/merge_sorter.h"
#include "model/my_complex.h"

#include <queue>
#include <unordered_map>
#include <set>
using namespace std;

static void GetSetDemo();
static void ConstObjDemo();
static void FriendDemo();
static void TemplateDemo();
static void ClassTemplDemo();
static void ListDemo();
static void FileTextDemo();
static void FileBinaryDemo();
static void InOutputDemo();

[[noreturn]] static void CompanyDemo();
static void CompanyTest();

static void VectorDemo();
static void VectorDemo2();
const vector<int>& VectorDemo3();
vector<Student *>& VectorDemo4();
static void StringDemo1();
static void OrderedMapDemo();
static void UnOrderedMapDemo();
static void SetDemo();
static int JsonDemo();

static void PointRefDemo1();
static void ConstInputTest(const Person * person);
//测试 指针or指针指向内容 是否能改变
static void ConstTest1(Person * person);
static void ConstTest2(const Person * person);
static void ConstTest3(Person ** person);
static void ReferTest1(Person & person);
static void ReferTest2(const Person & person);
static void ConstPointerTest();
static void PointerTest();
static std::shared_ptr<Person> SharedPointTest();

void Print_Map(const map<int, int>& map1);

static void OperatorDemo();
static void SwapDemo(float * f1, float * f2);

static void InputDemo();

static vector<int> integers_;
static vector<Student *> students_;

static void QuickSortDemo();
static void MergeSortDemo();
static void MyComplexDemo();

static void LambdaExpression();
static void LambdaExpression2();

static void StdIterator();
static void PriorityQueue();

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
    //InOutputDemo();

    //CompanyDemo();
    //CompanyTest();

    //system("ls");

    //VectorTraversal();
    //VectorDemo2();
    //StringDemo1();

    //StdManager::PlayerDemo();
    //StdManager::SetSortDemo();

	//OrderedMapDemo();
    //return JsonDemo();

    //PointRefDemo1();

    //ConstPointerTest();

    //OperatorDemo();

    //PointerTest();

	//VectorDemo();

	//QuickSortDemo();

	//MergeSortDemo();

	//MyComplexDemo();

	//LambdaExpression();

	//LambdaExpression2();

	//StdIterator();

	//PriorityQueue();

	//InputDemo();

	//UnOrderedMapDemo();

	SetDemo();

//	auto person = SharedPointTest();
//	person->ShowInfo();
//	cout << "person.use_count() = " << person.use_count();

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
    DisPlay(stu);
    auto *pstu = new Student("李刚", 16, 99);
    DisPlay(*pstu);
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
    DisPlay(stu);//调用友元函数
    ifs.close();
}

void InOutputDemo(){
    char ch;
    //hello my name is alex -> hellomynameisalex 忽略了空格、换行
//    while(cin >> ch){
//        cout << ch;
//    }

    //hello my name is alex -> hello my name is alex
    //get()并不会读取newline字符
//    while(cin.get(ch)){
//        cout << ch;
//    }

    char strBuf[11];
    cin.getline(strBuf, 11);//读取10个字符，末尾为'/0'
    cout << strBuf << '\n';
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

void VectorDemo(){
	//StdManager::VectorTraversal();

	/** region ## test vectorDemo3 ## */

//	auto array = VectorDemo3();
//	array.push_back(100);
//	array[0] = 111;
//
//	cout << "array == integers_ = " << (array == integers_) << endl; // 0 = false
//
//	cout<< "in integers_: " <<endl;
//	for (auto & elem : integers_) {
//		cout<< elem << " " ;
//	}
//	cout<<endl;
//
//	cout << "\nin array:" << endl;
//	for (auto & elem : array) {
//		cout<< elem << " ";
//	}
	/** endregion */

	/** region ## test vectorDemo4 ## */
	//结论‼️更新‼️：
	// 1.返回 auto student_list 即 vector<Student *> student_list 后，新数组对原数组进行了复制，
	// 	 修改新数组指针成员变量的内容是ok的,但是增删的话，对原数组无效
	// 2.返回 auto& student_list 即 vector<Student *>& student_list 后，新数组与原数组一致，增删改均影响原数组
	// 3.返回数组后，用来做修改操作的话引用是不变的，但增删的话，引用就改变，与原数组引用不一样了【应该是经过了复制】，返回原数组引用&则保持更新一致
	// ‼️VectorDemo4().push_back(new_one);可以直接增删‼️

	//auto student_list = VectorDemo4();//‼️区分‼️
	auto& student_list = VectorDemo4();//‼️区分‼️
	cout << "student_list == students_ = " << (student_list == students_) << endl; // 1 = true (0 = false)

	for (const auto& item: student_list)
	{
		DisPlay(*item);
	}

	for (auto & elem : student_list) {
		elem->SetAge(100);
	}
//	//student_list.clear();
	auto new_one = new Student("e", 77, 77);
	student_list.push_back(new_one);

	cout << "student_list == students_ = " << (student_list == students_) << endl; // 0 = false

	for (int i = 0; i < students_.size(); ++i)
	{
		cout << "student_list.at(i) == students_.at(i) " << (student_list.at(i) == students_.at(i)) << endl; // 1 = true
	}

	//‼️ 这样可以对源数组进行直接增删的操作 ‼️
	//VectorDemo4().push_back(new_one);

//	students_.push_back(new_one);

//	cout << "after 1 change:" <<endl;
//	for (const auto& item: student_list)
//	{
//		DisPlay(*item);
//	}
//	cout << endl;
//	cout << "student_list == students_ = " << (student_list == students_) << endl; // 0 = false

	cout << "in origin students_" << endl;
	for (const auto& item: students_)
	{
		//item->SetAge(100);
		DisPlay(*item);
	}
//
//	auto new_two = new Student("f", 66, 66);
//	students_.push_back(new_two);
////	students_.push_back(new_one);
//
//	cout << "after 2 change:" <<endl;
//
//	for (const auto& item: student_list)
//	{
//		DisPlay(*item);
//	}
//	cout << endl;
//	cout << "student_list == students_ = " << (student_list == students_) << endl; // 0 = false
//
//	cout << "in origin students_" << endl;
//	for (const auto& item: students_)
//	{
//		DisPlay(*item);
//	}
	/** endregion */
}


void VectorDemo2() {
    vector<Student*> v;
    Student stu1("a", 11, 111);
    Student stu2("b", 22, 222);
    Student stu3("c", 33, 333);
    Student stu4("d", 44, 444);
    v.push_back(&stu1);
    v.push_back(&stu2);
    v.push_back(&stu3);
    v.push_back(&stu4);

/*    for (auto it = integers_.begin(); it != integers_.end() ; it++) {
        (*it)->show();
    }*/

    for (auto & elem : v) {
        elem->show();
    }

    //拷贝构造
//    vector<Student*> v2(v);
//    for(Student* & elem : v2){
//        elem->show();
//    }
	//v2[0] = new Student("e", 19, 89);

	//1.引用
	//const vector<Student*> v3 = v;
	//auto& s3 = v3.at(0);
	//s3 = new Student("e", 19, 89);//‼️当v3是const、并且s3为取v3中第0个元素的引用【别名】时，s3禁止重新new，因为不能给Student* const&赋值

	//2.const赋值拷贝
	//const vector<Student*> v3 = v;
	////‼️当v3是const、并且s3为取v3中第0个元素的指针时，s3可以重新new，但跟原v3.at(0)没有关系了，因为v3[0]指针进行了拷贝
	////auto* s3 = v3.at(0);//指针拷贝给了s3
	//auto s3 = v3.at(0);
	//s3 = new Student("e", 19, 89);//‼️s3指针指向别处，跟原v3.at(0)没有关系,不会修改原先指向的内容

	//v3[0] = new Student("e", 55, 555);//‼️当v3是const时禁止，因为v3是常量数组，它不能更改某一个位置上的对象，此处为Student* const&

	//3.赋值构造
	vector<Student*> v3 = v;
	//auto& s3 = v3.at(0);
	////‼️当v3是const、并且s3为取v3中第0个元素的引用时禁止，不能给Student* const&赋值
	////‼️当v3是const、并且s3为取v3中第0个元素的指针时可以new，但跟原v3.at(0)没有关系了，因为v3[0]指针进行了拷贝
	//s3 = new Student("e", 55, 555);//✅ 可以修改v3，但无法修改v[0]

	auto* s = v3.at(0);//指针进行了copy
	s->SetAge(000);//可以修改原数组v的成员的内容,因为拷贝后的指针与V[0]均指向同一位置
	v3[0] = new Student("e", 55, 555); //v3[0]指向新实例，但不会影响v[0]的指向

	//auto s4 = new Student("f", 88, 99);
	//v3.push_back(s4);//v3被const修饰时则不能push_back

	cout << "after operated :" << endl;
	cout << "In new v3 :" << endl;
	for (auto & elem : v3) {
		elem->show();
	}
	cout << "In origin v :" << endl;
	for (auto & elem : v) {
		elem->show();
	}
}

const vector<int>& VectorDemo3()
{
	integers_.push_back(10);
	integers_.push_back(11);
	integers_.push_back(12);
	return integers_;
}

 vector<Student *>& VectorDemo4()
{
	Student* stu1 = new Student("a", 18, 90);
	Student* stu2 = new Student("b", 19, 89);
	Student* stu3 = new Student("c", 20, 88);
	Student* stu4 = new Student("d", 21, 92);
	students_.push_back(stu1);
	students_.push_back(stu2);
	students_.push_back(stu3);
	students_.push_back(stu4);

	return students_;
}

void StringDemo1(){
    //StdManager::StringConstructDemo();
    StdManager::VectorFuncDemo();
}

void OrderedMapDemo(){
    map<int, int> m;
    //插入方式
    //第一种
    m.insert(pair<int, int>(4, 44));
    //第二种
    m.insert(make_pair(3, 33));
    //第三种
    m.insert(map<int, int>::value_type(2, 22));
	//第四种
	m[10] = 100;
    Print_Map(m);

    //删除
    m.erase(m.begin());
    Print_Map(m);

    auto pos = m.find(2); // 返回的迭代器
    if(pos != m.end()){
        cout << "查到了元素 key = " << pos->first << endl;
    }else{
        cout << "未找到元素" << endl;
    }

	map<int, std::string> orderedMap;
	orderedMap[5] = "five";
	orderedMap[3] = "three";
	orderedMap[8] = "eight";
	orderedMap[1] = "one";
	//1.有序查询
	for(auto it = orderedMap.begin(); it != orderedMap.end(); it++){
		std::cout << it->first << " : " << it->second << std::endl;
	}

	for(auto& pair : orderedMap){
		std::cout << pair.first << " : " << pair.second << std::endl;
	}

	//2.范围查询  查找所有键在 [2, 6] 之间的键值对
	auto start = orderedMap.lower_bound(2);
	auto end = orderedMap.upper_bound(6);
	for (auto it = start; it != end; ++it) {
		std::cout << it->first << ": " << it->second << std::endl;
	}

	//3.最大最小值
	std::cout << "\nSmallest key: " << orderedMap.begin()->first;
	std::cout << "\nLargest key: " << orderedMap.rbegin()->first << std::endl;
}

void Print_Map(const map<int, int>& m){
    for(auto & it : m){
        cout << "key = " << it.first <<", val = " << it.second << endl;
    }
}

void UnOrderedMapDemo(){
	std::unordered_map<int, std::string> map = {{ 1, "one" },
												{ 2, "two" },
												{ 3, "three" }};
	if(map.count(3)){
		std::cout << "map.count(3) = " << map.count(3) << ", map[3] = " << map[3] << std::endl;
	}

	map[3] = "三";
	std::cout << "After modification, map[3] = " << map[3] << std::endl;
}

void SetDemo(){
	std::set<std::string> strSet;
	strSet.insert("google");
	strSet.insert("apple");
	strSet.insert("byte");
	if(strSet.count("apple")){
		std::cout << "apple exits in strSet" << std::endl;
	}

	if(!strSet.count("banana")){
		std::cout << "banana doesn't exit in strSet" << std::endl;
	}

	if(strSet.find("byte") != strSet.end()){
		std::cout << "byte exits in strSet" << std::endl;
	}

	for(auto iter = std::next(strSet.begin()); iter != strSet.end(); iter++){
		if((*iter) == "byte"){
			std::cout << "get byte" << std::endl;
			std::cout << "get prev " << (*std::prev(iter)) << std::endl;
		}
	}
}

static int JsonDemo(){
//    Json::Value root;
//    std::ifstream ifs;
//    ifs.open("/Users/dev/shared/file/demo.json");
//
//    Json::CharReaderBuilder builder;
//    builder["collectComments"] = false;
//    JSONCPP_STRING errs;
//    if(!Json::parseFromStream(builder, ifs, &root, &errs)){
//        std::cout << errs << std::endl;
//        return EXIT_FAILURE;
//    }
//
//    std::cout << root << std::endl;
    return EXIT_SUCCESS;
}

static void PointRefDemo1(){
    //auto p = new Person(20, "dev");
    //ConstTest1(p);
    //ConstTest2(p);
    //ConstTest3(&p);
//    if(p == nullptr){
//        std::cout << "p has been changed to null." << std::endl;
//    }
//    else{
//        p->ShowInfo();
//    }

    Person p(90, "dev", 100);
//    p.ShowInfo();
//    ReferTest1(p);
//    p.ShowInfo();

    auto * p1 = new Person(10, "leb", 30);
    auto * p2 = new Person(10, "leb", 30);
    vector<Person *> vec;
    vec.push_back(p1);
    vec.push_back(p2);
}

static void ConstInputTest(const Person * person){

}

void ConstTest1(Person * person){
    //person = nullptr; ❌无效
    person->age_ = 2;
    person->name_ = "google";
}

void ConstTest2(const Person * person){
    auto new_p = new Person(11, "kid");
    person = new_p;//无效
    //person->age_ = 1; ❌禁止
}

//函数修改指针本身地址
void ConstTest3(Person ** person){
    auto new_p = new Person(11, "kid");
    *person = new_p;
    //*person = nullptr;
}

void ReferTest1(Person & person){
    person.name_ = "lol";
    person.age_ = 10;

    Person np(11, "ll", 111);
    person = np;//赋值有效
}

void ReferTest2(const Person & person){
    //person.age_ = 1;//❌禁止
}

/*
 * https://www.zhihu.com/question/443195492
 * const默认作用于其左边的东西，若左边没有则作用于其右边的东西
 * */
void ConstPointerTest(){
    //const在*的左侧，表示指针指向的对象为常量
    //const在*的右侧，表示指针本身为常量

    //指向常量的指针
    const int first = 1;
    //int *p_first = &first;// ❌ 类型不匹配 得是：const int
    const int *p_first = &first;// p_first是一个指向const int类型常量对象的指针，p_first本身并不是常量

    //‼️ 非const对象的地址可以赋值给指向const对象的指针
    int first_val = 1;
    const int *p_first_val = &first_val;
	//*p_first_val = 11;//❌

	int* const q_first_val = &first_val;

    //const指针必须初始化
    int num = 0;
    int * const ptr = &num;
    *ptr = 10;
    cout << *ptr << endl;

    //指向常量的常指针
    //const常指针指向const常量
    const int second = 2;
    //int * const p2 = &second;//❌ error: 该常量指针p2指向的为int变量，而&second为常量的引用
    const int * const p2 = &second;//✅ 该常量指针p2指向的为const int*常量类型
}

void OperatorDemo(){
    /*SalaryManager s(3);
    s["levi"] = 11.1;
    s["james"] = 22.2;
    s["kd"] = 33.3;
    s.display();*/

    Sales s1("levi", "110110110", 18);
    cout<<s1;
    cout<<endl;
    cin>>s1;
    cout<<s1;
}

void PointerTest(){
    float a = 1.3f;
    float b = 2.4f;
    cout << "before : a = " << a << ", b = " << b << endl;
    SwapDemo(&a, &b);
    cout << "after swap : a = " << a << ", b = " << b << endl;
}

std::shared_ptr<Person> SharedPointTest()
{
	auto person = std::make_shared<Person>(19, "xxx", 2000);
	return person;
}

void SwapDemo(float * f1, float * f2){
    float temp = *f1;
    *f1 = *f2;
    *f2 = temp;
}

void InputDemo(){
	cout << "<start read strings from input>" << endl;
	int n;
	std::vector<std::string> input;
	while(cin >> n){
		std::string str;
		for(int i = 0; i < n; i++){
			cin >> str;
			input.push_back(str);
		}

		int idx = 0;
		for(auto& s : input){
			std::cout << idx << ":" << s <<endl;
			idx++;
		}
	}

	cout << "<end read n strings>" << endl;
}

void QuickSortDemo()
{
	vector<int> v = { -1, 0, 3, 8, 2, 5, 1, 27, 10, 14, 9, 8, 26 };
//	QuickSorter::QuickSortLomuto(v, 0, v.size() - 1);
//	QuickSorter::QuickSortHoare(v, 0, v.size() - 1);
	QuickSorter::QuickSortBook(v, 0, v.size() - 1);
	for (const auto& item: v)
	{
		cout << item << " ";
	}
}

void MergeSortDemo()
{
	vector<int> v = { -1, 0, 3, 8, 2, 5, 1, 27, 10, 14, 9, 8, 26 };
	MergeSorter::Sort(v);

	for (const auto& item: v)
	{
		cout << item << " ";
	}
}

void MyComplexDemo()
{
	Complex c1(5, 1);
	Complex c2(2);
	//c1 += c2;
	cout << Real(c1) << endl;
	cout << Imag(c2) << endl;

	Complex c3 = c1 + c2;
	Complex c4 = c1 + 5;
	Complex c5 = 8 + c1;

	Complex c6 = +c1;
	Complex c7 = -c1;

	c6.Real();

	Complex c8 = conj(c1);
	cout << "c1 : " << c1 << endl;;
	cout << "c8 : " << c8 << endl;;
}

void LambdaExpression2()
{
	std::vector<int> v{1, 2, 2, 2, 3, 3, 4};
	int n1 = 3;
	int n2 = 5;
	auto is_even = [](int i){ return i%2 == 0; };

	auto result1 = std::find(begin(v), end(v), n1);
	auto result2 = std::find(begin(v), end(v), n2);
	auto result3 = std::find_if(begin(v), end(v), is_even);

	(result1 != std::end(v))
	? std::cout << "v contains " << n1 << '\n'
	: std::cout << "v does not contain " << n1 << '\n';

	(result2 != std::end(v))
	? std::cout << "v contains " << n2 << '\n'
	: std::cout << "v does not contain " << n2 << '\n';

	(result3 != std::end(v))
	? std::cout << "v contains an even number: " << *result3 << '\n'
	: std::cout << "v does not contain even numbers\n";
}

void LambdaExpression()
{
	vector<Student*> list;
	auto* stu1 = new Student("a", 18, 90);
	auto* stu2 = new Student("b", 19, 89);
	auto* stu3 = new Student("c", 20, 44);
	auto* stu4 = new Student("d", 21, 55);

	list.push_back(stu1);
	list.push_back(stu2);
	list.push_back(stu3);
	list.push_back(stu4);

	vector<std::shared_ptr<Student>> all_singer;

	all_singer.push_back(static_cast<const shared_ptr<Student>>(stu1));
	all_singer.push_back(static_cast<const shared_ptr<Student>>(stu2));
	all_singer.push_back(static_cast<const shared_ptr<Student>>(stu3));
	all_singer.push_back(static_cast<const shared_ptr<Student>>(stu4));

	vector<std::shared_ptr<Student>> official_singer_list;

	auto is_official = [](const std::shared_ptr<Student>& singer) {
		return singer->GetScore() > 60;
	};

	std::copy_if(all_singer.begin(),
			all_singer.end(),
			std::inserter(official_singer_list,
					official_singer_list.end()), is_official);

//	std::copy_if(all_singer.begin(),
//			all_singer.end(),
//			std::back_inserter(official_singer_list), is_official);

	for (auto item: official_singer_list)
	{
		DisPlay(*item);
	}
}

void StdIterator()
{
	vector<Student*> list;
	auto* stu1 = new Student("a", 18, 90);
	auto* stu2 = new Student("b", 19, 89);
	auto* stu3 = new Student("c", 20, 44);
	auto* stu4 = new Student("d", 21, 55);

	list.push_back(stu1);
	list.push_back(stu2);
	list.push_back(stu3);
	list.push_back(stu4);

	auto in_order = std::next(list.begin(), 2);
	auto reverse_order = std::prev(list.end(), 2);

	auto res = in_order == reverse_order;
	cout << "in_order == reverse_order : " << (res ? "true" : "false");
}

void PriorityQueue(){
	std::priority_queue<Student, std::vector<Student>, CompareAge> pq;
	pq.emplace("a", 42, 90);
	pq.emplace("b", 19, 89);
	pq.emplace("c", 30, 44);

	while(!pq.empty()){
		std::cout << pq.top().GetName() << " - " << pq.top().GetAge() << std::endl;
		pq.pop();
	}
}