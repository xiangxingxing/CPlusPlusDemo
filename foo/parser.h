//
// Created by DEV on 2021/11/4.
//

#ifndef CPPRACTICE_PARSER_H
#define CPPRACTICE_PARSER_H

#include <iostream>
#include <string>
#include <curl/curl.h>
#include <json/json.h>
#include <json/reader.h>
#include <json/writer.h>
#include <json/json.h>

using namespace std;

class Parser{
public:
    Parser(){}
    Parser(const string &url) : url_(url){}
    void request();
    string get_json_string();
    Json::Value get_json(){ return root_; }
private:
    string url_;
    Json::Value root_;
};

#endif //CPPRACTICE_PARSER_H
