#pragma once
#include <iostream>
#include <string>
#include "config.h"

class Debugger {
    public: 
    bool debugMode;
    Debugger(bool debugMode);
    Debugger();
    void print(const std::string& message);
    void setDebugMode(bool debugMode);
};