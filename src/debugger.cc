#include <iostream>
#include <string>
#include "debugger.h"

Debugger::Debugger() : debugMode(false) {}
Debugger::Debugger(bool debugMode) : debugMode(debugMode) {}


void Debugger::print(const std::string& message) {
    if (debugMode) {
        std::cout << "[DEBUG]: " << message << std::endl;
    }    
}


void Debugger::setDebugMode(bool debugMode) {
    this->debugMode = debugMode;
}