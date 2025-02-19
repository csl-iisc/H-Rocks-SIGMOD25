#include <chrono>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include "timer.h"

void Timer::startTimer(const std::string& category, int instanceId) {
    std::string key = category + std::to_string(instanceId);
    start_times[key] = std::chrono::steady_clock::now();
}

// Stop the timer and record the elapsed time in the specified category and instance
void Timer::stopTimer(const std::string& category, int instanceId) {
    std::string key = category + std::to_string(instanceId);
    auto end_time = std::chrono::steady_clock::now();
    auto start_time = start_times.find(key);
    if (start_time != start_times.end()) {
        std::chrono::duration<double> elapsed = end_time - start_time->second;
        records[key].push_back(elapsed);
    }
}

// Calculate and return the total time spent in a specific category across all instances
double Timer::getTotalTime(const std::string& category) {
    double total = 0.0;
    for (const auto& item : records) {
        if (item.first.find(category) == 0) {  // Check if the category name is at the start of the key
            for (const auto& time : item.second) {
                total += time.count();
            }
        }
    }
    return total;
}

// Calculate and return the average time spent in a specific category across all instances
double Timer::getAverageTime(const std::string& category) {
    double total = 0.0;
    int count = 0;
    for (const auto& item : records) {
        if (item.first.find(category) == 0) {  // Check if the category name is at the start of the key
            for (const auto& time : item.second) {
                total += time.count();
                count++;
            }
        }
    }
    return count > 0 ? total / count : 0.0;
}

// Calculate and return the total time across all categories and instances
double Timer::getTotalTime() {
    double total = 0.0;
    for (const auto& category : records) {
        for (const auto& time : category.second) {
            total += time.count();
        }
    }
    return total;
}

// Calculate the throughput (number of operations per second) for a specific category
double Timer::getThroughput(const std::string& category) {
    double total = 0.0;
    for (const auto& item : records) {
        if (item.first.find(category) == 0) {  // Check if the category name is at the start of the key
            for (const auto& time : item.second) {
                total += time.count();
            }
        }
    }
    return total > 0 ? numOps / total : 0.0;
}

//Calculate the throughput (number of operations per second) across all categories
double Timer::getThroughput() {
    double total = 0.0;
    for (const auto& category : records) {
        for (const auto& time : category.second) {
            total += time.count();
        }
    }
    return total > 0 ? numOps / total : 0.0;
}

// Print all times for debugging purposes
void Timer::printTimes() {
    for (const auto& category : records) {
        std::cout << "Category and Instance: " << category.first << std::endl;
        for (const auto& time : category.second) {
            std::cout << "Time: " << time.count() << " seconds" << std::endl;
        }
    }
}
