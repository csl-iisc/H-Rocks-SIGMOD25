#include <chrono>
#include <map>
#include <vector>
#include <string>
#include <iostream>

class Timer {
private:
    // Map from a string category to a vector of duration<double>
    std::map<std::string, std::vector<std::chrono::duration<double>>> records;

    // Map to store start times for each timer
    std::map<std::string, std::chrono::steady_clock::time_point> start_times;

public:

    uint64_t numOps; 
    // Start the timer for a given category and instance
    void startTimer(const std::string& category, int instanceId); 

    // Stop the timer and record the elapsed time in the specified category and instance
    void stopTimer(const std::string& category, int instanceId); 

    // Calculate and return the total time spent in a specific category across all instances
    double getTotalTime(const std::string& category);

    // Calculate and return the average time spent in a specific category across all instances
    double getAverageTime(const std::string& category);

    // Calculate and return the total time across all categories and instances
    double getTotalTime();

    // Calculate the throughput (number of operations per second) for a specific category
    double getThroughput(const std::string& category); 

    //Calculate the throughput (number of operations per second) across all categories
    double getThroughput();

    // Print all times for debugging purposes
    void printTimes();
};
