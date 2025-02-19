#include <iostream> 
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <functional>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <getopt.h>
#include <unistd.h>
#include <algorithm>  //for std::generate_n
#include <set>

#include <bits/stdc++.h>

#include "hrocksdb.h"
#include <stdio.h>
#include "config.h"



#define TIME_NOW std::chrono::high_resolution_clock::now()

typedef std::vector<char> char_array;

char_array charset()
{
    //Change this to suit
    return 
        char_array({'A','B','C','D','E','F',
                'G','H','I','J','K',
                'L','M','N','O','P',
                'Q','R','S','T','U',
                'V','W','X','Y','Z',
                });
};

std::string genRandomStrings(size_t length, std::function<char(void)> rand_char)
{
    std::string str(length,0);
    std::generate_n(str.begin(), length, rand_char);
    return str;
}

using namespace ROCKSDB_NAMESPACE;
int main(int argc, char **argv) {
    int optionChar;
    uint64_t numPuts, numGets;
    size_t keySize, valueSize;
    uint64_t prefill = 0;
    uint64_t numOps = 0;


    while ((optionChar = getopt (argc, argv, ":p:g:k:v:")) != -1) {
        switch (optionChar) {
            case 'p': prefill = atoi (optarg); break;
            case 'n': numOps = atoi (optarg); break;
            case 'k': keySize = atoi (optarg); break;
            case 'v': valueSize = atoi (optarg); break;
            case ':': fprintf (stderr, "option needs a value\n");
            case '?': fprintf (stderr, "usage: %s [-n <number of keys>] [-k <key size>] [-v     <value size>]\n", argv[0]);
        }
    }


    numPuts = prefill + numOps/2;
    numGets = numOps/2;
    numPuts = numPuts * 5; 
    numGets = numGets * 5; 
    
    std::cout << "Number of puts: " << numPuts << std::endl;
    std::cout << "Number of gets: " << numGets << std::endl;
    std::cout << "Key size: " << keySize << std::endl;
    std::cout << "Value size: " << valueSize << std::endl;

    std::vector<std::string> keys(numPuts); 
    std::vector<std::string> values(numPuts); 
    const auto ch_set = charset();
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, ch_set.size()-1);
    auto randchar = [ch_set, &dist, &rng](){return ch_set[dist(rng)];};

    for(uint64_t i = 0; i < numPuts; ++i) {
        keys[i] = genRandomStrings(keySize - 1, randchar); 
        values[i] = genRandomStrings(valueSize - 1, randchar); 
    }

    Config conf; 
    conf.setMemtableSize(20000000);
    conf.setNumMemtables(10);
    HRocksDB* hdb = new HRocksDB(conf); // Initialize the db variable
    hdb->HOpen("test_ycsbA");

    char *key, *value; 
    key = (char*)malloc(keySize);
    value = (char*)malloc(valueSize);
    for(uint64_t i = 0; i < numPuts; ++i) {
        strcpy(key, keys[i].c_str()); 
        strcpy(value, values[i].c_str()); 
        hdb->Put(key, value);  
    }

    for(uint64_t i = 0; i < numGets; ++i) {
        strcpy(key, keys[i % numPuts].c_str()); 
        hdb->Get(key);  
    }

    hdb->Close();
    free(key);
    free(value);
    return 0; 
}
