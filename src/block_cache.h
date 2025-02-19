#pragma once

#include <iostream>
#include <cstdlib>
#include <cstring>

class CacheEntry {
public:
    char* key;
    char* value;
    int freq;
    CacheEntry* next;
    CacheEntry* prev;

    CacheEntry(char* k, char* v) : key(k), value(v), freq(1), next(nullptr), prev(nullptr) {}
};

class CacheSet {
public:
    int capacity;
    int size;
    CacheEntry** frequencyList;

    CacheSet() : capacity(0), size(0), frequencyList(nullptr) {}
    void initialize(int cap) {
        capacity = cap;
        frequencyList = (CacheEntry**)calloc(capacity, sizeof(CacheEntry*));
    }

    ~CacheSet() {
        // Individual CacheEntry clean-up is not required here as the memory is managed by BlockCache
        free(frequencyList);
    }
};

class BlockCache {
public:
    uint64_t numSets;
    int setSize;
    CacheSet* sets;
    char* keyBuffer;
    char* valueBuffer;
    int keyLength;
    int valueLength;
    int bufferUsed;
    uint64_t totalEntries;

    unsigned int hash(const char* key) {
        unsigned long int value = 0;
        for (int i = 0; key[i] != '\0'; i++) {
            value = value * 37 + key[i];
        }
        return value % numSets;
    }

    BlockCache(int kLength, int vLength);

    ~BlockCache() {
        free(sets);
        free(keyBuffer);
        free(valueBuffer);
    }

    void insert(const char* key, const char* value);
    char* get(const char* key);
    void displayCache();
};
