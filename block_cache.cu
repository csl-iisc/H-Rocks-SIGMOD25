#include "block_cache.h"
#include "debugger.h"
#include "helper.cuh"


#define NSETS 10000000
#define SETSIZE 8

void BlockCache::insert(const char* key, const char* value) {
    unsigned int idx = hash(key);
    CacheSet* set = &sets[idx];

    if (bufferUsed >= numSets * setSize) {
        std::cerr << "Cache is full, unable to add more items!" << std::endl;
        return;
    }

    char* keyLocation = keyBuffer + bufferUsed * keyLength;
    char* valueLocation = valueBuffer + bufferUsed * valueLength;
    strncpy(keyLocation, key, keyLength);
    strncpy(valueLocation, value, valueLength);
    bufferUsed++;

    CacheEntry* entry = new CacheEntry(keyLocation, valueLocation);
    // Logic to place the entry in the appropriate frequency list
}

char* BlockCache::get(const char* key) {
    unsigned int idx = hash(key);
    CacheSet* set = &sets[idx];
    CacheEntry* entry = set->frequencyList[0];  // Start looking from the lowest frequency

    while (entry != nullptr) {
        if (strncmp(entry->key, key, keyLength) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }
    return nullptr;
}


void BlockCache::displayCache() {
    std::cout << "Cache Content:\n";
    for (int i = 0; i < numSets; i++) {
        std::cout << "Set " << i << ": ";
        for (int j = 0; j < setSize; j++) {
            CacheEntry* entry = sets[i].frequencyList[j];
            if (entry != nullptr) {
                std::cout << "[" << entry->key << " -> " << entry->value << ", freq: " << entry->freq << "] ";
            }
        }
        std::cout << std::endl;
    }
}


BlockCache::BlockCache(int kLength, int vLength) : keyLength(kLength), valueLength(vLength), bufferUsed(0) {
    numSets = NSETS; 
    setSize = SETSIZE;
    totalEntries = numSets * setSize;
    allocateMemoryManaged((void**)&sets, numSets * sizeof(CacheSet));
    for (int i = 0; i < numSets; i++) {
        sets[i].initialize(setSize);
    }
    allocateMemoryManaged((void**)&keyBuffer, totalEntries * keyLength);
    allocateMemoryHost((void**)&valueBuffer, totalEntries * valueLength);
}