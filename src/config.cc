#include "config.h"
#include <iostream>
#include <stdio.h>  

uint64_t Config::getMemtableSize() {
    return memtableSize;
}

uint64_t Config::getBlkCacheSize() {
    return blkCacheSize;
}

int Config::getGrowFactor() {
    return growFactor;
}

int Config::getShrinkFactor() {
    return shrinkFactor;
}

void Config::setGrowFactor(int factor) {
    growFactor = factor;
}

void Config::setShrinkFactor(int factor) {
    shrinkFactor = factor;
}s

int Config::getNumMemtables() {
    return maxMemtables;
}

void Config::setMemtableSize(uint64_t size) {
    memtableSize = size;
}

void Config::setNumMemtables(int num) {
    maxMemtables = num;
}

uint64_t Config::getBatchSize() {
    return batchSize;
}

void Config::setBatchSize(uint64_t size) {
    batchSize = size;
}