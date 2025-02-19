#include "gmemtable.h"
#include "config.h"
#include "helper.cuh"

void GMemtable::freeGMemtable() {
    // check if this is free or not
    freeMemory(keys);
    freeMemory(valuePointer);
    freeMemory(opID);
}

void CMemtable::freeCMemtable() {
    delete keys;
    delete valuePointer;
    delete opID;
}