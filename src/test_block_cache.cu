#include "block_cache.h"
#include <iostream>

int main() {
    // Define cache with fixed key and value lengths
    BlockCache cache(3, 2, 10, 20, 6);  // 3 sets, each with 2 entries, key length 10, value length 20, total 6 entries

    cache.insert("key1", "Value1");
    cache.insert("key2", "Value2");
    cache.insert("key3", "Value3");

    cache.displayCache();

    char* value = cache.get("key1");
    if (value) {
        std::cout << "Retrieved for key1: " << value << std::endl;
    } else {
        std::cout << "Key1 not found." << std::endl;
    }

    return 0;
}
