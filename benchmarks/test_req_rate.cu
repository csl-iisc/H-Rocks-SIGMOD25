#include "hrocksdb.h"
#include "config.h"

#include <chrono>
#include <thread>
#include <random>
#include <string>
#include <iostream>
#include <getopt.h>
#include <unistd.h>
#include <cassert>

using Clock = std::chrono::steady_clock;

static std::string rand_fixed(size_t len, std::mt19937_64& rng) {
    static const char alphanum[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<size_t> dist(0, sizeof(alphanum) - 2);
    std::string s;
    s.resize(len);
    for (size_t i = 0; i < len; ++i) s[i] = alphanum[dist(rng)];
    return s;
}

int main(int argc, char** argv) {
    // Defaults
    uint64_t rate = 0;          // requests per second (required)
    size_t   klen = 8;          // key size
    size_t   vlen = 8;          // value size
    int      dur_s = 5;         // duration seconds (default 5)
    std::string path = "test";  // relative -> resolved to /pmem/test by HOpen()

    int opt;
    while ((opt = getopt(argc, argv, "n:k:v:d:p:")) != -1) {
        switch (opt) {
            case 'n': rate = std::stoull(optarg); break;
            case 'k': klen = static_cast<size_t>(std::stoul(optarg)); break;
            case 'v': vlen = static_cast<size_t>(std::stoul(optarg)); break;
            case 'd': dur_s = std::stoi(optarg); break;  // optional
            case 'p': path = std::string(optarg); break; // optional
            case '?':
            default:
                std::cerr << "Usage: " << argv[0]
                          << " -n <req/s> -k <key-bytes> -v <value-bytes>"
                          << " [-d <seconds>] [-p <db-path>]\n";
                return 1;
        }
    }

    if (rate == 0) {
        std::cerr << "Error: -n <req/s> must be > 0\n";
        return 1;
    }
    if (klen == 0 || vlen == 0) {
        std::cerr << "Error: -k and -v must be > 0\n";
        return 1;
    }
    if (dur_s < 1) dur_s = 1;

    std::cout << "Target rate: " << rate << " req/s  | key=" << klen
              << " | value=" << vlen << " | duration=" << dur_s
              << "s | path=" << path << "\n";

    // DB setup
    Config conf;
    // (Optional) tune these if you want fewer commits/chunks:
    // conf.setMemtableSize(5'000'000);
    // conf.setNumMemtables(64);

    auto* hdb = new HRocksDB(conf);
    hdb->HOpen(path);

    // Random generators (deterministic seed for reproducibility)
    std::mt19937_64 rng{123456789};

    const auto t0   = Clock::now();
    const auto tend = t0 + std::chrono::seconds(dur_s);

    // Period in nanoseconds. If rate is very high, period could be 0.
    const long double period_ns_ld = 1e9L / static_cast<long double>(rate);
    const auto period_ns = (period_ns_ld < 1.0L)
                               ? std::chrono::nanoseconds::zero()
                               : std::chrono::nanoseconds(static_cast<int64_t>(period_ns_ld));

    uint64_t sent = 0;
    auto next_deadline = t0;

    while (true) {
        auto now = Clock::now();
        if (now >= tend) break;

        // Issue one request
        std::string key   = rand_fixed(klen, rng);
        std::string value = rand_fixed(vlen, rng);
        hdb->Put(key, value);
        ++sent;

        // Schedule next send
        if (period_ns == std::chrono::nanoseconds::zero()) {
            // No sleeping if target period < 1 ns; just keep pushing
            continue;
        }
        next_deadline += period_ns;

        // If we fell behind, catch up by skipping sleeps until we're ahead
        now = Clock::now();
        if (next_deadline > now) {
            std::this_thread::sleep_until(next_deadline);
        } else {
            // We're behind; realign to now to avoid unbounded drift
            next_deadline = now;
        }
    }

    hdb->Close();
    delete hdb;

    const auto elapsed_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - t0).count();
    const double achieved = (elapsed_s > 0.0) ? (sent / elapsed_s) : 0.0;

    std::cout << "Sent " << sent << " requests in " << elapsed_s
              << " s  (achieved ~" << achieved << " req/s)\n";
    return 0;
}
