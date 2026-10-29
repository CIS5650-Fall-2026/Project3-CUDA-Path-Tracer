#include "timing.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace
{
bool enabled = false;
// A vector rather than a map so the report keeps first-use order.
std::vector<std::pair<std::string, double>> accumulators;
}  // namespace

void setTiming(bool e)
{
    enabled = e;
}

bool timingEnabled()
{
    return enabled;
}

void timingAdd(const char* name, double value)
{
    for (auto& a : accumulators)
    {
        if (a.first == name)
        {
            a.second += value;
            return;
        }
    }
    accumulators.emplace_back(name, value);
}

double timingGet(const char* name)
{
    for (const auto& a : accumulators)
    {
        if (a.first == name)
        {
            return a.second;
        }
    }
    return 0.0;
}

TimingScope::TimingScope(const char* n)
    : name(n), start(std::chrono::steady_clock::now())
{
}

TimingScope::~TimingScope()
{
    const auto end = std::chrono::steady_clock::now();
    timingAdd(name, std::chrono::duration<double, std::milli>(end - start).count());
}

void timingReport()
{
    for (const auto& a : accumulators)
    {
        printf("TIMING,%s,%.3f\n", a.first.c_str(), a.second);
    }
}
