#pragma once

#include <chrono>

// Named host timers behind --timing, in ms unless the name ends in _bytes or _mb; profiling/'s sweep script parses the report.

void setTiming(bool enabled);
bool timingEnabled();

void timingAdd(const char* name, double value);
// 0 when the accumulator does not exist
double timingGet(const char* name);

// Adds the time from construction to destruction to name.
class TimingScope
{
public:
    explicit TimingScope(const char* name);
    ~TimingScope();
    TimingScope(const TimingScope&) = delete;
    TimingScope& operator=(const TimingScope&) = delete;

private:
    const char* name;
    std::chrono::steady_clock::time_point start;
};

// Prints every accumulator in first-use order as "TIMING,<name>,<value>".
void timingReport();
