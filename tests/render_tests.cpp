#include "pathtrace.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

int failures = 0;

void expect(bool condition, const char* message)
{
    if (!condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

template<typename Exception, typename Function>
void expectThrows(Function&& function, const char* message)
{
    try
    {
        function();
        expect(false, message);
    }
    catch (const Exception&)
    {
    }
    catch (...)
    {
        expect(false, message);
    }
}

template<typename Exception, typename Function>
void expectThrowsMessage(Function&& function, const char* expected,
                         const char* message)
{
    try
    {
        function();
        expect(false, message);
    }
    catch (const Exception& error)
    {
        expect(std::string(error.what()) == expected, message);
    }
    catch (...)
    {
        expect(false, message);
    }
}

} // namespace

int main()
{
    const RendererConfig defaults;
    expect(defaults.wavefront && defaults.rr && defaults.rrStart == 3,
        "new renderer defaults must enable wavefront and Russian roulette");

    RendererConfig config;
    config.seed = 0x0123456789abcdefULL;
    config.maxDepth = 17;
    config.aa = false;
    config.nee = true;
    config.useBvh = false;
    config.compact = true;
    config.sort = true;
    config.wavefront = false;
    config.rr = false;
    config.rrStart = 19;
    config.benchmark = true;
    config.debug = RendererDebugMode::Uv;
    pathtraceSetConfig(config);
    const RendererConfig read = pathtraceGetConfig();
    expect(read.seed == config.seed && read.maxDepth == config.maxDepth
        && read.aa == config.aa && read.nee == config.nee
        && read.useBvh == config.useBvh && read.compact == config.compact
        && read.sort == config.sort && read.wavefront == config.wavefront
        && read.rr == config.rr && read.rrStart == config.rrStart
        && read.benchmark == config.benchmark
        && read.debug == config.debug,
        "renderer configuration round trip");

    RendererConfig valid = config;
    valid.rrStart = 1;
    pathtraceSetConfig(valid);
    expect(pathtraceGetConfig().rrStart == 1,
        "minimum roulette start must be accepted");
    valid.rrStart = 4096;
    pathtraceSetConfig(valid);
    expect(pathtraceGetConfig().rrStart == 4096,
        "maximum roulette start must be accepted even beyond maxDepth");

    RendererConfig invalid = config;
    invalid.maxDepth = 0;
    expectThrows<std::invalid_argument>([&]() { pathtraceSetConfig(invalid); },
        "zero depth must be rejected");
    invalid.maxDepth = 4097;
    expectThrows<std::invalid_argument>([&]() { pathtraceSetConfig(invalid); },
        "oversized depth must be rejected");
    invalid = config;
    invalid.rrStart = 0;
    expectThrows<std::invalid_argument>([&]() { pathtraceSetConfig(invalid); },
        "zero roulette start must be rejected even when roulette is disabled");
    invalid.rrStart = 4097;
    expectThrows<std::invalid_argument>([&]() { pathtraceSetConfig(invalid); },
        "oversized roulette start must be rejected even when roulette is disabled");
    invalid = config;
    invalid.wavefront = true;
    expectThrowsMessage<std::invalid_argument>(
        [&]() { pathtraceSetConfig(invalid); },
        "--sort on requires --wavefront off",
        "material sorting with wavefront scheduling must have an actionable rejection");
    invalid = config;
    invalid.debug = static_cast<RendererDebugMode>(99);
    expectThrows<std::invalid_argument>([&]() { pathtraceSetConfig(invalid); },
        "invalid debug mode must be rejected");
    expectThrows<std::logic_error>([]() { (void)pathtraceSampleCount(); },
        "sample count before initialization must be rejected");

    RendererValidationReport report;
    report.passed = false;
    report.checks.push_back({"quote\"newline\n", false, "detail\\value", {}});
    const std::string json = pathtraceValidationJson(report);
    expect(json ==
        "{\"passed\":false,\"checks\":[{\"name\":\"quote\\\"newline\\n\","
        "\"passed\":false,\"detail\":\"detail\\\\value\"}]}",
        "validation JSON must be deterministic and escaped");

    if (failures == 0)
    {
        std::cout << "render_tests: PASS\n";
        return EXIT_SUCCESS;
    }
    std::cerr << "render_tests: " << failures << " failure(s)\n";
    return EXIT_FAILURE;
}
