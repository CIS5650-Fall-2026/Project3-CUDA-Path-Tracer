#include "output_path.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace {

namespace fs = std::filesystem;

std::size_t gChecks = 0;
bool gCaseCollation = false;
bool gUnicodeCollation = false;

void require(bool condition, const std::string& message) {
    ++gChecks;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class DirectoryGuard {
public:
    explicit DirectoryGuard(fs::path path) : path_(std::move(path)) {}

    ~DirectoryGuard() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

private:
    fs::path path_;
};

class CurrentPathGuard {
public:
    CurrentPathGuard() : original_(fs::current_path()) {}

    ~CurrentPathGuard() {
        std::error_code ignored;
        fs::current_path(original_, ignored);
    }

private:
    fs::path original_;
};

void writeFile(const fs::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(output.is_open(), "could not create path-test fixture");
    output << contents;
    output.close();
    require(!output.fail(), "could not finish path-test fixture");
}

std::string readFile(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "could not open preserved path-test fixture");
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

void requireAliases(const fs::path& first,
                    const fs::path& second,
                    bool expected,
                    const std::string& caseName) {
    std::error_code error = std::make_error_code(std::errc::io_error);
    const bool result = output_path::aliases(first, second, error);
    require(!error, caseName + " produced a filesystem error: " + error.message());
    require(result == expected,
            caseName + (expected ? " missed an alias" : " reported a false alias"));
}

void testSymlinkedParentAndAbsentTail(const fs::path& directory) {
    const fs::path real = directory / "parent-real";
    const fs::path alias = directory / "parent-alias";
    fs::create_directories(real);
    fs::create_directory_symlink(real, alias);

    requireAliases(alias / "frame.json", real / "frame.json", true,
                   "symlinked parent");
    requireAliases(alias / "missing" / "frame.json",
                   real / "missing" / "frame.json", true,
                   "symlinked parent with absent tail");
    require(!fs::exists(real / "frame.json"),
            "symlink-parent reservation was not removed");
    require(!fs::exists(real / "missing"),
            "absent-tail reservations were not removed");
}

void testFinalSymlinks(const fs::path& directory) {
    const fs::path existingTarget = directory / "existing-target.json";
    const fs::path existingAlias = directory / "existing-alias.json";
    writeFile(existingTarget, "existing");
    fs::create_symlink(existingTarget.filename(), existingAlias);
    requireAliases(existingAlias, existingTarget, true,
                   "existing final symlink");

    const fs::path missingTarget = directory / "future-target.json";
    const fs::path danglingAlias = directory / "dangling-alias.json";
    fs::create_symlink(missingTarget.filename(), danglingAlias);
    require(!fs::exists(missingTarget),
            "dangling-symlink target unexpectedly exists");
    requireAliases(missingTarget, danglingAlias, true,
                   "dangling final output symlink");
    require(!fs::exists(missingTarget),
            "dangling-target reservation was not removed");
}

void testHardLinks(const fs::path& directory) {
    const fs::path first = directory / "hardlink-first.json";
    const fs::path second = directory / "hardlink-second.json";
    writeFile(first, "same inode");
    const std::string before = readFile(first);
    fs::create_hard_link(first, second);
    requireAliases(first, second, true, "hard links");
    require(readFile(first) == before && readFile(second) == before,
            "hard-link comparison modified an existing file");
}

void testRelativePaths(const fs::path& directory) {
    const fs::path real = directory / "relative-real";
    fs::create_directories(real);
    CurrentPathGuard restoreCurrentPath;
    fs::current_path(directory);
    requireAliases(fs::path("relative-real") / "frame.json",
                   real / "frame.json", true, "relative path");
}

void testDotDotAfterSymlink(const fs::path& directory) {
    const fs::path parent = directory / "dotdot-real";
    const fs::path nested = parent / "nested";
    const fs::path alias = directory / "dotdot-alias";
    fs::create_directories(nested);
    fs::create_directory_symlink(nested, alias);

    requireAliases(alias / ".." / "frame.json", parent / "frame.json", true,
                   "dot-dot after symlinked directory");
    requireAliases(alias / ".." / "frame.json", directory / "frame.json",
                   false, "dot-dot lexical-parent decoy");
}

void testOrdinaryPaths(const fs::path& directory) {
    const fs::path existingFirst = directory / "ordinary-first.json";
    const fs::path existingSecond = directory / "ordinary-second.json";
    writeFile(existingFirst, "first");
    writeFile(existingSecond, "second");
    requireAliases(existingFirst, existingSecond, false,
                   "distinct existing files");

    const fs::path missingFirst = directory / "not-created" / "first.json";
    const fs::path missingSecond = directory / "not-created" / "second.json";
    requireAliases(missingFirst, missingSecond, false,
                   "distinct nonexistent outputs");
    requireAliases(missingFirst, missingFirst, true,
                   "identical nonexistent output");
    require(!fs::exists(directory / "not-created"),
            "missing-parent reservations were not removed");
}

bool observesSameFilesystemName(const fs::path& directory,
                                const fs::path& firstName,
                                const fs::path& secondName) {
    fs::create_directory(directory);
    const fs::path first = directory / firstName;
    const fs::path second = directory / secondName;
    require(fs::create_directory(first),
            "could not create filesystem-collation probe");

    std::error_code error;
    const bool secondExists = fs::exists(second, error);
    require(!error, "could not inspect filesystem-collation probe");
    bool same = false;
    if (secondExists) {
        same = fs::equivalent(first, second, error);
        require(!error, "could not compare filesystem-collation probe");
    }

    require(fs::remove(first), "could not remove filesystem-collation probe");
    require(fs::remove(directory),
            "could not remove filesystem-collation probe directory");
    return same;
}

void testFilesystemCollation(const fs::path& directory) {
    gCaseCollation = observesSameFilesystemName(
        directory / "case-collation-probe", "Frame.JSON", "frame.json");
    gUnicodeCollation = observesSameFilesystemName(
        directory / "unicode-collation-probe", u8"caf\u00e9.json",
        u8"cafe\u0301.json");

    const fs::path caseParent = directory / "case-parent";
    fs::create_directory(caseParent);
    const fs::path caseMarker = caseParent / "marker.bin";
    const std::string markerContents("preserve\0case", 13);
    writeFile(caseMarker, markerContents);
    const std::string markerBefore = readFile(caseMarker);
    const fs::path upperLeaf = caseParent / "Frame.JSON";
    const fs::path lowerLeaf = caseParent / "frame.json";
    requireAliases(upperLeaf, lowerLeaf, gCaseCollation,
                   "absent case-variant leaves");
    require(!fs::exists(upperLeaf) && !fs::exists(lowerLeaf),
            "case-variant leaf reservations were not removed");
    require(readFile(caseMarker) == markerBefore,
            "case-collation probe modified a preexisting file");

    const fs::path realDirectory = directory / "RealDir";
    fs::create_directory(realDirectory);
    const fs::path realMarker = realDirectory / "marker.bin";
    writeFile(realMarker, markerContents);
    const fs::path alternateDirectory = directory / "realdir";
    requireAliases(realDirectory / "same.json",
                   alternateDirectory / "same.json", gCaseCollation,
                   "existing parent case variant");
    require(!fs::exists(realDirectory / "same.json"),
            "existing-parent leaf reservation was not removed");
    std::error_code error;
    const bool alternateExists = fs::exists(alternateDirectory, error);
    require(!error && alternateExists == gCaseCollation,
            "existing-parent cleanup changed directory identity");
    require(readFile(realMarker) == markerContents,
            "existing-parent comparison modified a preexisting file");

    const fs::path unicodeParent = directory / "unicode-parent";
    fs::create_directory(unicodeParent);
    const fs::path unicodeMarker = unicodeParent / "marker.bin";
    writeFile(unicodeMarker, markerContents);
    const fs::path composed = unicodeParent / u8"caf\u00e9.json";
    const fs::path decomposed = unicodeParent / u8"cafe\u0301.json";
    requireAliases(composed, decomposed, gUnicodeCollation,
                   "absent Unicode-normalization variants");
    require(!fs::exists(composed) && !fs::exists(decomposed),
            "Unicode leaf reservations were not removed");
    require(readFile(unicodeMarker) == markerContents,
            "Unicode-collation probe modified a preexisting file");
}

void testFilesystemFailures(const fs::path& directory) {
    const fs::path regular = directory / "regular-parent";
    writeFile(regular, "not a directory");
    std::error_code error;
    bool result = output_path::aliases(
        regular / "child.json", directory / "other.json", error);
    require(!result, "regular-file parent comparison unexpectedly aliased");
    require(error == std::errc::not_a_directory,
            "regular-file parent did not report not-a-directory");
    require(!error.message().empty(),
            "regular-file parent failure did not provide a diagnostic");

    const fs::path cycleA = directory / "cycle-a";
    const fs::path cycleB = directory / "cycle-b";
    fs::create_symlink(cycleB.filename(), cycleA);
    fs::create_symlink(cycleA.filename(), cycleB);
    error.clear();
    result = output_path::aliases(cycleA, directory / "other.json", error);
    require(!result, "symbolic-link cycle unexpectedly aliased");
    require(error == std::errc::too_many_symbolic_link_levels,
            "symbolic-link cycle did not report its resolution failure");
    require(!error.message().empty(),
            "symbolic-link cycle did not provide a diagnostic");
}

} // namespace

int main() {
    try {
        const std::uint64_t unique = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const fs::path directory = fs::temp_directory_path() /
            ("cis565-output-path-tests-" + std::to_string(unique));
        fs::create_directories(directory);
        DirectoryGuard cleanup(directory);

        std::size_t groups = 0;
        testSymlinkedParentAndAbsentTail(directory);
        ++groups;
        testFinalSymlinks(directory);
        ++groups;
        testHardLinks(directory);
        ++groups;
        testRelativePaths(directory);
        ++groups;
        testDotDotAfterSymlink(directory);
        ++groups;
        testOrdinaryPaths(directory);
        ++groups;
        testFilesystemCollation(directory);
        ++groups;
        testFilesystemFailures(directory);
        ++groups;

        std::cout << "output_path_tests: PASS (" << groups << " groups, "
                  << gChecks << " checks, case-collation="
                  << (gCaseCollation ? "on" : "off")
                  << ", unicode-collation="
                  << (gUnicodeCollation ? "on" : "off") << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "output_path_tests: FAIL after " << gChecks
                  << " checks: " << error.what() << '\n';
        return 1;
    }
}
