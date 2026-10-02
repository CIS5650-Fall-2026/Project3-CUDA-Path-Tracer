#include "output_path.h"

#include <deque>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace output_path {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kMaximumSymlinkExpansions = 64;

void removeEmptyDirectoryDuringUnwind(const fs::path& path) noexcept {
#if defined(_WIN32)
    ::RemoveDirectoryW(path.c_str());
#else
    ::rmdir(path.c_str());
#endif
}

void prependComponents(std::deque<fs::path>& pending, const fs::path& path) {
    std::deque<fs::path> components;
    for (const fs::path& component : path.relative_path()) {
        components.push_back(component);
    }
    while (!components.empty()) {
        pending.push_front(std::move(components.back()));
        components.pop_back();
    }
}

fs::file_status symlinkStatusAllowingMissing(const fs::path& path,
                                             std::error_code& error) {
    fs::file_status status = fs::symlink_status(path, error);
    if (error == std::errc::no_such_file_or_directory) {
        error.clear();
        status = fs::file_status(fs::file_type::not_found);
    }
    return status;
}

fs::file_status statusAllowingMissing(const fs::path& path) {
    try {
        return fs::status(path);
    } catch (const fs::filesystem_error& failure) {
        if (failure.code() == std::errc::no_such_file_or_directory) {
            return fs::file_status(fs::file_type::not_found);
        }
        throw;
    }
}

bool resolveIdentity(const fs::path& input,
                     fs::path& identity,
                     std::error_code& error) {
    error.clear();
    const fs::path absolute = fs::absolute(input, error);
    if (error) {
        return false;
    }

    fs::path resolved = absolute.root_path();
    std::deque<fs::path> pending;
    prependComponents(pending, absolute);
    std::size_t symlinkExpansions = 0;

    while (!pending.empty()) {
        fs::path component = std::move(pending.front());
        pending.pop_front();

        if (component.empty() || component == fs::path(".")) {
            continue;
        }
        if (component == fs::path("..")) {
            const fs::path parent = resolved.parent_path();
            if (!parent.empty()) {
                resolved = parent;
            }
            continue;
        }

        const fs::path candidate = resolved / component;
        const fs::file_status status =
            symlinkStatusAllowingMissing(candidate, error);
        if (error) {
            return false;
        }

        if (fs::is_symlink(status)) {
            if (++symlinkExpansions > kMaximumSymlinkExpansions) {
                error = std::make_error_code(
                    std::errc::too_many_symbolic_link_levels);
                return false;
            }

            const fs::path target = fs::read_symlink(candidate, error);
            if (error) {
                return false;
            }
            if (target.is_absolute()) {
                resolved = target.root_path();
            } else if (target.has_root_directory()) {
                // A Windows root-relative target keeps the drive of the
                // link's parent while replacing the directory portion.
                // POSIX rooted paths are already absolute.
                resolved = resolved.root_name() / target.root_directory();
            }
            prependComponents(pending, target);
            continue;
        }

        resolved = candidate;
        if (fs::exists(status) && !fs::is_directory(status) &&
            !pending.empty()) {
            error = std::make_error_code(std::errc::not_a_directory);
            return false;
        }
    }

    identity = std::move(resolved);
    return true;
}

class DirectoryReservations {
public:
    struct Range {
        std::size_t begin = 0;
        std::size_t end = 0;
    };

    DirectoryReservations(const fs::path& first, const fs::path& second) {
        entries_.reserve(componentCount(first) + componentCount(second));
        first_ = append(first);
        second_ = append(second);
    }

    ~DirectoryReservations() noexcept {
        if (cleanupFinished_) {
            return;
        }
        // An exception may arrive after an exclusive creation but before the
        // normal cleanup path. Every owned entry was absent immediately
        // before this object created it. The native operation uses each
        // prebuilt path buffer, succeeds only for an empty directory, and
        // cannot throw while another exception is unwinding.
        for (auto entry = entries_.rbegin(); entry != entries_.rend(); ++entry) {
            if (!entry->owned) {
                continue;
            }
            removeEmptyDirectoryDuringUnwind(entry->path);
        }
    }

    const Range& first() const noexcept { return first_; }
    const Range& second() const noexcept { return second_; }

    bool reserve(const Range& range, std::error_code& error) {
        error.clear();
        try {
            for (std::size_t index = range.begin; index < range.end; ++index) {
                Entry& entry = entries_[index];
                fs::file_status status = statusAllowingMissing(entry.path);
                if (!fs::exists(status)) {
                    bool created = false;
                    try {
                        created = fs::create_directory(entry.path);
                    } catch (const fs::filesystem_error& failure) {
                        if (failure.code() != std::errc::file_exists) {
                            throw;
                        }
                    }
                    if (created) {
                        // Every entry and path copy was prepared before the
                        // first create_directory call. This ownership bit is
                        // the only operation between creation and a recorded
                        // cleanup obligation.
                        entry.owned = true;
                        status = fs::file_status(fs::file_type::directory);
                    } else {
                        status = statusAllowingMissing(entry.path);
                        if (!fs::exists(status)) {
                            error = std::make_error_code(
                                std::errc::no_such_file_or_directory);
                            return false;
                        }
                    }
                }

                if (index + 1 < range.end && !fs::is_directory(status)) {
                    error = std::make_error_code(std::errc::not_a_directory);
                    return false;
                }
            }
        } catch (const fs::filesystem_error& failure) {
            error = failure.code();
            return false;
        }
        return true;
    }

    bool cleanup(std::error_code& error) {
        error.clear();
        std::error_code firstFailure;
        for (auto entry = entries_.rbegin(); entry != entries_.rend(); ++entry) {
            if (!entry->owned) {
                continue;
            }
            fs::file_status status;
            try {
                status = fs::symlink_status(entry->path);
            } catch (const fs::filesystem_error& failure) {
                if (failure.code() ==
                    std::errc::no_such_file_or_directory) {
                    entry->owned = false;
                    continue;
                }
                if (!firstFailure) {
                    firstFailure = failure.code();
                }
                continue;
            }
            if (!fs::exists(status)) {
                entry->owned = false;
                continue;
            }
            if (!fs::is_directory(status)) {
                if (!firstFailure) {
                    firstFailure = std::make_error_code(
                        std::errc::operation_not_permitted);
                }
                continue;
            }
            bool removed = false;
            try {
                removed = fs::remove(entry->path);
            } catch (const fs::filesystem_error& failure) {
                if (!firstFailure) {
                    firstFailure = failure.code();
                }
                continue;
            }
            if (removed) {
                entry->owned = false;
            } else if (!firstFailure) {
                firstFailure = std::make_error_code(std::errc::io_error);
            }
        }
        cleanupFinished_ = true;
        error = firstFailure;
        return !error;
    }

private:
    struct Entry {
        fs::path path;
        bool owned = false;
    };

    static std::size_t componentCount(const fs::path& identity) {
        std::size_t count = 0;
        for (const fs::path& component : identity.relative_path()) {
            if (!component.empty()) {
                ++count;
            }
        }
        return count;
    }

    Range append(const fs::path& identity) {
        Range range{entries_.size(), entries_.size()};
        fs::path current = identity.root_path();
        for (const fs::path& component : identity.relative_path()) {
            if (component.empty()) {
                continue;
            }
            current /= component;
            entries_.push_back(Entry{current, false});
        }
        range.end = entries_.size();
        return range;
    }

    std::vector<Entry> entries_;
    Range first_;
    Range second_;
    bool cleanupFinished_ = false;
};

} // namespace

bool aliases(const fs::path& first,
             const fs::path& second,
             std::error_code& error) {
    fs::path firstIdentity;
    if (!resolveIdentity(first, firstIdentity, error)) {
        return false;
    }
    fs::path secondIdentity;
    if (!resolveIdentity(second, secondIdentity, error)) {
        return false;
    }

    DirectoryReservations reservations(firstIdentity, secondIdentity);
    bool result = false;
    bool operationSucceeded = reservations.reserve(reservations.first(), error);
    if (operationSucceeded) {
        operationSucceeded = reservations.reserve(reservations.second(), error);
    }
    if (operationSucceeded) {
        try {
            result = fs::equivalent(firstIdentity, secondIdentity);
        } catch (const fs::filesystem_error& failure) {
            error = failure.code();
            operationSucceeded = false;
        }
    }

    const std::error_code operationError = error;
    std::error_code cleanupError;
    reservations.cleanup(cleanupError);
    if (cleanupError) {
        error = cleanupError;
        return false;
    }
    error = operationError;
    return operationSucceeded && result;
}

} // namespace output_path
