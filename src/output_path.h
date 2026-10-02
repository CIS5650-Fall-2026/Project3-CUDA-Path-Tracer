#pragma once

#include <filesystem>
#include <system_error>

// Filesystem-aware identity checks for prospective output paths.  These are
// kept outside the headless executable so host tests exercise the same guard
// used before rendering begins.
namespace output_path {

// Returns true when two paths name the same existing filesystem object or
// would name the same location after resolving existing (including dangling)
// symbolic links and the destination filesystem's actual name collation.
// Missing components are briefly reserved as empty directories and removed
// before return; existing files are never modified.  On a reservation,
// inspection, comparison, or cleanup failure, returns false and stores the
// reason in error.
bool aliases(const std::filesystem::path& first,
             const std::filesystem::path& second,
             std::error_code& error);

} // namespace output_path
