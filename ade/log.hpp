///
/// log.hpp
/// ade
///
/// Purpose:
/// Define log message handling interface
///

#include <memory>

#include "ade/string.hpp"

#ifndef ADE_LOG_HPP
#define ADE_LOG_HPP

namespace ade
{

/// Interface of logger used in ADE
struct iLogger
{
	virtual ~iLogger (void) = default;

	/// Warn user of message regarding poor decisions
	virtual void warn (std::string msg) const = 0;

	/// Notify user of message regarding recoverable error
	virtual void error (std::string msg) const = 0;

	/// Notify user of message regarding fatal error, then finish him
	virtual void fatal (std::string msg) const = 0;
};

/// Set input logger for ADE global logger
void set_logger (std::shared_ptr<iLogger> logger);

/// Get reference to ADE global logger
const iLogger& get_logger (void);

/// C++ version of sprintf
template <typename... ARGS>
std::string sprintf (std::string format, ARGS... args)
{
	size_t size = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
    char buf[size];
    std::snprintf(buf, size, format.c_str(), args...);
	return std::string(buf, buf + size - 1);
}

/// Warn using global logger
void warn (std::string msg);

/// Error using global logger
void error (std::string msg);

/// Fatal using global logger
void fatal (std::string msg);

/// Warn using global logger with arguments
template <typename... ARGS>
void warnf (std::string format, ARGS... args)
{
    get_logger().warn(ade::sprintf(format, args...));
}

/// Error using global logger with arguments
template <typename... ARGS>
void errorf (std::string format, ARGS... args)
{
    get_logger().error(ade::sprintf(format, args...));
}

/// Fatal using global logger with arguments
template <typename... ARGS>
void fatalf (std::string format, ARGS... args)
{
    get_logger().fatal(ade::sprintf(format, args...));
}

}

#endif // ADE_LOG_HPP
