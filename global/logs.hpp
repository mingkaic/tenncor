
#include "logs/logs.hpp"

#include "global/config.hpp"

#ifndef GLOBAL_LOGS_HPP
#define GLOBAL_LOGS_HPP

namespace global
{

void set_logger (logs::iLogger* logger, global::CfgMapptrT ctx = context());

logs::iLogger& get_logger (const global::CfgMapptrT& ctx = context());

/// Set log level
void set_log_level (const std::string& log_level,
	global::CfgMapptrT ctx = context());

/// Return log level
std::string get_log_level (const global::CfgMapptrT& ctx = context());

/// Log at debug level
void debug (const std::string& msg);

/// Log at info level
void info (const std::string& msg);

/// Log at warn level
void warn (const std::string& msg);

/// Log at error level
void error (const std::string& msg);

/// Log at throw level
void throw_err (const std::string& msg);

/// Log at fatal level
void fatal (const std::string& msg);

/// Log at debug level with arguments
template <typename... ARGS>
void debugf (const std::string& format, ARGS... args)
{
	debug(fmts::sprintf(format, args...));
}

/// Log at info level with arguments
template <typename... ARGS>
void infof (const std::string& format, ARGS... args)
{
	info(fmts::sprintf(format, args...));
}

/// Log at warn level with arguments
template <typename... ARGS>
void warnf (const std::string& format, ARGS... args)
{
	warn(fmts::sprintf(format, args...));
}

/// Log at error level with arguments
template <typename... ARGS>
void errorf (const std::string& format, ARGS... args)
{
	error(fmts::sprintf(format, args...));
}

/// Log at throw_error level with arguments
template <typename... ARGS>
void throw_errf (const std::string& format, ARGS... args)
{
	throw_err(fmts::sprintf(format, args...));
}

/// Log at fatal level with arguments
template <typename... ARGS>
void fatalf (const std::string& format, ARGS... args)
{
	fatal(fmts::sprintf(format, args...));
}

}

#endif // GLOBAL_LOGS_HPP
