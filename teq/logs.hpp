#ifndef TEQ_LOGS_HPP
#define TEQ_LOGS_HPP

#include "logs/logs.hpp"

#include "teq/config.hpp"

namespace teq
{

const std::string logger_key = "logger";

/// Return log level
std::string get_log_level (void);

/// Set log level
void set_log_level (const std::string& log_level);

/// Log at trace level
void trace (const std::string& msg);

/// Log at debug level
void debug (const std::string& msg);

/// Log at info level
void info (const std::string& msg);

/// Log at warn level
void warn (const std::string& msg);

/// Log at error level
void error (const std::string& msg);

/// Log at fatal level
void fatal (const std::string& msg);

/// Log at trace level with arguments
template <typename... ARGS>
void tracef (const std::string& format, ARGS... args)
{
	trace(fmts::sprintf(format, args...));
}

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

/// Log at fatal level with arguments
template <typename... ARGS>
void fatalf (const std::string& format, ARGS... args)
{
	fatal(fmts::sprintf(format, args...));
}

#define LOG_INIT(LOG_TYPE)::config::global_config.add_entry<LOG_TYPE>(teq::logger_key)

}

#endif // TEQ_LOGS_HPP
