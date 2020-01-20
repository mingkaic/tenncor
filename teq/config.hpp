#ifndef TEQ_CONFIG_HPP
#define TEQ_CONFIG_HPP

#include "logs/logs.hpp"

#include "estd/config.hpp"

namespace teq
{

extern std::shared_ptr<estd::iConfig> global_config;

const std::string logger_key = "logger";

/// Log at trace level
void trace (std::string msg);

/// Log at debug level
void debug (std::string msg);

/// Log at info level
void info (std::string msg);

/// Log at warn level
void warn (std::string msg);

/// Log at error level
void error (std::string msg);

/// Log at fatal level
void fatal (std::string msg);

/// Log at trace level with arguments
template <typename... ARGS>
void tracef (std::string format, ARGS... args)
{
	trace(fmts::sprintf(format, args...));
}

/// Log at debug level with arguments
template <typename... ARGS>
void debugf (std::string format, ARGS... args)
{
	debug(fmts::sprintf(format, args...));
}

/// Log at info level with arguments
template <typename... ARGS>
void infof (std::string format, ARGS... args)
{
	info(fmts::sprintf(format, args...));
}

/// Log at warn level with arguments
template <typename... ARGS>
void warnf (std::string format, ARGS... args)
{
	warn(fmts::sprintf(format, args...));
}

/// Log at error level with arguments
template <typename... ARGS>
void errorf (std::string format, ARGS... args)
{
	error(fmts::sprintf(format, args...));
}

/// Log at fatal level with arguments
template <typename... ARGS>
void fatalf (std::string format, ARGS... args)
{
	fatal(fmts::sprintf(format, args...));
}

}

#endif // TEQ_CONFIG_HPP
