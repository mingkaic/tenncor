///
/// log.hpp
/// ade
///
/// Purpose:
/// Define log message handling interface
///

#include <iostream>
#include <memory>

#include "ade/log/string.hpp"

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

/// String tagged prepending a warning message in default logger
const std::string warn_tag = "[WARNING]:";

/// String tagged prepending an error message in default logger
const std::string err_tag = "[ERROR]:";

/// Default implementation of iLogger used in ADE
struct DefLogger : public iLogger
{
	/// Implementation of iLogger
	void warn (std::string msg) const override
	{
		std::cerr << warn_tag << msg << std::endl;
	}

	/// Implementation of iLogger
	void error (std::string msg) const override
	{
		std::cerr << err_tag << msg << std::endl;
	}

	/// Implementation of iLogger
	void fatal (std::string msg) const override
	{
		throw std::runtime_error(msg);
	}
};

/// Set input logger for ADE global logger
void set_logger (std::shared_ptr<iLogger> logger);

/// Get reference to ADE global logger
const iLogger& get_logger (void);

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
	warn(ade::sprintf(format, args...));
}

/// Error using global logger with arguments
template <typename... ARGS>
void errorf (std::string format, ARGS... args)
{
	error(ade::sprintf(format, args...));
}

/// Fatal using global logger with arguments
template <typename... ARGS>
void fatalf (std::string format, ARGS... args)
{
	fatal(ade::sprintf(format, args...));
}

}

#endif // ADE_LOG_HPP
