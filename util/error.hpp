///
/// error.hpp
/// util
///
/// Purpose:
/// Define error handling mechanisms
///

#include "util/strify.hpp"

#ifndef UTIL_ERROR_HPP
#define UTIL_ERROR_HPP

namespace util
{

/// Interface error argument for implicit string conversion
struct iErrArg
{
	operator std::string (void) const
	{
		std::stringstream ss;
		streamify(ss);
		return ss.str();
	}

	/// Stream argument representation to string stream
	virtual void streamify (std::stringstream& ss) const = 0;
};

/// Templated error argument for capturing key-value info
template <typename T>
struct ErrArg : public iErrArg
{
	ErrArg (std::string key, T value) :
		key_(key), value_(value) {}

	/// Implementation of iErrArg
	void streamify (std::stringstream& ss) const override
	{
		ss << util::BEGIN << key_ << util::DELIM;
		to_stream(ss, value_);
		ss << util::END;
	}

private:
	/// Key of the argument used to describe what the value is
	std::string key_;

	/// Value of the argument to indicate error state
	T value_;
};

/// Stream string converted error argument to stream
void handle_args (std::stringstream& ss, std::string entry);

/// Stream string converted variadic error argument to stream
template <typename... Args>
void handle_args (std::stringstream& ss, std::string entry, Args... args)
{
	handle_args(ss, entry);
	handle_args(ss, args...);
}

/// Throw error of a plain message
void handle_error (std::string msg);

/// Throw error of a plain message with arguments
template <typename... Args>
void handle_error (std::string msg, Args... args)
{
	std::stringstream ss;
	ss << msg;
	handle_args(ss, args...);
	throw std::runtime_error(ss.str());
}

}

#endif // UTIL_ERROR_HPP
