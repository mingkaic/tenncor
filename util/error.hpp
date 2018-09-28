#include "util/strify.hpp"

#ifndef UTIL_ERROR_HPP
#define UTIL_ERROR_HPP

namespace util
{

struct iErrArg
{
	operator std::string (void) const
	{
		std::stringstream ss;
		streamify(ss);
		return ss.str();
	}

	virtual void streamify (std::stringstream& ss) const = 0;
};

template <typename T>
struct ErrArg : public iErrArg
{
	ErrArg (std::string key, T value) :
		key_(key), value_(value) {}

	void streamify (std::stringstream& ss) const override
	{
		ss << util::BEGIN << key_ << util::DELIM;
		to_stream(ss, value_);
		ss << util::END;
	}

private:
	std::string key_;
	T value_;
};

void handle_args (std::stringstream& ss, std::string entry);

template <typename... Args>
void handle_args (std::stringstream& ss, std::string entry, Args... args)
{
	handle_args(ss, entry);
	handle_args(ss, args...);
}

void handle_error (std::string msg);

template <typename... Args>
void handle_error (std::string msg, Args... args)
{
	std::stringstream ss;
	ss << msg;
	handle_args(ss, args...);
	throw std::runtime_error(ss.str());
}

}

#endif /* UTIL_ERROR_HPP */
