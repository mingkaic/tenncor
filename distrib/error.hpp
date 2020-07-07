
#include <memory>
#include <string>

#include "fmts/fmts.hpp"

#ifndef ERROR_IERROR_HPP
#define ERROR_IERROR_HPP

namespace err
{

struct iError
{
	virtual ~iError (void) = default;

	virtual std::string to_string (void) const = 0;
};

struct ErrMsg final : public iError
{
	ErrMsg (const std::string& msg) : msg_(msg) {}

	template <typename ...ARGS>
	ErrMsg (const std::string& fmt, ARGS... args) :
		ErrMsg(fmts::sprintf(fmt, args...)) {}

	std::string to_string (void) const override
	{
		return msg_;
	}

private:
	std::string msg_;
};

using ErrptrT = std::shared_ptr<iError>;

}

#endif // ERROR_IERROR_HPP
