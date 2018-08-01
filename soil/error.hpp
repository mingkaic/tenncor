#include <sstream>
#include <string>
#include <vector>

#ifndef ERROR_HPP
#define ERROR_HPP

struct iErrArg
{
	operator std::string (void) const;

	virtual std::string to_string (void) const = 0;
};

template <typename T>
struct ErrArg : public iErrArg
{
	ErrArg (std::string key, T value) :
		key_(key), value_(value) {}

	std::string to_string (void) const override
	{
		std::stringstream ss;
		ss << key_ << "=" << value_;
		return ss.str();
	}

private:
	std::string key_;
	T value_;
};

template <typename T>
struct ErrVec : public iErrArg
{
	ErrVec (std::string key, std::vector<T> value) :
		key_(key), value_(value) {}

	std::string to_string (void) const override
	{
		std::stringstream ss;
		size_t n = value_.size();
		ss << key_ << "=[";
		if (n > 0)
		{
			ss << value_[0];
			for (size_t i = 1; i < n; ++i)
			{
				ss << "," << value_[i];
			}
		}
		ss << "]";
		return ss.str();
	}

private:
	std::string key_;
	std::vector<T> value_;
};

void handle_args (std::stringstream& ss, std::string entry);

template <typename... Args>
void handle_args (std::stringstream& ss, std::string entry, Args... args)
{
	handle_args(ss, entry);
	ss << ",";
	handle_args(ss, args...);
}

void handle_error (std::string msg);

template <typename... Args>
void handle_error (std::string msg, Args... args)
{
	std::stringstream ss;
	ss << msg << ":";
	handle_args(ss, args...);
	throw std::runtime_error(ss.str());
}

#endif /* ERROR_HPP */
