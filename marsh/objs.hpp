#include <functional>

#include "estd/estd.hpp"
#include "fmts/fmts.hpp"

#include "marsh/imarshal.hpp"

#ifndef MARSH_OBJS_HPP
#define MARSH_OBJS_HPP

namespace marsh
{

struct iObject
{
	virtual ~iObject (void) = default;

	virtual size_t class_code (void) const = 0;

	virtual std::string to_string (void) const = 0;

	virtual bool equals (const iObject& other) const = 0;

	virtual void accept (iMarshaler& marshaler) const = 0;
};

using ObjptrT = std::unique_ptr<iObject>;

struct iNumber : public iObject
{
	virtual ~iNumber (void) = default;

	virtual double to_float64 (void) const = 0;

	virtual int64_t to_int64 (void) const = 0;

	void accept (iMarshaler& marshaler) const override
	{
		marshaler.marshal(*this);
	}
};

template <typename T,
	typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
struct Number final : public iNumber
{
	Number (T val) : val_(val) {}

	size_t class_code (void) const override
	{
		static const std::type_info& tp = typeid(T);
		return tp.hash_code();
	}

	std::string to_string (void) const override
	{
		return fmts::to_string(val_);
	}

	bool equals (const iObject& other) const override
	{
		if (other.class_code() != this->class_code())
		{
			return false;
		}
		return val_ == static_cast<const Number<T>*>(&other)->val_;
	}

	double to_float64 (void) const override
	{
		return val_;
	}

	int64_t to_int64 (void) const override
	{
		return val_;
	}

	T val_;
};

struct iArray : public iObject
{
	virtual ~iArray (void) = default;

	virtual void foreach (std::function<void(ObjptrT&)> consume) = 0;

	virtual void foreach (std::function<void(const ObjptrT&)> consume) const = 0;

	virtual size_t size (void) const = 0;

	void accept (iMarshaler& marshaler) const override
	{
		marshaler.marshal(*this);
	}
};

struct ObjArray final : public iArray
{
	size_t class_code (void) const override
	{
		static const std::type_info& tp = typeid(ObjArray);
		return tp.hash_code();
	}

	std::string to_string (void) const override
	{
		std::vector<std::string> strs;
		strs.reserve(contents_.size());
		for (auto& c : contents_)
		{
			strs.push_back(c->to_string());
		}
		return fmts::to_string(strs.begin(), strs.end());
	}

	bool equals (const iObject& other) const override
	{
		if (other.class_code() != this->class_code())
		{
			return false;
		}
		auto& ocontents = static_cast<const ObjArray*>(&other)->contents_;
		size_t n = contents_.size();
		if (ocontents.size() != n)
		{
			return false;
		}
		for (size_t i = 0; i < n; ++i)
		{
			if (false == contents_[i]->equals(*ocontents[i]))
			{
				return false;
			}
		}
		return true;
	}

	size_t size (void) const override
	{
		return contents_.size();
	}

	void foreach (std::function<void(ObjptrT&)> consume) override
	{
		for (ObjptrT& obj : contents_)
		{
			consume(obj);
		}
	}

	void foreach (std::function<void(const ObjptrT&)> consume) const override
	{
		for (const ObjptrT& obj : contents_)
		{
			consume(obj);
		}
	}

	std::vector<ObjptrT> contents_;
};

template <typename T,
	typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
struct NumArray final : public iArray
{
	NumArray (void) = default;

	NumArray (const std::vector<T>& contents) :
		contents_(contents.begin(), contents.end()) {}

	size_t class_code (void) const override
	{
		static const std::type_info& tp = typeid(NumArray<T>);
		return tp.hash_code();
	}

	std::string to_string (void) const override
	{
		return fmts::to_string(contents_.begin(), contents_.end());
	}

	bool equals (const iObject& other) const override
	{
		if (other.class_code() != this->class_code())
		{
			return false;
		}
		auto& ocontents = static_cast<const NumArray<T>*>(&other)->contents_;
		return std::equal(contents_.begin(), contents_.end(), ocontents.begin());
	}

	size_t size (void) const override
	{
		return contents_.size();
	}

	void foreach (std::function<void(ObjptrT&)> consume) override
	{
		for (T val : contents_)
		{
			ObjptrT obj = std::make_unique<Number<T>>(val);
			consume(obj);
		}
	}

	void foreach (std::function<void(const ObjptrT&)> consume) const override
	{
		for (T val : contents_)
		{
			consume(std::make_unique<Number<T>>(val));
		}
	}

	std::vector<T> contents_;
};

struct Maps final : public iObject
{
	size_t class_code (void) const override
	{
		static const std::type_info& tp = typeid(Maps);
		return tp.hash_code();
	}

	std::string to_string (void) const override
	{
		auto ks = keys();
		return fmts::to_string(ks.begin(), ks.end());
	}

	bool equals (const iObject& other) const override
	{
		if (other.class_code() != this->class_code())
		{
			return false;
		}
		auto& ocontents = static_cast<const Maps*>(&other)->contents_;
		if (ocontents.size() != contents_.size())
		{
			return false;
		}
		for (auto& cpair : contents_)
		{
			if (false == estd::has(ocontents, cpair.first) ||
				false == ocontents.at(cpair.first)->equals(*cpair.second))
			{
				return false;
			}
		}
		return true;
	}

	void accept (iMarshaler& marshaler) const override
	{
		marshaler.marshal(*this);
	}

	std::vector<std::string> keys (void) const
	{
		std::vector<std::string> out;
		out.reserve(contents_.size());
		for (auto& cpair : contents_)
		{
			out.push_back(cpair.first);
		}
		std::sort(out.begin(), out.end());
		return out;
	}

	std::unordered_map<std::string,ObjptrT> contents_;
};

}

#endif // MARSH_OBJS_HPP
