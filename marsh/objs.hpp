#include <functional>

#include "estd/estd.hpp"
#include "fmts/fmts.hpp"

#include "marsh/attrs.hpp"

#ifndef MARSH_OBJS_HPP
#define MARSH_OBJS_HPP

namespace marsh
{

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
	Number (void) : val_(0) {}

	Number (T val) : val_(val) {}

	Number<T>* clone (void) const
	{
		return static_cast<Number<T>*>(clone_impl());
	}

	size_t class_code (void) const override
	{
		static const std::type_info& tp = typeid(Number<T>);
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

private:
	iObject* clone_impl (void) const override
	{
		return new Number<T>(val_);
	}
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
	ObjArray* clone (void) const
	{
		return static_cast<ObjArray*>(clone_impl());
	}

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

private:
	iObject* clone_impl (void) const override
	{
		auto cpy = new ObjArray();
		for (auto& obj : contents_)
		{
			cpy->contents_.insert(cpy->contents_.end(),
				ObjptrT(obj->clone()));
		}
		return cpy;
	}
};

template <typename T,
	typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
struct NumArray final : public iArray
{
	NumArray (void) = default;

	NumArray (const std::vector<T>& contents) :
		contents_(contents) {}

	NumArray<T>* clone (void) const
	{
		return static_cast<NumArray<T>*>(clone_impl());
	}

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
		return contents_.size() == ocontents.size() &&
			std::equal(contents_.begin(), contents_.end(), ocontents.begin());
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

private:
	iObject* clone_impl (void) const override
	{
		return new NumArray<T>(contents_);
	}
};

struct Maps final : public iObject, public iAttributed
{
	Maps* clone (void) const
	{
		return static_cast<Maps*>(clone_impl());
	}

	size_t class_code (void) const override
	{
		static const std::type_info& tp = typeid(Maps);
		return tp.hash_code();
	}

	std::string to_string (void) const override
	{
		auto ks = ls_attrs();
		std::sort(ks.begin(), ks.end());
		std::vector<std::pair<std::string,std::string>> pairs;
		for (std::string key : ks)
		{
			pairs.push_back({key, contents_.at(key)->to_string()});
		}
		return fmts::to_string(pairs.begin(), pairs.end());
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

	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return estd::has(contents_, attr_name) ?
			contents_.at(attr_name).get() : nullptr;
	}

	std::vector<std::string> ls_attrs (void) const override
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

private:
	iObject* clone_impl (void) const override
	{
		auto cpy = new Maps();
		for (auto& cpair : contents_)
		{
			cpy->contents_.emplace(cpair.first,
				ObjptrT(cpair.second->clone()));
		}
		return cpy;
	}
};

void get_attrs (marsh::Maps& mvalues, const iAttributed& attributed);

}

#endif // MARSH_OBJS_HPP
