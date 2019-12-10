#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "marsh/imarshal.hpp"

#ifndef MARSH_JSON_MARSH_HPP
#define MARSH_JSON_MARSH_HPP

namespace marsh
{

struct JsonMarshaler final : public iMarshaler
{
	void marshal (const iNumber& num) override
	{
		jreps_[&num].put("", num.to_string());
	}

	void marshal (const iArray& arr) override
	{
		auto& jarr = jreps_[&arr];
		arr.foreach(
			[&](const ObjptrT& entry)
			{
				entry->accept(*this);
				jarr.push_back(std::make_pair("", jreps_[entry.get()]));
			});
	}

	void marshal (const Maps& mm) override
	{
		auto& json_map = jreps_[&mm];
		auto keys = mm.ls_attrs();
		for (auto& key : keys)
		{
			auto& val = mm.contents_.at(key);
			val->accept(*this);
			json_map.add_child(key, jreps_[val.get()]);
		}
	}

	void parse (std::ostream& out, const iObject& obj, bool pretty)
	{
		if (false == estd::has(jreps_, &obj))
		{
			obj.accept(*this);
		}
		auto& rep = jreps_[&obj];
		if (rep.empty())
		{
			out << obj.to_string();
		}
		else
		{
			boost::property_tree::write_json(out, rep, pretty);
		}
	}

	std::string parse (const iObject& obj, bool pretty = true)
	{
		std::ostringstream buf;
		this->parse(buf, obj, pretty);
		return buf.str();
	}

private:
	std::unordered_map<const iObject*,boost::property_tree::ptree> jreps_;
};

}

#endif // MARSH_JSON_MARSH_HPP
