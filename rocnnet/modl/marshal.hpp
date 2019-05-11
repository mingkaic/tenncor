#include <memory>

#include "pbm/save.hpp"
#include "pbm/load.hpp"

#include "ead/gradhelper.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifndef MODL_MARSHAL_HPP
#define MODL_MARSHAL_HPP

namespace modl
{

struct iMarshaler
{
	virtual ~iMarshaler (void) = default;

	iMarshaler* clone (void) const
	{
		return clone_impl();
	}

	virtual std::string get_label (void) const = 0;

	virtual pbm::PathedMapT list_bases (void) const = 0;

	/// Return all variables labelled specific to this model
	virtual pbm::PathedTens* get_variables (void) const = 0;

	/// Set all the variables associated with this label in storage
	virtual void set_variables (pbm::PathedTens* storage) = 0;

protected:
	virtual iMarshaler* clone_impl (void) const = 0;
};

using MarsptrT = std::shared_ptr<iMarshaler>;

using MarsarrT = std::vector<MarsptrT>;

struct iMarshalSet : public iMarshaler
{
	iMarshalSet (std::string label) : label_(label) {}

	iMarshalSet (const iMarshalSet& other) = default;

	iMarshalSet& operator = (const iMarshalSet& other) = default;

	iMarshalSet (iMarshalSet&& other) = default;

	iMarshalSet& operator = (iMarshalSet&& other) = default;

	virtual ~iMarshalSet (void) = default;

	std::string get_label (void) const override
	{
		return label_;
	}

	pbm::PathedMapT list_bases (void) const override
	{
		pbm::PathedMapT out;
		auto marshalers = get_subs();
		for (MarsptrT& marshal : marshalers)
		{
			auto temp = marshal->list_bases();
			for (auto& bpairs : temp)
			{
				bpairs.second.push_front(marshal->get_label());
			}
			out.insert(temp.begin(), temp.end());
		}
		return out;
	}

	pbm::PathedTens* get_variables (void) const override
	{
		auto out = new pbm::PathedTens();
		auto marshalers = get_subs();
		for (MarsptrT& marshal : marshalers)
		{
			std::string child_label = marshal->get_label();
			auto child = marshal->get_variables();
			auto it = out->children_.find(child_label);
			if (out->children_.end() != it)
			{
				it->second->join(child);
				delete child;
			}
			else
			{
				out->children_.emplace(child_label, child);
			}
		}
		return out;
	}

	void set_variables (pbm::PathedTens* storage) override
	{
		auto marshalers = get_subs();
		for (MarsptrT& marshal : marshalers)
		{
			std::string child_label = marshal->get_label();
			auto it = storage->children_.find(child_label);
			if (storage->children_.end() == it)
			{
				logs::warnf("label %s not found", child_label.c_str());
				continue;
			}
			marshal->set_variables(it->second);
		}
	}

	virtual MarsarrT get_subs (void) const = 0;

	std::string label_;
};

struct MarshalVar final : public iMarshaler
{
	MarshalVar (ead::VarptrT<PybindT> var) : var_(var) {}

	MarshalVar (const MarshalVar& other)
	{
		copy_helper(other);
	}

	MarshalVar& operator = (const MarshalVar& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	MarshalVar (MarshalVar&& other) = default;

	MarshalVar& operator = (MarshalVar&& other) = default;

	std::string get_label (void) const override
	{
		return var_->get_label();
	}

	pbm::PathedMapT list_bases (void) const override
	{
		auto var = var_->get_tensor();
		return pbm::PathedMapT{
			std::pair<ade::TensptrT,pbm::StringsT>{
				var,
				pbm::StringsT{var_->get_label()}
			}
		};
	}

	pbm::PathedTens* get_variables (void) const override
	{
		auto out = new pbm::PathedTens();
		out->tens_.emplace(var_->get_label(), var_->get_tensor());
		return out;
	}

	void set_variables (pbm::PathedTens* storage) override
	{
		auto it = storage->tens_.find(var_->get_label());
		if (storage->tens_.end() == it)
		{
			logs::warnf("variable %s not found", var_->get_tensor()->to_string().c_str());
		}
		else
		{
			var_->assign(ead::NodeConverters<PybindT>::to_node(it->second)->data(), it->second->shape());
		}
	}

	ead::VarptrT<PybindT> var_;

private:
	void copy_helper (const MarshalVar& other)
	{
		auto ov = static_cast<const ead::Variable<PybindT>*>(
			other.var_->get_tensor().get());
		var_ = std::make_shared<ead::VariableNode<PybindT>>(
			std::shared_ptr<ead::Variable<PybindT>>(
				ead::Variable<PybindT>::get(*ov)));
	}

	iMarshaler* clone_impl (void) const override
	{
		return new MarshalVar(*this);
	}
};

using MarVarsptrT = std::shared_ptr<MarshalVar>;

bool save (std::ostream& outs, ade::TensptrT source,
	iMarshaler* source_graph);

void load (std::istream& ins, iMarshaler* target);

}

#endif // MODL_MARSHAL_HPP
