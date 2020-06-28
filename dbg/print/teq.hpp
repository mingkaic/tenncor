///
/// teq.hpp
/// dbg
///
/// Purpose:
/// Draw an equation graph as an ascii tree
///

#include "teq/teq.hpp"

#include "dbg/print/tree.hpp"

#ifndef DBG_TEQ_HPP
#define DBG_TEQ_HPP

/// Map tensor to label
using LabelsMapT = teq::TensMapT<std::string>;

const std::string dummy_label = "DEPENDENCIES";

struct DummyFunctor final : public teq::iFunctor
{
	DummyFunctor (teq::TensptrsT dependencies) :
		dependencies_(dependencies) {}

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return teq::Shape();
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return dummy_label;
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return teq::Opcode{dummy_label, 0};
	}

	/// Implementation of iFunctor
	teq::TensptrsT get_args (void) const override
	{
		return dependencies_;
	}

	/// Implementation of iFunctor
	teq::TensptrsT get_dependencies (void) const override
	{
		return dependencies_;
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override {}

	/// Implementation of iTensor
	teq::iDeviceRef& device (void) override
	{
		teq::fatal("shouldn't be called");
	}

	/// Implementation of iTensor
	const teq::iDeviceRef& device (void) const override
	{
		teq::fatal("shouldn't be called");
	}

	/// Implementation of iTensor
	const teq::iMetadata& get_meta (void) const override
	{
		teq::fatal("shouldn't be called");
	}

	std::vector<std::string> ls_attrs (void) const override
	{
		return {};
	}

	const marsh::iObject* get_attr (const std::string& attr_key) const override
	{
		return nullptr;
	}

	marsh::iObject* get_attr (const std::string& attr_key) override
	{
		return nullptr;
	}

	void add_attr (const std::string& attr_key, marsh::ObjptrT&& attr_val) override {}

	void rm_attr (const std::string& attr_key) override {}

private:
	teq::iTensor* clone_impl (void) const override
	{
		teq::fatal("shouldn't be called");
	}

	teq::TensptrsT dependencies_;
};

/// Use PrettyTree to render teq::TensptrT graph as an ascii art
struct PrettyEquation final
{
	PrettyEquation (void) : drawer_(
		[this](teq::iTensor*& root, size_t depth) -> teq::TensT
		{
			if (auto f = dynamic_cast<teq::iFunctor*>(root))
			{
				auto children = f->get_args();
				std::vector<teq::iTensor*> tens;
				tens.reserve(children.size());
				std::transform(children.begin(), children.end(),
					std::back_inserter(tens),
					[](teq::TensptrT child)
					{
						return child.get();
					});
				auto deps = f->get_dependencies();
				if (deps.size() > children.size())
				{
					auto dummy = std::make_shared<DummyFunctor>(
						teq::TensptrsT(deps.begin() + children.size(), deps.end()));
					tens.push_back(dummy.get());
					this->dummies_.push_back(dummy);
				}
				return tens;
			}
			return {};
		},
		[this](std::ostream& out, teq::iTensor*& root)
		{
			if (root)
			{
				auto it = this->labels_.find(root);
				if (this->labels_.end() != it)
				{
					out << it->second << "=";
				}
				if (auto var = dynamic_cast<teq::iLeaf*>(root))
				{
					out << teq::get_usage_name(var->get_usage()) << ":";
				}
				out << root->to_string();
				if (showshape_)
				{
					out << root->shape().to_string();
				}
				if (auto fnc = dynamic_cast<teq::iFunctor*>(root))
				{
					if (lsattrs_)
					{
						auto attrs = fnc->ls_attrs();
						out << ":attrkeys=" << fmts::to_string(attrs.begin(), attrs.end());
					}
					if (auto attr = fnc->get_attr(showattr_))
					{
						out << ":attr=" << attr->to_string();
					}
				}
			}
		}) {}

	/// Stream equation of ptr to out
	void print (std::ostream& out, const teq::TensptrT& ptr)
	{
		drawer_.print(out, ptr.get());
		dummies_.clear();
	}

	/// Stream equation of raw ptr to out
	void print (std::ostream& out, teq::iTensor* ptr)
	{
		drawer_.print(out, ptr);
		dummies_.clear();
	}

	/// For every label associated with a tensor, show LABEL=value in the tree
	LabelsMapT labels_;

	/// Print every tensor's shape if true, otherwise don't
	bool showshape_ = false;

	bool lsattrs_ = false;

	std::string showattr_;

private:
	/// Actual ascii renderer
	PrettyTree<teq::iTensor*> drawer_;

	teq::TensptrsT dummies_;
};

#endif // DBG_TEQ_HPP
