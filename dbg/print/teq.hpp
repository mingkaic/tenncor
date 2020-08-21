///
/// teq.hpp
/// dbg
///
/// Purpose:
/// Draw an equation graph as an ascii tree
///

#include "internal/teq/teq.hpp"

#include "internal/teq/mock/functor.hpp"

#include "dbg/print/tree.hpp"

#ifndef DBG_TEQ_HPP
#define DBG_TEQ_HPP

/// Map tensor to label
using LabelsMapT = teq::TensMapT<std::string>;

const std::string dummy_label = "DEPENDENCIES";

/// Use PrettyTree to render teq::TensptrT graph as an ascii art
struct PrettyEquation final
{
	PrettyEquation (void) : renderer_(
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
					auto dummy = std::make_shared<MockFunctor>(
						teq::TensptrsT(deps.begin() + children.size(), deps.end()),
						teq::Opcode{dummy_label, 0});
					tens.push_back(dummy.get());
					this->dummies_.push_back(dummy);
				}
				return tens;
			}
			return {};
		},
		[this](std::ostream& out, teq::iTensor*& root, const std::string& prefix)
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
		renderer_.print(out, ptr.get());
		dummies_.clear();
	}

	/// Stream equation of raw ptr to out
	void print (std::ostream& out, teq::iTensor* ptr)
	{
		renderer_.print(out, ptr);
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
	PrettyTree<teq::iTensor*> renderer_;

	teq::TensptrsT dummies_;
};

#endif // DBG_TEQ_HPP
