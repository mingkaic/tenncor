///
/// teq.hpp
/// dbg
///
/// Purpose:
/// Draw an equation graph as an ascii tree
///

#include <unordered_map>

#include "teq/ileaf.hpp"
#include "teq/functor.hpp"

#include "dbg/stream/tree.hpp"

#ifndef DBG_TEQ_HPP
#define DBG_TEQ_HPP

using LabelsMapT = std::unordered_map<teq::iTensor*,std::string>;

/// Use PrettyTree to render teq::TensptrT graph as an ascii art
struct PrettyEquation final
{
	PrettyEquation (void) : drawer_(
		[](teq::iTensor*& root) -> std::vector<teq::iTensor*>
		{
			if (teq::iFunctor* f = dynamic_cast<teq::iFunctor*>(root))
			{
				auto& children = f->get_children();
				std::vector<teq::iTensor*> tens(children.size());
				std::transform(children.begin(), children.end(), tens.begin(),
				[](const teq::FuncArg& child)
				{
					return child.get_tensor().get();
				});
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
					out << (var->is_const() ? "constant:" : "variable:");
				}
				out << root->to_string();
				if (showshape_)
				{
					out << root->shape().to_string();
				}
			}
		}) {}

	/// Stream equation of ptr to out
	void print (std::ostream& out, const teq::TensptrT& ptr)
	{
		drawer_.print(out, ptr.get());
	}

	void print (std::ostream& out, teq::iTensor* ptr)
	{
		drawer_.print(out, ptr);
	}

	/// For every label associated with a tensor, show LABEL=value in the tree
	LabelsMapT labels_;

	bool showshape_ = false;

private:
	/// Actual ascii renderer
	PrettyTree<teq::iTensor*> drawer_;
};

#endif // DBG_TEQ_HPP
