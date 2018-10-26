///
/// ade.hpp
/// dbg
///
/// Purpose:
/// Draw an equation graph as an ascii tree
///

#include <unordered_map>

#include "ade/functor.hpp"

#include "dbg/tree.hpp"

#ifndef DBG_ADE_HPP
#define DBG_ADE_HPP

/// Use PrettyTree to render ade::Tensorptr graph as an ascii art
struct PrettyEquation final
{
	PrettyEquation (void) : drawer_(
		[](ade::iTensor*& root) -> std::vector<ade::iTensor*>
		{
			if (ade::iFunctor* f = dynamic_cast<ade::iFunctor*>(root))
			{
				auto& children = f->get_children();
				std::vector<ade::iTensor*> tens(children.size());
				std::transform(children.begin(), children.end(), tens.begin(),
				[](const std::pair<ade::CoordPtrT,ade::Tensorptr>& child)
				{
					return child.second.get();
				});
				return tens;
			}
			return {};
		},
		[this](std::ostream& out, ade::iTensor*& root)
		{
			if (root)
			{
				auto it = this->labels_.find(root);
				if (this->labels_.end() != it)
				{
					out << it->second << "=";
				}
				if (root == ade::Tensor::SYMBOLIC_ONE.get())
				{
					out << "[1]<SYMBOLIC_1>";
				}
				else if (root == ade::Tensor::SYMBOLIC_ZERO.get())
				{
					out << "[1]<SYMBOLIC_0>";
				}
				else
				{
					out << root->to_string();
				}
				if (showshape_ && nullptr != dynamic_cast<ade::iFunctor*>(root))
				{
					out << root->shape().to_string();
				}
			}
		}) {}

	/// Stream equation of ptr to out
	void print (std::ostream& out, ade::Tensorptr& ptr)
	{
		drawer_.print(out, ptr.get());
	}

	/// For every label associated with a tensor, show LABEL=value in the tree
	std::unordered_map<ade::iTensor*,std::string> labels_;

	bool showshape_ = false;

private:
	/// Actual ascii renderer
	PrettyTree<ade::iTensor*> drawer_;
};

#endif // DBG_ADE_HPP
