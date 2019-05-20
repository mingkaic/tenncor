///
/// ade.hpp
/// dbg
///
/// Purpose:
/// Draw an equation graph as an ascii tree
///

#include <unordered_map>

#include "ade/functor.hpp"

#include "dbg/stream/tree.hpp"

#ifndef DBG_ADE_HPP
#define DBG_ADE_HPP

using LabelsMapT = std::unordered_map<ade::iTensor*,std::string>;

/// Use PrettyTree to render ade::TensptrT graph as an ascii art
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
				[](const ade::FuncArg& child)
				{
					return child.get_tensor().get();
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
				out << root->to_string();
				if (showshape_ && nullptr != dynamic_cast<ade::iFunctor*>(root))
				{
					out << root->shape().to_string();
				}
			}
		}) {}

	/// Stream equation of ptr to out
	void print (std::ostream& out, ade::TensptrT& ptr)
	{
		drawer_.print(out, ptr.get());
	}

	void print (std::ostream& out, ade::iTensor* ptr)
	{
		drawer_.print(out, ptr);
	}

	/// For every label associated with a tensor, show LABEL=value in the tree
	LabelsMapT labels_;

	bool showshape_ = false;

private:
	/// Actual ascii renderer
	PrettyTree<ade::iTensor*> drawer_;
};

#endif // DBG_ADE_HPP
