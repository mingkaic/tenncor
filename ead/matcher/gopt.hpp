#include "ade/ade.hpp"

#include "ead/matcher/transform.hpp"

#ifndef OPT_GOPT_HPP
#define OPT_GOPT_HPP

namespace opt
{

struct GraphOpt final : ade::iTraveler
{
	GraphOpt (TransformsT transforms) : transforms_(transforms) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		std::string leaf_id = encode_tens(leaf);
		if (token_roots_.end() == token_roots_.find(leaf_id))
		{
			token_roots_.emplace(leaf_id, std::make_shared<TokenNode>(leaf));
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		std::string func_id = encode_tens(func);
		if (token_roots_.end() == token_roots_.find(func_id))
		{
			auto func_root = std::make_shared<TokenNode>(func);
			assert(func_id == func_root->tens_id_);

			auto children = func->get_children();
			for (auto child : children)
			{
				auto tens = child.get_tensor();
				auto shaper = child.get_shaper();
				auto coorder = child.get_coorder();
				tens->accept(*this);

				auto child_root = token_roots_[encode_tens(tens.get())];
				{
					auto tmp = child_root->clone(coord_map_,
						shaper, coorder);
					if (nullptr != tmp)
					{
						child_root = TokenptrT(tmp);
					}
				}
				func_root->children_.push_back(child_root);
				tens_map_.emplace(child_root->tens_id_, tens);
				if (child_root->changed_)
				{
					func_root->changed_ = true;
				}
			}
			// simplify and save
			for (Transform& transform : transforms_)
			{
				if (transform.simplify(func_root, token_roots_))
				{
					break;
				}
			}
			token_roots_.emplace(func_id, func_root);
		}
	}

	template <typename T>
	ead::NodesT<T> apply_optimization (ead::NodesT<T> roots)
	{
		// convert serials for each root to graphs, then reuse
		ead::NodesT<T> outs;
		for (ead::NodeptrT<T> root : roots) // todo: rename root to avoid mistakening for roots
		{
			auto it = token_roots_.find(
				encode_tens(root->get_tensor().get()));
			if (token_roots_.end() == it)
			{
				logs::fatal("cannot optimize non-serialized root");
			}
			outs.push_back(it->second->template decode<T>(
				root->shape(), tens_map_, coord_map_));
		}
		return outs;
	}

	TransformsT transforms_;

	IdTokenMapT token_roots_;

	IdTensMapT tens_map_;

	IdCoordMapT coord_map_;
};

}

#endif // OPT_GOPT_HPP
