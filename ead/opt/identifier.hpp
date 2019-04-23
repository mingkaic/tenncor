#include "ade/itraveler.hpp"

#include "ead/constant.hpp"

#ifndef EAD_IDENTIFIER_HPP
#define EAD_IDENTIFIER_HPP

namespace ead
{

std::string pid (const void* addr)
{
	std::stringstream ss;
	ss << std::hex << (size_t) addr;
	return ss.str();
}

template <typename T>
struct Identifier final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (ids_.end() == ids_.find(leaf))
		{
			if (static_cast<iLeaf<T>*>(leaf)->is_const())
			{
				std::string data_str;
				auto ldata = (T*) leaf->data();
				auto shape = leaf->shape();
				size_t n = shape.n_elems();
				if (std::all_of(ldata + 1, ldata + n,
					[&](const T& e) { return e == ldata[0]; }))
				{
					// is a scalar
					std::stringstream ss;
					ss << "scalar_" << std::hex << ldata[0];
					data_str = ss.str();
				}
				else
				{
					// todo: make encoding better so we don't lose precision
					data_str = "array_" + fmts::join("", ldata, ldata + n);
				}
				ids_.emplace(leaf,
					fmts::join("", shape.begin(), shape.end()) + data_str);
			}
			else
			{
				ids_.emplace(leaf, pid(leaf));
			}
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (ids_.end() == ids_.find(func))
		{
			auto& children = func->get_children();
			std::vector<std::string> arg_ids;
			arg_ids.reserve(children.size());
			for (auto& child : children)
			{
				auto tens = child.get_tensor();
				tens->accept(*this);
				std::string id = ids_[tens.get()];
				auto shaper = child.get_shaper();
				auto coorder = child.get_coorder();
				if (nullptr != shaper)
				{
					// todo: make shaper encoding shorter
					id += shaper->to_string();
				}
				id += "_";
				if (nullptr != coorder)
				{
					// todo: make coorder encoding shorter
					id += coorder->to_string();
				}
				arg_ids.push_back(id);
			}
			std::string id;
			std::string arg_imprint = func->get_opcode().name_ + ":" +
				fmts::join(",", arg_ids.begin(), arg_ids.end());
			auto it = imprints_.find(arg_imprint);
			if (imprints_.end() == it)
			{
				id = pid(func);
				imprints_.emplace(arg_imprint, id);
			}
			else
			{
				id = it->second;
			}
			ids_.emplace(func, id);
		}
	}

	// 1-to-1 map longer function imprint to a shorter id
	std::unordered_map<std::string,std::string> imprints_;

	// n-to-1 map tensor node to id
	// duplicate ids indicate tensor node is replaceable
	std::unordered_map<ade::iTensor*,std::string> ids_;
};

}

#endif // EAD_IDENTIFIER_HPP
