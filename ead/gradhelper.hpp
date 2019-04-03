#include "ead/generated/api.hpp"

#include "ead/constant.hpp"
#include "ead/variable.hpp"
#include "ead/functor.hpp"

#ifndef EAD_GRADHELPER_HPP
#define EAD_GRADHELPER_HPP

namespace ead
{

template <typename T>
NodeptrT<T> to_node (ade::TensptrT tens)
{
	NodeptrT<T> out;
	if (auto func = std::dynamic_pointer_cast<Functor<T>>(tens))
	{
		out = std::make_shared<FuncNode<T>>(func);
	}
	else if (auto cst = std::dynamic_pointer_cast<Constant<T>>(tens))
	{
		out = std::make_shared<ConstantNode<T>>(cst);
	}
	else if (auto var = std::dynamic_pointer_cast<Variable<T>>(tens))
	{
		out = std::make_shared<VariableNode<T>>(var);
	}
	else
	{
		logs::fatalf("unknown tensor type with %s dtype",
			age::name_type(age::get_type<T>()).c_str());
	}
	return out;
}

template <typename T>
NodeptrT<T> reduce_sum_grad (ade::iFunctor* fwd,
	NodeptrT<T> bwd, ade::TensT args, size_t idx)
{
	const ade::Shape& shape = args[0]->shape();
	// assert shape == bwd->get_tensor()->shape()
	const auto& child = fwd->get_children()[0];
	ade::CoordptrT revshaper(child.get_shaper()->reverse());
	CoordptrT revcoord;
	{
		auto coorder = child.get_coorder();
		assert(nullptr != coorder);
		ade::CoordT dims;
		coorder->forward(dims.begin(), dims.begin());
		ade::CoordT bcast;
		std::fill(bcast.begin(), bcast.end(), 1);
		for (uint8_t d : dims)
		{
			if (d < ade::rank_cap)
			{
				bcast[d] = shape.at(d);
			}
		}
		revcoord = std::make_shared<CoordMap>(EXTEND, bcast, false);
	}
	return make_functor<T>(ade::Opcode{"EXTEND",age::EXTEND}, {
		FuncArg<T>(bwd, revshaper, revcoord)
	});
}

template <typename T>
NodeptrT<T> reduce_prod_grad (ade::iFunctor* fwd,
	NodeptrT<T> bwd, ade::TensT args, size_t idx)
{
	const auto& child = fwd->get_children()[0];
	NodeptrT<T> childnode = to_node<T>(child.get_tensor());
	NodeptrT<T> fwd_cpy = make_functor<T>(fwd->get_opcode(),
		{FuncArg<T>(childnode, child.get_shaper(),
			std::static_pointer_cast<CoordMap>(child.get_coorder()))});
	NodeptrT<T> rev_fwd = reduce_sum_grad(fwd, fwd_cpy, args, idx);
	return age::mul(age::div(rev_fwd, childnode),
		reduce_sum_grad(fwd, bwd, args, idx));
}

template <typename T>
NodeptrT<T> reduce_comp_grad (ade::iFunctor* fwd,
	NodeptrT<T> bwd, ade::TensT args, size_t idx)
{
	const auto& child = fwd->get_children()[0];
	NodeptrT<T> childnode = to_node<T>(child.get_tensor());
	NodeptrT<T> fwd_cpy = make_functor<T>(fwd->get_opcode(),
		{FuncArg<T>(childnode, child.get_shaper(),
			std::static_pointer_cast<CoordMap>(child.get_coorder()))});
	NodeptrT<T> rev_fwd = reduce_sum_grad(fwd, fwd_cpy, args, idx);
	return age::mul(age::eq(rev_fwd, childnode),
		reduce_sum_grad(fwd, bwd, args, idx));
}

template <typename T>
NodeptrT<T> permute_grad (ade::iFunctor* fwd,
	NodeptrT<T> bwd, ade::TensT args, size_t idx)
{
	const auto& child = fwd->get_children()[0];
	ade::CoordptrT revshaper(child.get_shaper()->reverse());
	CoordptrT revcoord;
	{
		auto coorder = child.get_coorder();
		assert(nullptr != coorder);
		ade::CoordT dims;
		coorder->forward(dims.begin(), dims.begin());

		ade::CoordT order;
		for (uint8_t i = 0; i < ade::rank_cap; ++i)
		{
			order[dims[i]] = i;
		}
		revcoord = std::make_shared<CoordMap>(PERMUTE, order, true);
	}
	return make_functor<T>(ade::Opcode{"PERMUTE",age::PERMUTE},{
		FuncArg<T>(bwd, revshaper, revcoord)
	});
}

template <typename T>
NodeptrT<T> extend_grad (ade::iFunctor* fwd,
	NodeptrT<T> bwd, ade::TensT args, size_t idx)
{
	const auto& child = fwd->get_children()[0];
	ade::CoordptrT revshaper(child.get_shaper()->reverse());
	CoordptrT revcoord;
	{
		auto coorder = child.get_coorder();
		assert(nullptr != coorder);
		ade::CoordT dims;
		coorder->forward(dims.begin(), dims.begin());
		std::vector<uint8_t> red_dims;
		for (uint8_t i = 0; i < ade::rank_cap; ++i)
		{
			if (dims[i] > 1)
			{
				red_dims.push_back(i);
			}
		}
		revcoord = reduce(red_dims);
	}
	return make_functor<T>(ade::Opcode{"REDUCE_SUM",age::REDUCE_SUM},{
		FuncArg<T>(bwd, revshaper, revcoord)
	});
}

template <typename T>
NodeptrT<T> matmul_grad (ade::iFunctor* fwd,
	NodeptrT<T> bwd, ade::TensT args, size_t idx)
{
	const auto& children = fwd->get_children();
	ade::TensptrT a = children[0].get_tensor();
	ade::TensptrT b = children[1].get_tensor();
	NodeptrT<T> lhs = to_node<T>(a);
	NodeptrT<T> rhs = to_node<T>(b);

	NodeptrT<T> ext_a = age::permute(
		age::extend(lhs, 2, {b->shape().at(0)}), {2,1,0});
	NodeptrT<T> ext_b = age::permute(
		age::extend(rhs, 2, {a->shape().at(1)}), {0,2,1});

	NodeptrT<T> ext;
	std::vector<uint8_t> perm;
	if (0 == idx)
	{
		ext = ext_a;
		perm = {2, 1, 0};
	}
	else
	{
		ext = ext_b;
		perm = {0, 2, 1};
	}

	NodeptrT<T> ext_bwd = age::extend(bwd, 2, {a->shape().at(0)});

	return age::reduce_sum(
		age::permute(
			age::mul(
				age::div(age::mul(ext_a, ext_b), ext),
				ext_bwd
			), perm), 2);
}

}

#endif // EAD_GRADHELPER_HPP
