//
//  matmul.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/operations/operations.hpp"
#include "include/tensor/actors/tens_matmul.hpp"

#ifdef TENNCOR_MATMUL_HPP

namespace nnet
{

static inline tensorshape matmul_shaper (std::vector<tensorshape> shapes)
{
	tensorshape& t1s = shapes[0];
	tensorshape& t2s = shapes[1];

	std::vector<size_t> al = t1s.as_list();
	std::vector<size_t> bl = t2s.as_list();
	size_t rank1 = t1s.rank();
	size_t rank2 = t2s.rank();

	// account for vectors
	size_t ax = rank1 ? al[0] : 0;
	size_t ay = rank1> 1 ? al[1] : 1;
	size_t bx = rank2 ? bl[0] : 0;
	size_t by = rank2> 1 ? bl[1] : 1;

	// ensure the dimensions beyond 2d are equal
	size_t minend = std::min(rank1, rank2);
	std::vector<size_t> beyond2d;
	if (minend> 2)
	{
		auto ait = al.begin()+2;
		auto aet = al.begin()+minend;
		if (std::equal(ait, aet, bl.begin()+2))
		{
			beyond2d.insert(beyond2d.end(), ait, aet);
		}
		else
		{
			std::stringstream ss;
			ss << "attempting to matrix multiple shapes ";
			print_shape(t1s, ss);
			ss << " and ";
			print_shape(t2s, ss);
			throw std::logic_error(ss.str());
		}
		// check that remaining shape values are ones,
		// otherwise one shape is larger than the other
		auto it = rank1> rank2 ? al.begin() : bl.begin();
		auto et = rank1> rank2 ? al.end() : bl.end();
		if (!std::all_of(it + minend, et, [](size_t e) { return e == 1; }))
		{
			std::stringstream ss;
			ss << "attempting to matrix multiple different sized shapes ";
			print_shape(t1s, ss);
			ss << " and ";
			print_shape(t2s, ss);
			throw std::logic_error(ss.str());
		}
	}

	// get resulting shape
	std::vector<size_t> res_shape;
	if (ax == by)
	{
		res_shape = {bx, ay};
	}
	else
	{
		std::stringstream ss;
		ss << "matmul shapes ";
		print_shape(t1s, ss);
		ss << "and ";
		print_shape(t2s, ss);
		ss << "do not match";
		throw std::logic_error(ss.str());
	}
	res_shape.insert(res_shape.end(), beyond2d.begin(), beyond2d.end());
	return res_shape;
}

varptr matmul (const varptr a, const varptr b, bool transposeA, bool transposeB)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;

	inode* adata = a.get();
	inode* bdata = b.get();
	if (transposeA)
	{
		if (inode* parent = unary_parent_search(adata, "transpose_0_1"))
		{
			adata = parent;
		}
		else
		{
			adata = transpose(a, {0, 1});
		}
	}
	if (transposeB)
	{
		if (inode* parent = unary_parent_search(bdata, "transpose_0_1"))
		{
			bdata = parent;
		}
		else
		{
			bdata = transpose(b, {0, 1});
		}
	}

	std::string opname = "matmul";
	if (inode* parent = ordered_binary_parent_search(adata, bdata, opname))
	{
		return parent;
	}

	immutable* mmul = immutable::get(std::vector<inode*>{adata, bdata},
	matmul_shaper,
	new actor_func(
	CONN_ACTOR([](out_wrapper<void>& dest, std::vector<in_wrapper<void> >& srcs,
		tenncor::tensor_proto::tensor_t type) -> itens_actor*
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				return new tens_matmul<double>(dest, srcs);
			case tenncor::tensor_proto::SIGNED_T:
				return new tens_matmul<signed>(dest, srcs);
			default:
			break;
		}
		return nullptr;
	})),
	[](std::vector<std::pair<inode*,inode*>> args)
	{
		// todo: create alternative operation to eq (since eq prevents higher order derivatives)
		// desired behavior, create a matrix of 1 output in the same shape as the tforward operation
		// tforward shape may not be defined at this point,
		// so the shape instantiation must be performed during base_immutable::update
		nnet::varptr tforward = nnet::matmul(args[0].first, args[1].first);
		return nnet::eq(tforward, tforward);
	}, opname);

	std::unordered_set<ileaf*> temp = mmul->get_leaves();
	std::vector<variable*> leef;
	for (ileaf* ilef : temp)
	{
		if (variable* var = dynamic_cast<variable*>(ilef))
		{
			leef.push_back(var);
		}
	}

	mmul->set_jacobian_back(
	[transposeA, transposeB](inode* root, std::vector<inode*> args, std::vector<inode*> grads) -> inode*
	{
		varptr arga = args[0];
		varptr argb = args[1];
		varptr grada = grads[0];
		varptr gradb = grads[1];

		constant* aconst = dynamic_cast<constant*>(grada.get());
		constant* bconst = dynamic_cast<constant*>(gradb.get());
		if (aconst && *aconst == 0 && bconst && *bconst == 0)
		{
			return root;
		}
		varptr mA = matmul(root, argb, false, true);
		varptr mB = matmul(arga, root, true);
		if (transposeA)
		{
			mA = transpose(mA);
		}
		if (transposeB)
		{
			mB = transpose(mB);
		}
		return mA * grada + mB * gradb;
	}, leef);
	return mmul;
}

}

#endif