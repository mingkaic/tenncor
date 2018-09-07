#include <cstdlib>
#include <cstring>
#include <cstdio>

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>

#include "ade/functor.hpp"

#include "ade_cli/ast/ascii.hpp"
#include "ade_cli/ast/calc_ast.h"

const uint8_t RANK_CAP = 8;
const uint8_t NBINS = 53;

struct ShapeHolder
{
	uint8_t d_[RANK_CAP];
	uint8_t rank_ = 0;
};

struct ShapeHolder* make (int dim)
{
	struct ShapeHolder* out = new ShapeHolder{};
	out->d_[0] = dim;
	out->rank_ = 1;
	return out;
}

void free_shape (struct ShapeHolder* shape)
{
	if (shape) delete shape;
}

void append (struct ShapeHolder* dest, int dim)
{
	if (dest)
	{
		if (dest->rank_ < RANK_CAP)
		{
			dest->d_[dest->rank_] = dim;
			++dest->rank_;
		}
		else
		{
			// warn
		}
	}
}

struct ASTNode
{
	ade::Tensorptr ptr_;
};

static std::unordered_map<std::string,ade::Tensorptr> asts;

struct ASTNode* emptyNode (void)
{
	return new ASTNode{ade::Tensor::get(ade::Shape())};
}

struct ASTNode* toNode (const struct ShapeHolder* shape)
{
	struct ASTNode* out = NULL;
	if (shape)
	{
		std::vector<ade::DimT> slist(shape->d_, shape->d_ + shape->rank_);
		out = new ASTNode{ade::Tensor::get(ade::Shape(slist))};
	}
	return out;
};

void free_ast (struct ASTNode* node)
{
	if (node) delete node;
}

void saveAST (char key[32], struct ASTNode* value)
{
	if (value)
	{
		std::string skey(key);
		auto it = asts.find(skey);
		if (it != asts.end())
		{
			asts.erase(it);
		}
		asts.emplace(skey, value->ptr_);
	}
}

struct ASTNode* loadAST (char key[32])
{
	std::string skey(key);
	auto it = asts.find(skey);
	struct ASTNode* out = NULL;
	if (it != asts.end())
	{
		out = new ASTNode{it->second};
	}
	return out;
}

struct ASTNode* unary (struct ASTNode* child, enum FUNCODE code)
{
	struct ASTNode* out = NULL;
	if (child)
	{
		out = new ASTNode{ade::runtime_functor((ade::OPCODE) code, {child->ptr_})};
	}
	return out;
}

struct ASTNode* binary (struct ASTNode* child,
	struct ASTNode* child1, enum FUNCODE code)
{
	struct ASTNode* out = NULL;
	if (child && child1)
	{
		out = new ASTNode{ade::runtime_functor(
			(ade::OPCODE) code, {child->ptr_, child1->ptr_})};
	}
	return out;
}

struct ASTNode* shapeop (struct ASTNode* child,
	struct ShapeHolder* shape, enum FUNCODE code)
{
	struct ASTNode* out = NULL;
	if (child)
	{
		switch (code)
		{
			case PERMUTE:
			{
				std::vector<uint8_t> perm(shape->d_,
					shape->d_ + shape->rank_);
				out = new ASTNode{ade::Functor<ade::PERMUTE,
					std::vector<uint8_t>>::get({child->ptr_}, perm)};
			}
			break;
			case EXTEND:
			{
				std::vector<ade::DimT> ext(shape->d_,
					shape->d_ + shape->rank_);
				out = new ASTNode{ade::Functor<ade::EXTEND,
					std::vector<ade::DimT>>::get({child->ptr_}, ext)};
			}
			break;
			case RESHAPE:
			{
				std::vector<ade::DimT> slist(shape->d_,
					shape->d_ + shape->rank_);
				out = new ASTNode{ade::Functor<ade::RESHAPE,
					std::vector<ade::DimT>>::get({child->ptr_}, slist)};
			}
			break;
			default:
				// is not shape op, treat like unary
				out = new ASTNode{ade::runtime_functor(
					(ade::OPCODE) code, {child->ptr_})};
		}
	}
	return out;
}

struct ASTNode* grad (struct ASTNode* node, struct ASTNode* wrt)
{
	struct ASTNode* out = NULL;
	if (node && wrt)
	{
		out = new ASTNode{node->ptr_->gradient(wrt->ptr_)};
	}
	return out;
}

void to_string (char out[64], struct ASTNode* node)
{
	if (node)
	{
		std::string str = node->ptr_->shape_.to_string();
		std::replace(str.begin(), str.end(), '\\', ',');
		std::memcpy(out, str.c_str(), std::min((size_t) 64, str.size() + 1));
	}
	else
	{
		std::memcpy(out, "<nil>", 6);
	}
}

void show_eq (struct ASTNode* node)
{
	ade::iTensor* root = node->ptr_.get();
	PrettyTree<ade::iTensor*> artist(
		[](ade::iTensor*& root) -> std::vector<ade::iTensor*>
		{
			if (ade::iFunctor* f = dynamic_cast<ade::iFunctor*>(root))
			{
				return f->get_refs();
			}
			return {};
		},
		[](std::ostream& out, ade::iTensor*& root)
		{
			if (root)
				out << root->to_string();
		});

	artist.print(std::cout, root);
}
