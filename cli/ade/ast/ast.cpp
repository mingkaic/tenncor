#include <cstdlib>
#include <cstring>
#include <cstdio>

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "ade/functor.hpp"

#include "dbg/ade.hpp"

#include "cli/ade/ast/ast.h"

struct ShapeHolder
{
	uint8_t d_[ade::rank_cap];
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
		if (dest->rank_ < ade::rank_cap)
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
static std::unordered_map<ade::iTensor*,std::string> varname;

struct ASTNode* empty_node (void)
{
	return new ASTNode{ade::Tensor::get(ade::Shape())};
}

struct ASTNode* to_node (const struct ShapeHolder* shape)
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

void save_ast (char key[32], struct ASTNode* value)
{
	if (value)
	{
		std::string skey(key);
		auto it = asts.find(skey);
		if (it != asts.end())
		{
			varname.erase(it->second.get());
			asts.erase(it);
		}
		asts.emplace(skey, value->ptr_);
		varname.emplace(value->ptr_.get(), skey);
	}
}

struct ASTNode* load_ast (char key[32])
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


static std::unordered_set<std::string> modes;

void use_mode (char mode[32])
{
	modes.emplace(std::string(mode));
}

void show_shape (struct ASTNode* node)
{
	if (node)
	{
		std::string prefix;
		if (modes.end() != modes.find("showvar"))
		{
			auto it = varname.find(node->ptr_.get());
			if (varname.end() != it)
			{
				prefix = it->second + "=";
			}
		}
		std::string str = node->ptr_->shape().to_string();
		std::replace(str.begin(), str.end(), '\\', ',');
		std::cout << prefix << str << std::endl;
	}
	else
	{
		std::cout << "<nil>" << std::endl;
	}
}

void show_eq (struct ASTNode* node)
{
	if (node)
	{
		PrettyEquation artist;
		if (modes.end() != modes.find("showvar"))
		{
			artist.labels_ = varname;
		}
		artist.print(std::cout, node->ptr_);
	}
	else
	{
		std::cout << "<nil>" << std::endl;
	}
}
