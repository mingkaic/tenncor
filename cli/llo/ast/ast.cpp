#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "llo/api.hpp"

#include "cli/util/ascii_tree.hpp"
#include "cli/util/ascii_tensor.hpp"

#include "cli/llo/ast/ast.h"

struct DataHolder
{
	virtual ~DataHolder (void) = default;

	virtual void shape_fill (std::vector<ade::DimT>& slist) const = 0;

	virtual void data_fill (double* data, const ade::Shape& shape) const = 0;

	virtual uint8_t depth (void) const = 0;
};

struct HolderHolder : public DataHolder
{
	HolderHolder (uint8_t depth, std::vector<DataHolder*> children) :
		depth_(depth), children_(children) {}

	void shape_fill (std::vector<ade::DimT>& slist) const override
	{
		slist[depth_] = std::max(slist[depth_], (uint8_t) children_.size());
		for (DataHolder* dh : children_)
		{
			dh->shape_fill(slist);
		}
	}

	void data_fill (double* data, const ade::Shape& shape) const override
	{
		auto it = shape.begin();
		ade::NElemT next_level = std::accumulate(it, it + depth_, 1,
			std::multiplies<ade::NElemT>());
		for (uint8_t i = 0, n = std::min(
				shape.at(depth_), (uint8_t) children_.size());
			i < n; ++i)
		{
			children_[i]->data_fill(data + i * next_level, shape);
		}
	}

	uint8_t depth (void) const override
	{
		return depth_;
	}

	uint8_t depth_;
	std::vector<DataHolder*> children_;
};

struct DoublesHolder : public DataHolder
{
	DoublesHolder (std::vector<double> data) : data_(data) {}

	void shape_fill (std::vector<ade::DimT>& slist) const override
	{
		slist[0] = std::max(slist[0], (uint8_t) data_.size());
	}

	void data_fill (double* data, const ade::Shape& shape) const override
	{
		std::memcpy(data, &data_[0], sizeof(double) * data_.size());
	}

	uint8_t depth (void) const override
	{
		return 0;
	}

	std::vector<double> data_;
};

struct DataHolder* make_data (struct DataHolder* e)
{
	return new HolderHolder(e->depth() + 1, {e});
}

struct DataHolder* make_data_d (double e)
{
	return new DoublesHolder({e});
}

void free_data (struct DataHolder* data)
{
	if (auto holder = dynamic_cast<HolderHolder*>(data))
	{
		for (DataHolder* dh : holder->children_)
		{
			free_data(dh);
		}
	}
	delete data;
}

void data_append (struct DataHolder* dest, struct DataHolder* e)
{
	assert(dest->depth() == e->depth() + 1);
	auto holder = static_cast<HolderHolder*>(dest);
	holder->children_.push_back(e);
}

void data_append_d (struct DataHolder* dest, double e)
{
	auto holder = static_cast<DoublesHolder*>(dest);
	holder->data_.push_back(e);
}


struct ASTNode
{
	ade::Tensorptr ptr_;
};

static std::unordered_map<std::string,ade::Tensorptr> asts;
static std::unordered_map<ade::iTensor*,std::string> varname;
static std::vector<uint16_t> data_limits = {15, 5, 5, 5, 1, 1, 1, 1};

struct ASTNode* empty_node (void)
{
	return new ASTNode{ade::Tensor::get(ade::Shape())};
}

struct ASTNode* to_node (const struct DataHolder* data)
{
	struct ASTNode* out = NULL;
	if (data)
	{
		std::vector<ade::DimT> slist(data->depth() + 1, 0);
		data->shape_fill(slist);
		ade::Shape shape = ade::Shape(slist);
		std::vector<double> vec(shape.n_elems(), 0);
		data->data_fill(&vec[0], shape);
		out = new ASTNode{llo::Source<double>::get(shape, vec)};
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
	else
	{
		std::cerr << "Failed to assign nil value to variable " << key << std::endl;
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
		out = new ASTNode{
			ade::runtime_functor((ade::OPCODE) code, {child->ptr_})};
	}
	else
	{
		std::cerr << "Failed to perform <" << ade::opname((ade::OPCODE) code)
			<< "> with input " << child << std::endl;
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
	else
	{
		std::cerr << "Failed to run <" << ade::opname((ade::OPCODE) code)
			<< "> with inputs " << child << ", " << child1 << std::endl;
	}
	return out;
}

struct ASTNode* unary_dim (struct ASTNode* child, double dim,
	enum FUNCODE code)
{
	struct ASTNode* out = NULL;
	if (child)
	{
		switch (code)
		{
			case FLIP:
				out  = new ASTNode{llo::flip(child->ptr_, dim)};
			break;
			case NDIMS:
				out  = new ASTNode{llo::n_dims(child->ptr_, dim)};
			break;
			default:
				std::cerr << "Failed to run <" << ade::opname((ade::OPCODE) code)
					<< "> as function with dimension " << dim << std::endl;
		}
	}
	else
	{
		std::cerr << "Failed to run dimensioned <" << ade::opname((ade::OPCODE) code)
			<< "> with input " << child << std::endl;
	}
	return out;
}

struct ASTNode* shapeop (struct ASTNode* child,
	struct DataHolder* shape, enum FUNCODE code)
{
	auto dholder = dynamic_cast<DoublesHolder*>(shape);
	struct ASTNode* out = NULL;
	if (child && dholder)
	{
		std::vector<double>& data = dholder->data_;
		switch (code)
		{
			case PERMUTE:
			{
				std::vector<uint8_t> perm(data.begin(), data.end());
				out = new ASTNode{ade::Functor<ade::PERMUTE,
					std::vector<uint8_t>>::get({child->ptr_}, perm)};
			}
			break;
			case EXTEND:
			{
				std::vector<ade::DimT> ext(data.begin(), data.end());
				out = new ASTNode{ade::Functor<ade::EXTEND,
					std::vector<ade::DimT>>::get({child->ptr_}, ext)};
			}
			break;
			case RESHAPE:
			{
				std::vector<ade::DimT> slist(data.begin(), data.end());
				out = new ASTNode{ade::Functor<ade::RESHAPE,
					std::vector<ade::DimT>>::get({child->ptr_}, slist)};
			}
			break;
			default:
				std::cerr << "Failed to run <" << ade::opname((ade::OPCODE) code)
					<< "> as function with shape" << std::endl;
		}
	}
	else
	{
		std::cerr << "Failed to run shaped <" << ade::opname((ade::OPCODE) code)
			<< "> with inputs " << child << ", " << dholder << std::endl;
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
	else
	{
		std::cerr << "Failed to run gradient with root=" << node << ", wrt="
			<< wrt << std::endl;
	}
	return out;
}

static std::unordered_set<std::string> modes;

void use_mode (char mode[32])
{
	modes.emplace(std::string(mode));
}

void print_datalimit (void)
{
	auto it = data_limits.begin();
	std::cout << (int) *it++;
	for (auto et = data_limits.end(); et != it; ++it)
	{
		std::cout << "," << (int) *it;
	}
	std::cout << std::endl;
}

void set_datalimit (struct DataHolder* limit)
{
	if (DoublesHolder* dh = dynamic_cast<DoublesHolder*>(limit))
	{
		auto it = dh->data_.begin();
		size_t nlimits = std::min((size_t) 8, dh->data_.size());
		data_limits = std::vector<uint16_t>(it, it + nlimits);
		if (nlimits < 8)
		{
			data_limits.insert(data_limits.end(), 8 - nlimits, 1);
		}
	}
	else
	{
		std::cerr << "Failed to set non-vector data limit" << std::endl;
	}
}

void show_data (struct ASTNode* node)
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
		llo::GenericData data = llo::evaluate(llo::DOUBLE, node->ptr_.get());
		std::vector<uint8_t> slist;
		if (data.shape_.n_rank() == 0)
		{
			slist = {1};
		}
		else
		{
			slist = data.shape_.as_list();
		}
		PrettyTensor<double> artist(slist, data_limits);
		std::cout << prefix;
		artist.print(std::cout, (double*) data.data_.get());
		std::cout << std::endl;
	}
	else
	{
		std::cout << "<nil>" << std::endl;
	}
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
				prefix = it->second + "_shape=";
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
		bool showvar = modes.end() != modes.find("showvar");
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
			[showvar](std::ostream& out, ade::iTensor*& root)
			{
				if (showvar)
				{
					auto it = varname.find(root);
					if (varname.end() != it)
					{
						out << it->second << "=";
					}
				}
				if (root)
				{
					if (root == ade::Tensor::SYMBOLIC_ONE.get())
					{
						out << 1;
					}
					else if (root == ade::Tensor::SYMBOLIC_ZERO.get())
					{
						out << 0;
					}
					else
					{
						out << root->to_string();
					}
				}
			});

		artist.print(std::cout, root);
	}
	else
	{
		std::cout << "<nil>" << std::endl;
	}
}
