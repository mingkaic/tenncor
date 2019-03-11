#include <regex>
#include <unordered_set>

#include "ade/ifunctor.hpp"

#include "ead/generated/opmap.hpp"

#include "ead/ead.hpp"
#include "ead/coord.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"

#ifndef OPT_IREP_HPP
#define OPT_IREP_HPP

namespace opt
{

static const std::string tuple_delim = "->";

static const std::string line_delim = ",,";

static const int _OPT_VARTYPE = -1;

static const int _OPT_CONSTTYPE = -2;

static const int _OPT_SCALRTYPE = -3;

static const std::string varname = "variable";

static const std::string constname = "constant";

static const std::string scalrname = "scalar";

static const std::regex token_pattern(
	"(?:(\\w+)\\((\\d*(?:\\.\\d+)?)\\))(?:<(\\d+)>)?(?:\\[(\\d+)\\])?");

static const std::unordered_set<std::string> reserved_opnames = {
	varname, constname, scalrname };

struct TokenNode;

using TokenptrT = std::shared_ptr<TokenNode>;

using IdTokenMapT = std::unordered_map<std::string,TokenptrT>;

using IdTensMapT = std::unordered_map<std::string,ade::TensptrT>;

using IdCoordMapT = std::unordered_map<std::string,ade::CoordptrT>;

std::string encode_tens (ade::iTensor* tens);

std::string encode_coorder (ade::iCoordMap* coorder);

template <typename T>
void rep_leaf (int& type, std::string& str_id,
	ead::iLeaf<T>* leaf)
{
	if (leaf->is_const())
	{
		T* begin = (T*) leaf->data();
		T* end = begin + leaf->shape().n_elems();
		if (std::adjacent_find(begin, end, std::not_equal_to<T>()) == end)
		{
			type = _OPT_SCALRTYPE;
			str_id = fmts::to_string(*begin);
			return;
		}
		type = _OPT_CONSTTYPE;
	}
	else
	{
		type = _OPT_VARTYPE;
	}
	str_id = encode_tens(leaf);
}

// every token has the format '<type>(<id>)'
// TokenNode holds these information and child information
struct TokenNode final
{
	// todo: upon code generation, store each op's properties somewhere
	static size_t get_nchildren (age::_GENERATED_OPCODE opcode)
	{
		switch (opcode)
		{
			case age::ABS:
			case age::COS:
			case age::CUBE:
			case age::EXP:
			case age::EXTEND:
			case age::LOG:
			case age::NEG:
			case age::PERMUTE:
			case age::REDUCE_MAX:
			case age::REDUCE_MIN:
			case age::REDUCE_PROD:
			case age::REDUCE_SUM:
			case age::ROUND:
			case age::SIGMOID:
			case age::SIN:
			case age::SQRT:
			case age::SQUARE:
			case age::TAN:
			case age::TANH:
				return 1;

			case age::ADD:
			case age::CONV:
			case age::DIV:
			case age::EQ:
			case age::GT:
			case age::LT:
			case age::MATMUL:
			case age::MAX:
			case age::MIN:
			case age::MUL:
			case age::NEQ:
			case age::POW:
			case age::RAND_UNIF:
			case age::SUB:
			default:
				return 2;
		}
	}

#define _TYPED_REPLEAF(realtype)\
rep_leaf<realtype>(type_, tens_id_,\
static_cast<ead::iLeaf<realtype>*>(leaf));

	TokenNode (ade::iLeaf* leaf)
	{
		TYPE_LOOKUP(_TYPED_REPLEAF, leaf->type_code())
	}

#undef _TYPED_REPLEAF

	TokenNode (ade::iFunctor* func) :
		type_(func->get_opcode().code_),
		tens_id_(encode_tens(func))
	{
		nallowed_children_ = get_nchildren((age::_GENERATED_OPCODE) type_);
	}

	TokenNode* clone (IdCoordMapT& coord_map,
		const ade::CoordptrT& shaper, const ade::CoordptrT& coorder)
	{
		std::string shaper_id;
		std::string coorder_id;
		if (false == ade::is_identity(shaper.get()))
		{
			shaper_id = encode_coorder(shaper.get());
			coord_map.emplace(shaper_id, shaper);
		}
		if (false == ade::is_identity(coorder.get()))
		{
			coorder_id = encode_coorder(coorder.get());
			coord_map.emplace(coorder_id, coorder);
		}
		TokenNode* out = nullptr;
		if (shaper_id.size() > 0 || coorder_id.size())
		{
			out = new TokenNode(*this);
			out->shaper_id_ = shaper_id;
			out->coorder_id_ = coorder_id;
		}
		return out;
	}

	TokenNode (std::string token)
	{
		std::smatch sm;
		if (false == std::regex_match(token, sm, token_pattern))
		{
			logs::fatalf("cannot match token %s", token.c_str());
		}
		std::string stype = sm[1].str();
		if (varname == stype)
		{
			type_ = _OPT_VARTYPE;
		}
		else if (constname == stype)
		{
			type_ = _OPT_CONSTTYPE;
		}
		else if (scalrname == stype)
		{
			type_ = _OPT_SCALRTYPE;
		}
		else
		{
			age::_GENERATED_OPCODE opcode = age::get_op(stype);
			nallowed_children_ = get_nchildren(opcode);
			type_ = opcode;
		}
		tens_id_ = sm[2].str();
		shaper_id_ = sm[3].str();
		coorder_id_ = sm[4].str();
		changed_ = tens_id_.size() == 0;
	}

	std::string to_string (void) const
	{
		std::string stype;
		switch (type_)
		{
			case _OPT_VARTYPE:
				stype = varname;
				break;
			case _OPT_CONSTTYPE:
				stype = constname;
				break;
			case _OPT_SCALRTYPE:
				stype = scalrname;
				break;
			default:
				stype = age::name_op((age::_GENERATED_OPCODE) type_);
				assert(reserved_opnames.end() == reserved_opnames.find(stype));
		}
		std::string token_rep = stype + "(" + tens_id_ + ")";
		if (shaper_id_.size() > 0)
		{
			token_rep += "<" + shaper_id_ + ">";
		}
		if (coorder_id_.size() > 0)
		{
			token_rep += "[" + coorder_id_ + "]";
		}
		return token_rep;
	}

	std::string encode (size_t maximum_height)
	{
		std::vector<std::string> levels(maximum_height);
		encode_helper(levels.begin(), levels.end());
		std::vector<std::string> clean_levels;
		std::copy_if(levels.begin(), levels.end(),
			std::back_inserter(clean_levels),
			[](std::string& s) { return s.size() > 0; });
		return fmts::join(line_delim, clean_levels.begin(), clean_levels.end());
	}

	template <typename T>
	ead::NodeptrT<T> decode (ade::Shape outshape,
		IdTensMapT& tensmap, IdCoordMapT& coordmap)
	{
		if (type_ == _OPT_SCALRTYPE)
		{
			// return scalar
			double db = std::stod(tens_id_);
			return ead::make_constant_scalar<T>((T) db, outshape);
		}
		auto tit = tensmap.find(tens_id_);
		if (tensmap.end() != tit && false == changed_)
		{
			// tens already exists
			// variables, constants, or non-modified functor
			return ead::to_node<T>(tit->second);
		}
		if (type_ < 0)
		{
			logs::fatal("cannot decode unknown variable/constant/scalar");
		}
		ead::ArgsT<T> children;
		for (TokenptrT child : children_)
		{
			ade::Shape childshape = outshape;
			ade::CoordptrT shaper;
			ead::CoordptrT coorder;
			auto sit = coordmap.find(child->shaper_id_);
			auto cit = coordmap.find(child->coorder_id_);
			if (coordmap.end() != sit)
			{
				shaper = sit->second;
				auto revshaper = shaper->reverse();
				ade::CoordT out;
				ade::CoordT in;
				std::copy(outshape.begin(), outshape.end(), in.begin());
				revshaper->forward(out.begin(), in.begin());
				std::vector<ade::DimT> slist(ade::rank_cap);
				std::transform(out.begin(), out.end(), slist.begin(),
					[](ade::CDimT cd) -> ade::DimT
					{
						if (cd < 0)
						{
							cd = -cd - 1;
						}
						return std::round(cd);
					});
				childshape = ade::Shape(slist);
			}
			else
			{
				shaper = ade::identity;
			}
			if (coordmap.end() != cit)
			{
				coorder = std::static_pointer_cast<
					ead::CoordMap>(cit->second);
			}
			auto child_node = child->template decode<T>(
				childshape, tensmap, coordmap);
			children.push_back(
				ead::FuncArg<T>(child_node, shaper, coorder));
		}
		age::_GENERATED_OPCODE opcode = (age::_GENERATED_OPCODE) type_;
		return ead::make_functor<T>(
			ade::Opcode{age::name_op(opcode), opcode}, children);
	}

	// Resources always initialized on allocation
	int type_;

	std::string tens_id_;

	size_t nallowed_children_ = 0;

	bool changed_ = false;

	// No guarantee on valid (non-default) init in allocation
	std::string shaper_id_;

	std::string coorder_id_;

	std::vector<TokenptrT> children_;

private:
	void encode_helper (
		std::vector<std::string>::iterator start,
		std::vector<std::string>::iterator last)
	{
		if (start == last)
		{
			return;
		}
		if (start->size() > 0)
		{
			*start += ",";
		}
		*start += to_string();
		++start;
		for (TokenptrT& child : children_)
		{
			child->encode_helper(start, last);
		}
	}
};

}

#endif // OPT_IREP_HPP
