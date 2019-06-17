#include "experimental/opt/rule/writer.hpp"

#ifdef OPT_RULE_WRITER_HPP

namespace opt
{

namespace rule
{

void communtative_sort (ade::ArgsT& args)
{
	ade::ArgsT imms;
	ade::ArgsT funcs;
	ade::ArgsT vars;
	imms.reserve(args.size());
	funcs.reserve(args.size());
	vars.reserve(args.size());
	for (auto& arg : args)
	{
		if (auto leaf = dynamic_cast<ade::iLeaf*>(arg.get_tensor().get()))
		{
			if (tag::has_property(leaf, tag::immutable_tag))
			{
				imms.push_back(arg);
			}
			else
			{
				vars.push_back(arg);
			}
		}
		else
		{
			funcs.push_back(arg);
		}
	}
	std::sort(imms.begin(), imms.end(),
		[](ade::FuncArg a, ade::FuncArg b)
		{
			return lt(a, b,
			[](const ade::TensptrT& a, const ade::TensptrT& b)
			{
				return is_equal(
                    static_cast<ade::iLeaf*>(a.get()),
					static_cast<ade::iLeaf*>(b.get()));
			},
			[](const ade::TensptrT& a, const ade::TensptrT& b)
			{
				return lt({},
					static_cast<ade::iLeaf*>(a.get()),
					static_cast<ade::iLeaf*>(b.get()));
			});
		});
	std::sort(funcs.begin(), funcs.end(),
		[](ade::FuncArg a, ade::FuncArg b)
		{
			return lt(a, b,
			[](const ade::TensptrT& a, const ade::TensptrT& b)
			{
				return is_equal(
                    static_cast<ade::iFunctor*>(a.get()),
					static_cast<ade::iFunctor*>(b.get()));
			},
			[](const ade::TensptrT& a, const ade::TensptrT& b)
			{
				return lt({},
					static_cast<ade::iFunctor*>(a.get()),
					static_cast<ade::iFunctor*>(b.get()));
			});
		});
	std::sort(vars.begin(), vars.end(),
		[](ade::FuncArg a, ade::FuncArg b)
		{
			return lt(a, b,
			[](const ade::TensptrT& a, const ade::TensptrT& b)
			{
				return a->to_string() == b->to_string();
			},
			[](const ade::TensptrT& a, const ade::TensptrT& b)
			{
				return a->to_string() < b->to_string();
			});
		});
	args.clear();
	args.insert(args.end(), imms.begin(), imms.end());
	args.insert(args.end(), funcs.begin(), funcs.end());
	args.insert(args.end(), vars.begin(), vars.end());
}

void communtative_sort (WriterArgsT& args)
{
	WriterArgsT sclrs;
	WriterArgsT funcs;
	WriterArgsT groups;
	WriterArgsT anys;
	sclrs.reserve(args.size());
	funcs.reserve(args.size());
	groups.reserve(args.size());
	anys.reserve(args.size());
	for (auto& arg : args)
	{
		if (auto sclr = dynamic_cast<ScalarWriter*>(arg.arg_.get()))
		{
			sclrs.push_back(arg);
		}
		else if (auto sclr = dynamic_cast<AnyWriter*>(arg.arg_.get()))
		{
			anys.push_back(arg);
		}
		else if (auto sclr = dynamic_cast<FuncWriter*>(arg.arg_.get()))
		{
			funcs.push_back(arg);
		}
		else
		{
			groups.push_back(arg);
		}
	}
	std::sort(sclrs.begin(), sclrs.end(),
		[](WriterArg a, WriterArg b)
		{
            auto lhs = static_cast<ScalarWriter*>(a.arg_.get());
            auto rhs = static_cast<ScalarWriter*>(b.arg_.get());
            if (lhs->scalar_ == rhs->scalar_)
            {
                return lt(a.coorder_, b.coorder_);
            }
			return lhs->scalar_ < rhs->scalar_;
		});
	std::sort(funcs.begin(), funcs.end(),
		[](WriterArg a, WriterArg b)
		{
            auto lhs = static_cast<FuncWriter*>(a.arg_.get());
            auto rhs = static_cast<FuncWriter*>(b.arg_.get());
            if (lhs->op_ == rhs->op_)
            {
                return lt(a.coorder_, b.coorder_);
            }
			return lhs->op_ < rhs->op_;
		});
	std::sort(groups.begin(), groups.end(),
		[](WriterArg a, WriterArg b)
		{
            auto lhs = static_cast<GroupWriter*>(a.arg_.get());
            auto rhs = static_cast<GroupWriter*>(b.arg_.get());
            if (lhs->group_id_ == rhs->group_id_)
            {
                return lt(a.coorder_, b.coorder_);
            }
			return lhs->group_id_ < rhs->group_id_;
		});
	std::sort(anys.begin(), anys.end(),
		[](WriterArg a, WriterArg b)
		{
            auto lhs = static_cast<AnyWriter*>(a.arg_.get());
            auto rhs = static_cast<AnyWriter*>(b.arg_.get());
            if (lhs->id_ == rhs->id_)
            {
                return lt(a.coorder_, b.coorder_);
            }
			return lhs->id_ < rhs->id_;
		});
	args.clear();
	args.insert(args.end(), sclrs.begin(), sclrs.end());
	args.insert(args.end(), funcs.begin(), funcs.end());
	args.insert(args.end(), groups.begin(), groups.end());
	args.insert(args.end(), anys.begin(), anys.end());
}

}

}

#endif
