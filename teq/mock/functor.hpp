#include "marsh/objs.hpp"

#include "teq/ifunctor.hpp"

#ifndef TEQ_MOCK_FUNCTOR_HPP
#define TEQ_MOCK_FUNCTOR_HPP

struct MockFunctor final : public teq::iFunctor
{
	MockFunctor (teq::TensptrsT tens,
		teq::Opcode opcode = teq::Opcode{},
		std::unordered_map<std::string,std::vector<double>> attrs = {}) :
		opcode_(opcode),
		shape_(tens.front()->shape()),
		args_(tens)
	{
		for (auto apair : attrs)
		{
			auto aval = apair.second;
			if (aval.size() > 0)
			{
				attrs_.add_attr(apair.first,
					std::make_unique<marsh::NumArray<double>>(aval));
			}
		}
	}

	MockFunctor (const MockFunctor& other) :
		opcode_(other.opcode_),
		shape_(other.shape_),
		args_(other.args_)
	{
		std::unique_ptr<marsh::Maps> oattr(other.attrs_.clone());
		attrs_ = std::move(*oattr);
	}

	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	teq::Shape shape (void) const override
	{
		return shape_;
	}

	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	teq::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	teq::TensptrsT get_children (void) const override
	{
		return args_;
	}

	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return attrs_.get_attr(attr_name);
	}

	std::vector<std::string> ls_attrs (void) const override
	{
		return attrs_.ls_attrs();
	}

	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override
	{
		attrs_.add_attr(attr_key, std::move(attr_val));
	}

	void rm_attr (std::string attr_key) override
	{
		attrs_.rm_attr(attr_key);
	}

	void update_child (teq::TensptrT arg, size_t index) override
	{
		args_[index] = arg;
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockFunctor(*this);
	}

	teq::Opcode opcode_;

	teq::Shape shape_;

	teq::TensptrsT args_;

	marsh::Maps attrs_;
};

#endif // TEQ_MOCK_FUNCTOR_HPP
