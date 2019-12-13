#include "marsh/objs.hpp"

#include "teq/iopfunc.hpp"

#ifndef TEQ_MOCK_OPFUNC_HPP
#define TEQ_MOCK_OPFUNC_HPP

struct MockOpfunc final : public teq::iOperableFunc
{
	// todo: make constructor similar to MockFunctor
	MockOpfunc (teq::TensptrT a,
		teq::Opcode opcode = teq::Opcode{},
		std::unordered_map<std::string,std::vector<double>> attrs = {}) :
		MockOpfunc(teq::TensptrsT{a}, opcode, attrs) {}

	MockOpfunc (teq::TensptrsT args,
		teq::Opcode opcode = teq::Opcode{},
		std::unordered_map<std::string,std::vector<double>> attrs = {}) :
		opcode_(opcode),
		shape_(args[0]->shape()),
		args_(args)
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

	MockOpfunc (const MockOpfunc& other) :
		updated_(other.updated_),
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

	void update (void) override
	{
		updated_ = true;
	}

	void* data (void) override
	{
		return nullptr;
	}

	const void* data (void) const override
	{
		return nullptr;
	}

	size_t type_code (void) const override
	{
		return 0;
	}

	std::string type_label (void) const override
	{
		return "";
	}

	size_t nbytes (void) const override
	{
		return 0;
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockOpfunc(*this);
	}

	bool updated_ = false;

	teq::Opcode opcode_;

	teq::Shape shape_;

	teq::TensptrsT args_;

	marsh::Maps attrs_;
};

#endif // TEQ_MOCK_OPFUNC_HPP
