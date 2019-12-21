#include "marsh/objs.hpp"

#include "teq/ifunctor.hpp"

#ifndef TEQ_MOCK_OPFUNC_HPP
#define TEQ_MOCK_OPFUNC_HPP

struct MockFunctor : public teq::iFunctor
{
	MockFunctor (teq::TensptrsT datas, std::vector<double> data,
		teq::Opcode opcode = teq::Opcode{}) : datas_(datas),
		opcode_(opcode), data_(data, datas.front()->shape()) {}

	MockFunctor (const MockFunctor& other) : updated_(other.updated_),
		datas_(other.datas_), opcode_(other.opcode_), data_(other.data_) {}

	virtual ~MockFunctor (void) = default;

	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(*this);
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
		return datas_;
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
		datas_[index] = arg;
	}

	void calc(void) override
	{
		updated_ = true;
	}

	void* data (void) override
	{
		return data_.data();
	}

	const void* data (void) const override
	{
		return data_.data();
	}

	teq::Shape shape (void) const override
	{
		return data_.shape();
	}

	size_t type_code (void) const override
	{
		return data_.type_code();
	}

	std::string type_label (void) const override
	{
		return data_.type_label();
	}

	size_t nbytes (void) const override
	{
		return data_.nbytes();
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockFunctor(*this);
	}

	bool updated_ = false;

	teq::TensptrsT datas_;

	teq::Opcode opcode_;

	marsh::Maps attrs_;

	MockLeaf data_;
};

#endif // TEQ_MOCK_OPFUNC_HPP
