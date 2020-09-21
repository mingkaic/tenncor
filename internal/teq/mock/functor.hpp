
#ifndef TEQ_MOCK_FUNCTOR_HPP
#define TEQ_MOCK_FUNCTOR_HPP

#include "internal/teq/teq.hpp"

#include "internal/teq/mock/leaf.hpp"
#include "internal/teq/mock/meta.hpp"

struct MockFunctor : public teq::iFunctor
{
	MockFunctor (teq::TensptrsT children, std::vector<double> data, teq::Opcode opcode) :
		children_(children), opcode_(opcode), data_(data, children.front()->shape()) {}

	MockFunctor (teq::TensptrsT children, std::vector<double> data) :
		MockFunctor(children, data, teq::Opcode()) {}

	MockFunctor (teq::TensptrsT children, teq::Opcode opcode) :
		MockFunctor(children, {}, opcode) {}

	MockFunctor (teq::TensptrsT children) :
		MockFunctor(children, {}, teq::Opcode()) {}

	MockFunctor (const MockFunctor& other) :
		children_(other.children_), opcode_(other.opcode_), data_(other.data_) {}

	MockFunctor (MockFunctor&& other) :
		children_(std::move(other.children_)),
		opcode_(std::move(other.opcode_)),
		data_(std::move(other.data_)) {}

	MockFunctor& operator = (const MockFunctor& other)
	{
		if (&other != this)
		{
			children_ = other.children_;
			opcode_ = other.opcode_;
			data_ = other.data_;
		}
		return *this;
	}

	MockFunctor& operator = (MockFunctor&& other)
	{
		if (&other != this)
		{
			children_ = std::move(other.children_);
			opcode_ = std::move(other.opcode_);
			data_ = std::move(other.data_);
		}
		return *this;
	}

	virtual ~MockFunctor (void) = default;

	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	teq::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	teq::TensptrsT get_args (void) const override
	{
		return children_;
	}

	types::StringsT ls_attrs (void) const override
	{
		return attrs_.ls_attrs();
	}

	const marsh::iObject* get_attr (const std::string& attr_key) const override
	{
		return attrs_.get_attr(attr_key);
	}

	marsh::iObject* get_attr (const std::string& attr_key) override
	{
		return attrs_.get_attr(attr_key);
	}

	void add_attr (const std::string& attr_key, marsh::ObjptrT&& attr_val) override
	{
		attrs_.add_attr(attr_key, std::move(attr_val));
	}

	void rm_attr (const std::string& attr_key) override
	{
		attrs_.rm_attr(attr_key);
	}

	size_t size (void) const override
	{
		return attrs_.size();
	}

	void update_child (teq::TensptrT arg, size_t index) override
	{
		children_[index] = arg;
	}

	teq::iDeviceRef& device (void) override
	{
		return data_.device();
	}

	const teq::iDeviceRef& device (void) const override
	{
		return data_.device();
	}

	const teq::iMetadata& get_meta (void) const override
	{
		return meta_;
	}

	teq::Shape shape (void) const override
	{
		return data_.shape();
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new MockFunctor(*this);
	}

	bool updated_ = false;

	teq::TensptrsT children_;

	teq::Opcode opcode_;

	marsh::Maps attrs_;

	MockLeaf data_;

	MockMeta meta_;
};

#endif // TEQ_MOCK_FUNCTOR_HPP
