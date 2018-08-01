#include "soil/inode.hpp"
#include "soil/error.hpp"
#include "soil/constant.hpp"

#ifndef VARIABLE_HPP
#define VARIABLE_HPP

struct Variable;

struct Varptr : public Nodeptr
{
	Varptr (Variable* var);

	template <typename T>
	void set_data (std::vector<T> data);
};

struct Variable final : public iNode
{
	static Varptr get (Shape shape, DTYPE type);

	template <typename T>
	static Varptr get (Shape shape, std::vector<T> data)
	{
		if (data.size() != shape.n_elems())
		{
			handle_error("vector size not fitting shape",
				ErrArg<size_t>{"vecsize", data.size()},
				ErrArg<std::string>{"shape", shape.to_string()});
		}
		Variable* out = new Variable(shape, get_type<T>());
		out->set_data(data);
		return Varptr(out);
	}

	Variable (const Variable& other) :
		shape_(other.shape_), type_(other.type_)
	{
		if (nullptr != other.data_)
		{
			data_ = make_data(other.data_.get(),
				type_size(other.type_) * other.shape_.n_elems());
		}
	}

	Variable (Variable&&) = default;

	Variable& operator = (const Variable& other)
	{
		if (this != &other)
		{
			if (nullptr != other.data_)
			{
				data_ = make_data(other.data_.get(),
					type_size(other.type_) * other.shape_.n_elems());
			}
			else
			{
				data_ = nullptr;
			}
			shape_ = other.shape_;
			type_ = other.type_;
		}
		return *this;
	}

	Variable& operator = (Variable&&) = default;

	std::shared_ptr<char> calculate (Session& sess) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

	Shape shape (void) const override;

	DTYPE type (void) const override
	{
		return type_;
	}

	template <typename T>
	void set_data (std::vector<T> data)
	{
		DTYPE settype = get_type<T>();
		if (settype != type_)
		{
			handle_error("set data with mismatch type",
				ErrArg<std::string>("set_type", name_type(settype)),
				ErrArg<std::string>("var_type", name_type(type_)));
		}
		if (data.size() != shape_.n_elems())
		{
			handle_error("vector size not fitting shape",
				ErrArg<size_t>{"vecsize", data.size()},
				ErrArg<std::string>{"shape", shape_.to_string()});
		}
		data_ = make_data((char*) &data[0],
			type_size(type_) * shape_.n_elems());
	}

private:
	Variable (Shape shape, DTYPE type);

	std::shared_ptr<char> data_ = nullptr; // todo: make unique
	Shape shape_;
	DTYPE type_;
};

template <typename T>
void Varptr::set_data (std::vector<T> data)
{
	return static_cast<Variable*>(this->ptr_.get())->set_data<T>(data);
}

#endif /* VARIABLE_HPP */
