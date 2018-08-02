#include "sand/node.hpp"

#include "soil/data.hpp"
#include "soil/constant.hpp"

#include "util/error.hpp"

#ifndef VARIABLE_HPP
#define VARIABLE_HPP

struct Variable;

struct Varptr : public Nodeptr
{
	Varptr (Variable* var);

	template <typename T>
	void set_data (std::vector<T> data);
};

struct Variable final : public Node
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

	Variable (const Variable& other) : Node(other)
	{
		if (nullptr != other.data_)
		{
			data_ = make_data(other.data_.get(), other.nbytes());
		}
	}

	Variable (Variable&&) = default;

	Variable& operator = (const Variable& other)
	{
		if (this != &other)
		{
			Node::operator = (other);
			if (nullptr != other.data_)
			{
				data_ = make_data(other.data_.get(), other.nbytes());
			}
			else
			{
				data_ = nullptr;
			}
		}
		return *this;
	}

	Variable& operator = (Variable&&) = default;

	std::shared_ptr<char> calculate (Session& sess) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

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
		data_ = make_data((char*) &data[0], nbytes());
	}

private:
	Variable (Shape shape, DTYPE type);

	std::shared_ptr<char> data_ = nullptr; // todo: make unique
};

template <typename T>
void Varptr::set_data (std::vector<T> data)
{
	return static_cast<Variable*>(this->ptr_.get())->set_data<T>(data);
}

#endif /* VARIABLE_HPP */
