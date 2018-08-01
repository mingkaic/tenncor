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
	static Varptr get (Shape shape);

	template <typename T>
	static Varptr get (Shape shape, std::vector<T> data)
	{
		if (data.size() != shape.n_elems())
		{
			handle_error("vector size not fitting shape",
				ErrArg<size_t>{"vecsize", data.size()},
				ErrArg<std::string>{"shape", shape.to_string()});
		}
		Variable* out = new Variable(shape);
		out->set_data(data);
		return Varptr(out);
	}

	DataSource calculate (void) override;

	Nodeptr gradient (Nodeptr& leaf) const override;

	Shape shape (void) const override;

	template <typename T>
	void set_data (std::vector<T> data)
	{
		if (data.size() != shape_.n_elems())
		{
			handle_error("vector size not fitting shape",
				ErrArg<size_t>{"vecsize", data.size()},
				ErrArg<std::string>{"shape", shape_.to_string()});
		}
		ds_ = std::make_unique<DataSource>(
			(char*) &data[0], get_type<T>(), shape_.n_elems());
	}

private:
	Variable (Shape shape);

	std::unique_ptr<DataSource> ds_;
	Shape shape_;
};

template <typename T>
void Varptr::set_data (std::vector<T> data)
{
	return static_cast<Variable*>(this->ptr_.get())->set_data<T>(data);
}

#endif /* VARIABLE_HPP */
