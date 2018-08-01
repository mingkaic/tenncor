#include "soil/variable.hpp"

#ifdef VARIABLE_HPP

Varptr Variable::get (Shape shape)
{
	return Varptr(new Variable(shape));
}

DataSource Variable::calculate (void)
{
	if (ds_ == nullptr)
	{
		handle_error("calculating uninitialized variable");
	}
	return *ds_;
}

Nodeptr Variable::gradient (Nodeptr& leaf) const
{
	if (ds_ == nullptr)
	{
		handle_error(
			"eliciting gradient from uninitialized variable");
	}
	if (this == leaf.get())
	{
		return get_one(shape_, ds_->type());
	}
	return get_zero(shape_, ds_->type());
}

Shape Variable::shape (void) const
{
	return shape_;
}

Variable::Variable (Shape shape) : shape_(shape) {}

Varptr::Varptr (Variable* var) : Nodeptr(var) {}

#endif
