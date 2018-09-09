#include "ade/functor.hpp"

#include "llo/opmap.hpp"

struct Evaluable
{
	virtual ~Evaluable (void) = default;

	virtual GenericData evaluate (DTYPE dtype) = 0;

	virtual ade::Tensorptr inner (void) const = 0;
};

GenericData evaluate (DTYPE dtype, ade::iTensor* tens);

template <typename T>
struct Source final : public ade::iTensor, public Evaluable
{
	static ade::Tensorptr get (ade::Shape shape, std::vector<T> data)
	{
		if (shape.n_elems() != data.size())
		{
			util::handle_error("data size does not match shape",
				util::ErrArg<size_t>("data.size", data.size()),
				util::ErrArg<size_t>("shape.n_elems", shape.n_elems()));
		}
		return new Source(shape, data);
	}

	const ade::Shape& shape (void) const override
	{
		return tens_->shape();
	}

	ade::Tensorptr gradient (ade::Tensorptr& leaf) const override
	{
		Evaluable* eval = dynamic_cast<Evaluable*>(leaf.get());
		ade::Tensorptr wrt = nullptr == eval ? leaf : eval->inner();
		ade::Tensorptr out = tens_->gradient(wrt);
		// optimize out reshaper
		ade::iFunctor* f = dynamic_cast<ade::iFunctor*>(out.get());
		if (f == nullptr || ade::RESHAPE != f->get_code())
		{
			util::handle_error("source gradient not reshaped",
				util::ErrArg<std::string>("grad_op", ade::opname(
					f->get_code())));
		}
		const ade::Shape& gshape = f->shape();
		std::vector<T> gdata(gshape.n_elems());
		if (ade::Tensor::SYMBOLIC_ONE.get() == f->get_refs()[0])
		{
			std::fill(gdata.begin(), gdata.end(), (T) 1);
		}
		return Source::get(gshape, gdata);
	}

	std::string to_string (void) const override
	{
		return tens_->to_string();
	}

	GenericData evaluate (DTYPE dtype) override
	{
		DTYPE curtype = get_type<T>();
		GenericData out(tens_->shape(), curtype);
		std::memcpy(out.data_.get(), &data_[0], sizeof(T) * data_.size());
		if (curtype != dtype)
		{
			return out.convert_to(dtype);
		}
		return out;
	}

	ade::Tensorptr inner (void) const override
	{
		return tens_;
	}

private:
	Source (ade::Shape shape, std::vector<T>& data) :
		tens_(ade::Tensor::get(shape)), data_(data) {}

	ade::Tensorptr tens_;
	std::vector<T> data_;
};

// maintains other argument
template <typename... Args>
struct DirectWrapper final : public ade::iFunctor, public Evaluable
{
	static ade::Tensorptr get (ade::Tensorptr tens, Args... args)
	{
		if (nullptr == dynamic_cast<ade::iFunctor*>(tens.get()))
		{
			util::handle_error("wrapping non-functor");
		}
		std::tuple<Args...> tp(args...);
		return new DirectWrapper(tens, tp);
	}

	const ade::Shape& shape (void) const override
	{
		return tens_->shape();
	}

	ade::Tensorptr gradient (ade::Tensorptr& leaf) const override
	{
		if (Evaluable* eval = dynamic_cast<Evaluable*>(leaf.get()))
		{
			ade::Tensorptr wrt = eval->inner();
			return tens_->gradient(wrt);
		}
		return tens_->gradient(leaf);
	}

	std::string to_string (void) const override
	{
		return "Wrapper_" + opname(get_code()) + "<" + util::tuple_to_string(args_) + ">";
	}

	ade::OPCODE get_code (void) const override
	{
		return static_cast<ade::iFunctor*>(tens_.get())->get_code();
	}

	std::vector<ade::iTensor*> get_refs (void) const override
	{
		return static_cast<ade::iFunctor*>(tens_.get())->get_refs();
	}

	GenericData evaluate (DTYPE dtype) override
	{
		return eval_helper(dtype, std::index_sequence_for<Args...>());
	}

	ade::Tensorptr inner (void) const override
	{
		return tens_;
	}

private:
	DirectWrapper (ade::Tensorptr& tens, std::tuple<Args...>& args) :
		tens_(tens), args_(args) {}

	template <size_t... I>
	GenericData eval_helper (DTYPE dtype, std::index_sequence<I...>) const
	{
		ade::iFunctor* f = static_cast<ade::iFunctor*>(tens_.get());
		ade::OPCODE opcode = f->get_code();

		std::vector<ade::iTensor*> refs = f->get_refs();
		uint8_t nargs = refs.size();

		GenericData out(f->shape(), dtype);
		std::vector<GenericData> argdata(nargs);
		for (uint8_t i = 0; i < nargs; ++i)
		{
			argdata[i] = ::evaluate(dtype, refs[i]);
		}
		op_exec(opcode, out, argdata, std::get<I>(args_)...);
		return out;
	}

	ade::Tensorptr tens_;
	std::tuple<Args...> args_;
};
