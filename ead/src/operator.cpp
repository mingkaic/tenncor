#include "ead/operator.hpp"

#ifdef EAD_OPERATOR_HPP

namespace ead
{

/// Return global random generator
EngineT& get_engine (void)
{
	static EngineT engine;
	return engine;
}

template <>
EigenptrT<double> rand_uniform<double> (ade::Shape& outshape, const OpArg<double>& a, const OpArg<double>& b)
{
	return make_tensop<double>(outshape, a.tensmap_->binaryExpr(*b.tensmap_,
		[](const ScalarT<double>& a, const ScalarT<double>& b)
		{
			std::uniform_real_distribution<double> dist(a, b);
			return dist(get_engine());
		}));
}

template <>
EigenptrT<float> rand_uniform<float> (ade::Shape& outshape, const OpArg<float>& a, const OpArg<float>& b)
{
	return make_tensop<float>(outshape, a.tensmap_->binaryExpr(*b.tensmap_,
		[](const ScalarT<float>& a, const ScalarT<float>& b)
		{
			std::uniform_real_distribution<float> dist(a, b);
			return dist(get_engine());
		}));
}

template <>
EigenptrT<double> sin (ade::Shape& outshape, const OpArg<double>& in)
{
	return make_tensop<double>(outshape, in.tensmap_->unaryExpr(
		[](const ScalarT<double>& a)
		{
			return std::sin(a);
		}));
}

template <>
EigenptrT<float> sin (ade::Shape& outshape, const OpArg<float>& in)
{
	return make_tensop<float>(outshape, in.tensmap_->unaryExpr(
		[](const ScalarT<float>& a)
		{
			return std::sin(a);
		}));
}

template <>
EigenptrT<double> cos (ade::Shape& outshape, const OpArg<double>& in)
{
	return make_tensop<double>(outshape, in.tensmap_->unaryExpr(
		[](const ScalarT<double>& a)
		{
			return std::cos(a);
		}));
}

template <>
EigenptrT<float> cos (ade::Shape& outshape, const OpArg<float>& in)
{
	return make_tensop<float>(outshape, in.tensmap_->unaryExpr(
		[](const ScalarT<float>& a)
		{
			return std::cos(a);
		}));
}

template <>
EigenptrT<double> tan (ade::Shape& outshape, const OpArg<double>& in)
{
	return make_tensop<double>(outshape, in.tensmap_->unaryExpr(
		[](const ScalarT<double>& a)
		{
			return std::tan(a);
		}));
}

template <>
EigenptrT<float> tan (ade::Shape& outshape, const OpArg<float>& in)
{
	return make_tensop<float>(outshape, in.tensmap_->unaryExpr(
		[](const ScalarT<float>& a)
		{
			return std::tan(a);
		}));
}

}

#endif
