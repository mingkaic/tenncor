#include "layr/rbm.hpp"

#ifndef LAYR_DBN_HPP
#define LAYR_DBN_HPP

namespace layr
{

struct DBN final : public iMarshalSet
{
	DBN (RBMptrT rbm, std::string label) :
		iMarshalSet(label), rbm_(rbm)
	{
		teq::DimT n_input = rbm_->get_ninput();
		teq::Shape weight_shape({rbm_->get_noutput(), n_input});
		teq::NElemT nweight = weight_shape.n_elems();

		PybindT bound = 1.0 / std::sqrt(n_input);
		std::uniform_real_distribution<PybindT> dist(-bound, bound);
		auto gen = [&dist]()
		{
			return dist(eteq::get_engine());
		};
		std::vector<PybindT> wdata(nweight);
		std::generate(wdata.begin(), wdata.end(), gen);

		eteq::VarptrT<PybindT> weight = eteq::make_variable<PybindT>(
				wdata.data(), weight_shape, "log_weight");

		eteq::VarptrT<PybindT> bias = eteq::make_variable_scalar<PybindT>(
			0.0, teq::Shape({hiddens.back()}), "log_bias");

		log_weight_ = std::make_shared<MarshalVar>(weight);
		log_bias_ = std::make_shared<MarshalVar>(bias);
	}

	DBN (const DBN& other) : iMarshalSet(other)
	{
		copy_helper(other);
	}

	DBN& operator = (const DBN& other)
	{
		if (this != &other)
		{
			iMarshalSet::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	DBN (DBN&& other) = default;

	DBN& operator = (DBN&& other) = default;

	// input of shape <n_input, n_batch>
	eteq::NodeptrT<PybindT> operator () (eteq::NodeptrT<PybindT> input)
	{
		eteq::NodeptrT<PybindT> output = (*rbm_)(input);
		return tenncor::softmax(tenncor::nn::fully_connect({output},
			{eteq::convert_to_node(log_weight_->var_)},
			eteq::convert_to_node(log_bias_->var_)));
	}

	uint8_t get_ninput (void) const
	{
		return rbm_->get_ninput();
	}

	uint8_t get_noutput (void) const
	{
		return rbm_->get_noutput();
	}

	MarsarrT get_subs (void) const override
	{
		return {rbm_, log_weight_, log_bias_};
	}

	RBMptrT rbm_;

	MarVarsptrT log_weight_;

	MarVarsptrT log_bias_;

private:
	void copy_helper (const DBN& other)
	{
		rbm_ = std::make_shared<RBM>(*other.rbm_);
		log_weight_ = std::make_shared<MarVarsptrT>(*other.log_weight_);
		log_bias_ = std::make_shared<MarVarsptrT>(*other.log_bias_);
	}

	iMarshaler* clone_impl (void) const override
	{
		return new DBN(*this);
	}
};

using DBNptrT = std::shared_ptr<DBN>;

}

#endif // LAYR_DBN_HPP