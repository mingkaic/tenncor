#include "rocnnet/modl/rbm.hpp"

#ifndef MODL_DBN_HPP
#define MODL_DBN_HPP

namespace modl
{

struct DBN final : public iMarshalSet
{
	DBN (uint8_t n_input, std::vector<uint8_t> hiddens, std::string label) :
		iMarshalSet(label),
		log_layer_(std::make_shared<FCon>(std::vector<uint8_t>{n_input},
			hiddens.back(), "logres"))
	{
		if (hiddens.empty())
		{
			logs::fatal("cannot db train with no hiddens");
		}
		for (size_t level = 0, n = hiddens.size(); level < n; ++level)
		{
			uint8_t n_output = hiddens[level];
			layers_.push_back(std::make_shared<RBM>(n_input, n_output,
				fmts::sprintf("rbm_%d", level)));
			n_input = n_output;
		}
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
	ead::NodeptrT<PybindT> operator () (ead::NodeptrT<PybindT> input)
	{
		// sanity check
		const ade::Shape& in_shape = input->shape();
		uint8_t ninput = get_ninput();
		if (in_shape.at(0) != ninput)
		{
			logs::fatalf("cannot dbn with input shape %s against n_input %d",
				in_shape.to_string().c_str(), ninput);
		}
		ead::NodeptrT<PybindT> output = input;
		for (RBMptrT& h : layers_)
		{
			output = h->prop_up(output);
		}
		return eqns::softmax((*log_layer_)({output}));
	}

	uint8_t get_ninput (void) const
	{
		return layers_.front()->get_ninput();
	}

	uint8_t get_noutput (void) const
	{
		return log_layer_->get_noutput();
	}

	std::vector<RBMptrT> get_layers (void) const
	{
		return layers_;
	}

	MarsarrT get_subs (void) const override
	{
		MarsarrT out(layers_.begin(), layers_.end());
		out.push_back(log_layer_);
		return out;
	}

private:
	void copy_helper (const DBN& other)
	{
		layers_.clear();
		for (const RBMptrT& olayer : other.layers_)
		{
			layers_.push_back(
				std::make_shared<RBM>(*olayer));
		}
		log_layer_ = std::make_shared<FCon>(*other.log_layer_);
	}

	iMarshaler* clone_impl (void) const override
	{
		return new DBN(*this);
	}

	std::vector<RBMptrT> layers_;

	FConptrT log_layer_;
};

using DBNptrT = std::shared_ptr<DBN>;

}

#endif // MODL_DBN_HPP
