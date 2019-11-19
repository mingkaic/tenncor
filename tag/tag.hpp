///
/// tag.hpp
/// tag
///
/// Purpose:
/// Define tag interface and tag registry
///

#include <map>
#include <set>

#include "teq/teq.hpp"

#ifndef TAG_TAG_HPP
#define TAG_TAG_HPP

namespace tag
{

/// Map tag key to a series of labels
using TagRepsT = std::map<std::string,std::vector<std::string>>;

/// Interface for tag instances of particular property that
/// store key and labels
struct iTag
{
	virtual ~iTag (void) = default;

	/// Return type hash of tag instance
	virtual size_t tag_id (void) const = 0;

	/// Add key and labels pairs of other
	virtual void absorb (std::unique_ptr<iTag>&& other) = 0;

	/// Return all key-labels associations
	virtual TagRepsT get_tags (void) const = 0;
};

/// Unique pointer of tag
using TagptrT = std::unique_ptr<iTag>;

/// Collective of generic iTag instances
//// only one instance of a particular type of iTag
/// can be stored in an instance of TagCollective,
/// adding subsequent instances of the same type
/// absorbs new instances into the collective
struct TagCollective final
{
	TagCollective (void) = default;

	TagCollective (TagCollective&& other) : tags_(std::move(other.tags_)) {}

	TagCollective& operator = (TagCollective&& other)
	{
		if (this != &other)
		{
			tags_ = std::move(other.tags_);
		}
		return *this;
	}

	/// Absorb iTags of the other collective
	void absorb (TagCollective&& other)
	{
		for (auto& tagpair : other.tags_)
		{
			size_t tid = tagpair.first;
			auto it = tags_.find(tid);
			if (tags_.end() == it)
			{
				tags_.emplace(tid, std::move(tagpair.second));
			}
			else
			{
				it->second->absorb(std::move(tagpair.second));
			}
		}
		other.tags_.clear();
	}

	/// Add new iTag if the entry is not in the collective,
	/// otherwise absorb the entry
	void add (TagptrT entry)
	{
		size_t tid = entry->tag_id();
		auto it = tags_.find(tid);
		if (tags_.end() == it)
		{
			tags_.emplace(tid, std::move(entry));
		}
		else
		{
			it->second->absorb(std::move(entry));
		}
	}

	/// Return all key-values under collected iTag
	TagRepsT get_tags (void) const
	{
		TagRepsT tags;
		for (auto& tpair : tags_)
		{
			auto temp = tpair.second->get_tags();
			tags.insert(temp.begin(), temp.end());
		}
		return tags;
	}

private:
	std::unordered_map<size_t,TagptrT> tags_;
};

/// Tensor ref key wrapper
struct TensKey final
{
	TensKey (teq::TensrefT tens) : val_(tens.lock().get()), ref_(tens) {}

	// used to match keys
	TensKey (teq::iTensor* tens) : val_(tens) {}

	TensKey (const teq::iTensor* tens) : val_(tens) {}

	/// Convert wrapper to hashable raw tensor pointer
	operator const teq::iTensor*() const
	{
		return val_;
	}

	/// Return true if weak reference is expired
	bool expired (void) const
	{
		return ref_.expired();
	}

	/// Raw tensor pointer for hashing
	/// (val_ can be nullptr or point to deleted tensor)
	const teq::iTensor* val_;

	/// Weak reference of tensor
	teq::TensrefT ref_;
};

/// TensKey hasher
struct TensKeyHash final
{
	size_t operator() (const TensKey& key) const
	{
		return std::hash<const void*>()(key.val_);
	}
};

/// TensKey equality overload
inline bool operator == (const TensKey& lhs, const TensKey& rhs)
{
	TensKeyHash hasher;
	return hasher(lhs) == hasher(rhs);
}

/// Function that associate tag key to tensor ref
using TagrF = std::function<void(teq::TensrefT,std::string)>;

// todo: move tag registry to some session that claims global context
// todo: make an interface for this
/// Registry for associating tensors to tag collectives
struct TagRegistry final
{
	/// Add tag to collective referenced by tens
	void add_tag (teq::TensrefT tens, TagptrT tag)
	{
		if (tens.expired())
		{
			logs::fatal("cannot tag with expired tensor ref");
		}
		auto it = registry_.find(TensKey(tens));
		// clear out previous entry that is expired
		if (registry_.end() != it && it->first.expired())
		{
			registry_.erase(tens.lock().get());
		}
		registry_[tens].add(std::move(tag));
	}

	/// Return all key-labels under the collective associated with tens
	TagRepsT get_tags (const teq::iTensor* tens)
	{
		auto it = registry_.find(TensKey(tens));
		if (registry_.end() == it || it->first.expired())
		{
			return {};
		}
		return it->second.get_tags();
	}

	/// Move all key-labels under collective associated with source
	/// to collective associated with dest
	/// If source did not have an associated collective,
	/// just create a new collective under dest
	/// Dest must not be expired, but source can be expired
	void move_tags (teq::TensrefT dest, const teq::iTensor* source)
	{
		if (dest.expired())
		{
			logs::fatal("cannot move with expired destination tensor");
		}
		auto src_it = registry_.find(TensKey(source));
		auto dest_it = registry_.find(TensKey(dest));
		if (registry_.end() == src_it || src_it->first.expired())
		{
			return;
		}

		if (registry_.end() == dest_it || dest_it->first.expired())
		{
			registry_[dest] = std::move(src_it->second);
		}
		else
		{
			dest_it->second.absorb(std::move(src_it->second));
		}
		registry_.erase(TensKey(source));
	}

	/// Return tagger associated to TagRepsT key
	TagrF tagr_by_key (std::string tag_key)
	{
		return estd::must_getf(key_tagr_assoc_, tag_key,
			"cannot find tagr associated with %s", tag_key.c_str());
	}

	/// Return successfully registered key of associated tagr
	/// Return value is the same as tag_key
	std::string register_tagr (std::string tag_key, TagrF tagr)
	{
		key_tagr_assoc_.emplace(tag_key, tagr);
		return tag_key;
	}

	/// Map tensor to tag collective
	std::unordered_map<TensKey,TagCollective,TensKeyHash> registry_;

private:
	std::unordered_map<std::string,TagrF> key_tagr_assoc_;
};

/// Return reference to global tag registry
TagRegistry& get_reg (void);

/// Recursive apply tag_op to nodes under root's graph
/// ignoring subgraphs of roots in stops set
void recursive_tag (teq::TensptrT root,
	teq::TensSetT stops,
	std::function<void(teq::TensrefT)> tag_op);

/// Map tag label to any tensor with label
using LTensT = std::unordered_map<std::string,std::vector<teq::iTensor*>>;

/// Map tag key to label-tensor association
using TTensT = std::unordered_map<std::string,LTensT>;

/// Implement traveler that gathers the tag-key+label to tensor
/// associations of visited subgraphs
struct Query final : public teq::OnceTraveler
{
	Query (TagRegistry& reg = get_reg()) : reg_(reg) {}

	/// Gather the tag key-label to tensor association of visited leaf
	void visit_leaf (teq::iLeaf* leaf) override
	{
		auto tags = reg_.get_tags(leaf);
		save_tags(tags, leaf);
	}

	/// Gather the tag key-label to tensor association of visited functor
	void visit_func (teq::iFunctor* func) override
	{
		auto children = func->get_children();
		for (const teq::iEdge& child : children)
		{
			child.get_tensor()->accept(*this);
		}

		auto tags = reg_.get_tags(func);
		save_tags(tags, func);
	}

	/// Map <key:label> to tensors found under tag regsitry
	TTensT labels_;

	/// Tag registry providing the reverse tensor to <key:labels> associations
	TagRegistry& reg_;

private:
	void save_tags (TagRepsT& tag, teq::iTensor* tens)
	{
		for (auto& tpair : tag)
		{
			auto& labs = labels_[tpair.first];
			for (auto lpair : tpair.second)
			{
				labs[lpair].push_back(tens);
			}
		}
	}
};

}

#endif // TAG_TAG_HPP
