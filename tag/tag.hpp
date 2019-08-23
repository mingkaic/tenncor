#include <map>
#include <set>

#include "ade/ade.hpp"

#ifndef TAG_TAG_HPP
#define TAG_TAG_HPP

namespace tag
{

using TagRepsT = std::map<std::string,std::vector<std::string>>;

// each tag instance is a set of tags with particular properties
// iTag is the interface for such instances
struct iTag
{
	virtual ~iTag (void) = default;

	virtual size_t tag_id (void) const = 0;

	virtual void absorb (std::unique_ptr<iTag>&& other) = 0;

	virtual TagRepsT get_tags (void) const = 0;
};

using TagptrT = std::unique_ptr<iTag>;

// TagCollective is a collective of generic iTag instances
// only 1 instance of a particular type of iTag can be stored in an instance of
// TagCollective, adding subsequent instances of the same type
// absorbs new instances into the collective
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

struct TensKey final
{
	TensKey (ade::TensrefT tens) : val_(tens.lock().get()), ref_(tens) {}

	// used to match keys
	TensKey (ade::iTensor* tens) : val_(tens) {}

	TensKey (const ade::iTensor* tens) : val_(tens) {}

	operator const ade::iTensor*() const
	{
		return val_;
	}

	bool expired (void) const
	{
		return ref_.expired();
	}

	const ade::iTensor* val_;

	ade::TensrefT ref_;
};

struct TensKeyHash final
{
	size_t operator() (const TensKey& key) const
	{
		return std::hash<const void*>()(key.val_);
	}
};

inline bool operator == (const TensKey& lhs, const TensKey& rhs)
{
	TensKeyHash hasher;
	return hasher(lhs) == hasher(rhs);
}

using TagrF = std::function<void(ade::TensrefT,std::string)>;

// todo: move tag registry to some session that claims global context
// todo: make an interface for this
struct TagRegistry final
{
	void add_tag (ade::TensrefT tens, TagptrT tag)
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

	TagRepsT get_tags (const ade::iTensor* tens)
	{
		auto it = registry_.find(TensKey(tens));
		if (registry_.end() == it || it->first.expired())
		{
			return {};
		}
		return it->second.get_tags();
	}

	void move_tags (ade::TensrefT dest, const ade::iTensor* source)
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

	std::string register_tagr (std::string tag_key, TagrF tagr)
	{
		key_tagr_assoc_.emplace(tag_key, tagr);
		return tag_key;
	}

	std::unordered_map<TensKey,TagCollective,TensKeyHash> registry_;

private:
	std::unordered_map<std::string,TagrF> key_tagr_assoc_;
};

TagRegistry& get_reg (void);

void recursive_tag (ade::TensptrT root,
	std::unordered_set<ade::iTensor*> stops,
	std::function<void(ade::TensrefT)> tag_op);

}

#endif // TAG_TAG_HPP
