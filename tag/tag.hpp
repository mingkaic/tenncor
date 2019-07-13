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
	template <typename TAG>
	static size_t register_tag (void)
	{
		static_assert(std::is_base_of<iTag,TAG>::value,
			"collective tags must inherit iTag");
		const std::type_info& tp = typeid(TAG);
		size_t code = tp.hash_code();
		auto& tag_types = get_types();
		auto it = tag_types.find(code);
		if (tag_types.end() == it)
		{
			tag_types.emplace(code);
		}
		return code;
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
		auto& tag_types = get_types();
		if (false == estd::has(tag_types, tid))
		{
			logs::fatalf("cannot find tag type %d", tid);
		}
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

	static std::unordered_set<size_t>& get_types (void)
	{
		static std::unordered_set<size_t> tag_types;
		return tag_types;
	};
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

// todo: move tag registry to some session that claims global context
struct TagRegistry final
{
	// return collective referenced by tens
	TagCollective& get_collective (ade::TensrefT tens)
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
		return registry_[tens];
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

	void erase (const ade::iTensor* tens)
	{
		registry_.erase(TensKey(tens));
	}

	void move_tags (const ade::iTensor* dest, const ade::iTensor* source)
	{
		auto src_it = registry_.find(TensKey(source));
		auto dest_it = registry_.find(TensKey(dest));
		if (registry_.end() == src_it || src_it->first.expired() ||
			registry_.end() == dest_it || dest_it->first.expired())
		{
			return;
		}

		dest_it->second.absorb(std::move(src_it->second));
	}

	std::unordered_map<TensKey,TagCollective,TensKeyHash> registry_;
};

TagRegistry& get_reg (void);

}

#endif // TAG_TAG_HPP
