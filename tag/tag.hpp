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
			return tag_types.size();
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

	void add (std::unique_ptr<iTag> entry)
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
	std::unordered_map<size_t,std::unique_ptr<iTag>> tags_;

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
// instead of singleton
struct Registry final
{
	static std::unordered_map<TensKey,TagCollective,
		TensKeyHash> registry; // todo: make thread-safe

	Registry (void) = delete;
};

TagRepsT get_tags (const ade::iTensor* tens);

void erase (const ade::iTensor* tens);

void move_tags (const ade::iTensor* dest, const ade::iTensor* source);

}

#endif // TAG_TAG_HPP
