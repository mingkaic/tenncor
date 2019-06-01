#include <functional>
#include <thread>
#include <future>

#ifndef JOB_MANAGED_HPP
#define JOB_MANAGED_HPP

namespace job
{

struct ManagedJob final
{
	ManagedJob (void) = default;

	template <typename FN, typename ...ARGS>
	ManagedJob (FN job, ARGS&&... args)
	{
		std::thread job_thd(job,
			exit_signal_.get_future(),
			std::forward<ARGS>(args)...);
		job_ = std::move(job_thd);
	}

	~ManagedJob (void)
	{
		if (job_.joinable())
		{
			exit_signal_.set_value();
			job_.detach();
		}
	}

	ManagedJob (const ManagedJob& other) = delete;

	ManagedJob (ManagedJob&& other) :
		exit_signal_(std::move(other.exit_signal_)),
		job_(std::move(other.job_)) {}

	ManagedJob& operator = (const ManagedJob& other) = delete;

	ManagedJob& operator = (ManagedJob&& other)
    {
        if (this != &other)
        {
            if (job_.joinable())
            {
                exit_signal_.set_value();
                job_.detach();
            }
            exit_signal_ = std::move(other.exit_signal_);
            job_ = std::move(other.job_);
        }
        return *this;
    }

	/// return thread id
	std::thread::id get_id (void) const
	{
		return job_.get_id();
	}

    /// return whether job is running
	bool is_running (void) const
	{
        // since job is never detached,
        // if job is running, then it's joinable
		return job_.joinable();
	}

	/// join if joinable
	void join (void)
	{
		if (job_.joinable())
		{
			job_.join();
		}
	}

	/// stop the job_
	void stop (void)
	{
		if (job_.joinable())
		{
			exit_signal_.set_value();
		}
	}

private:
	std::promise<void> exit_signal_;

	std::thread job_;
};

}

#endif // JOB_MANAGED_HPP
