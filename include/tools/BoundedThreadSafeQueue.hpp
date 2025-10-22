// BoundedThreadSafeQueue.hpp
#ifndef BOUNDED_THREAD_SAFE_QUEUE_HPP
#define BOUNDED_THREAD_SAFE_QUEUE_HPP

/**
 * @file BoundedThreadSafeQueue.hpp
 * @brief A thread-safe bounded queue implementation.
 * 
 * This class implements a bounded thread-safe queue with support for 
 * enqueueing and dequeueing items, and includes debugging utilities 
 * that can be enabled or disabled based on the DEBUG_MODE flag.
 */

// Include necessary libraries
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "tools/Debug.hpp" // Include the config file to access the flags


template<typename T>
class BoundedThreadSafeQueue {
public:
    BoundedThreadSafeQueue(size_t max_size) : max_size_(max_size), finished(false) {
        DEBUG_PRINT("BoundedThreadSafeQueue initialized with max size: " << max_size_);
    }

    bool enqueue(T item) {
        std::unique_lock<std::mutex> lock(m);
        cv_not_full.wait(lock, [this]() { return q.size() < max_size_ || finished; });
        if (finished) return false;
        q.push(std::move(item));
        DEBUG_PRINT("Enqueued item, current queue size: " << q.size());
        cv_not_empty.notify_one();
        return true;
    }

    bool dequeue(T& item) {
        std::unique_lock<std::mutex> lock(m);
        cv_not_empty.wait(lock, [this]() { return !q.empty() || finished; });
        if (q.empty()) return false;
        item = std::move(q.front());
        q.pop();
        DEBUG_PRINT("Dequeued item, current queue size: " << q.size());
        cv_not_full.notify_one();
        return true;
    }

    void set_finished() {
        std::unique_lock<std::mutex> lock(m);
        finished = true;
        DEBUG_PRINT("Queue marked as finished.");
        cv_not_empty.notify_all();
        cv_not_full.notify_all();
    }

private:
    std::queue<T> q;
    size_t max_size_;
    std::mutex m;
    std::condition_variable cv_not_empty;
    std::condition_variable cv_not_full;
    bool finished;
};

#endif // BOUNDED_THREAD_SAFE_QUEUE_HPP
