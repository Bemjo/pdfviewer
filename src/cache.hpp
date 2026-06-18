/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                           *
 *  Copyright (C) 2012-2020 Chuan Ji                                         *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *   http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// This file defines the template class Cache, which is a fixed-size generic
// cache that stores key-value pairs. Users will need to supply methods to load
// and free elements in child classes.

#ifndef CACHE_HPP
#define CACHE_HPP

#include <cassert>
#include <condition_variable>
#include <map>
#include <mutex>
#include <deque>
#include <set>
#include <thread>
#include <vector>

// A generic cache that stores <key, value> pairs. The semantics for Load() and
// Discard() are implemented in implementing child classes. Supports
// asynchronous pre-emptive loading with C++11 threads. For performance,
// multiple instances of Load() and Discard() may be executed at the same time,
// so these latter MUST be thread-safe. The class assumes K and V are copy-able
// and also cheap to copy; thus, they should be either primitive values or
// pointers.
template <typename K, typename V>
class Cache {
 public:
  // Create a cache with the given maximum delta per page
  // real size is size*2+1
  explicit Cache(int size);

  // DOES NOT CLEAR CACHE because it cannot call the virtual function Discard.
  // Child classes MUST call Clear() in their destructors. Waits for background
  // loading threads to terminate first.
  virtual ~Cache();

  // Returns true if key is currently in the cache.
  bool Contains(const K& key);

  // Retrieves an item. If the item is in the cache, simply returns it. If
  // not, loads it using the Load() function defined in an implementation.
  // refresh=false (default): if cached, returns immediately; otherwise pushes
  //   to the front of the work queue and blocks.
  // refresh=true: forces a refresh of the requested item before retrieving it
  // In both cases, blocks until the value is available.
  V Get(const K& key, bool refresh = false);

  // Schedules key to be loaded in the background (BACK of work queue).
  // No-op if key is already cached, queued, or currently loading.
  void Prepare(const K& key);

  // Clears all cached entries and the work queue.
  // Any load currently in progress is allowed to finish; its result is
  // stored normally (it will be the only entry in cache afterwards).
  void Flush();

  // Returns the size of the cache.
  int GetSize() const;

  // Clears the cache, calling Discard() on all existing elements. Waits for
  // background loading threads to terminate first. MUST BE CALLED from the
  // destructor of a child class.
  void Clear();

  // Sets the center key used by EvictBefore() to determine eviction priority.
  void SetCenter(const K& key);

  // Removes queued-but-not-started items whose keys fall outside [low, high].
  void CancelOutsideRange(const K& low, const K& high);

 protected:
  // Loads a new element. This should be overridden in child classes. MUST BE
  // THREAD-SAFE.
  virtual V Load(const K& key) = 0;

  // Frees an element that has been evicted from the cache. This should be
  // overridden in child classes. MUST BE THREAD-SAFE.
  virtual void Discard(const K& key, const V& value) = 0;

  // Returns true if key a should be evicted before key b given center.
  // Default preserves FIFO order.  Override for distance-based eviction.
  virtual bool EvictBefore(const K& a, const K& b, const K& center) const {
    return false;
  }

 private:
  void WorkerLoop();
  std::thread _worker;
  // A lock on this object. Calls to Get() and Prepare() will block for access.
  std::mutex _mutex;
  // A map from keys to values.
  std::map<K, V> _map;
  // Max size of this cache.
  int _size;
  // Keys that are being loaded by some thread.
  std::set<K> _work_set;
  // Condition variable used to broadcast work done.
  std::condition_variable _condition;
  // pending loads; front = highest priority
  std::deque<K> _work_queue;
  // wakes worker when queue has items
  std::condition_variable _worker_cv;
  // key the worker is currently loading
  K _loading_key;
  // true while worker is inside Load()
  bool _loading_active;
  // set by Clear() to ask the worker to exit
  bool _shutdown;
  // reference point for EvictBefore
  K _eviction_center;
};


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                              Implementation                               *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
template <typename K, typename V>
Cache<K, V>::Cache(int size)
    : _size(size*2 + 1),
      _loading_active(false),
      _shutdown(false) {
  _worker = std::thread(&Cache<K, V>::WorkerLoop, this);
}

template <typename K, typename V>
Cache<K, V>::~Cache() {
  // Child destructor must have already called Clear. If not (shouldn't happen),
  // do a best-effort shutdown without invoking virtual Discard.
  if (_worker.joinable()) {
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _shutdown = true;
    }
    _worker_cv.notify_all();
    _worker.join();
  }
}

template <typename K, typename V>
void Cache<K, V>::WorkerLoop() {
  for (;;) {
    K key;

    {
      std::unique_lock<std::mutex> lock(_mutex);
      _worker_cv.wait(lock, [this] {
        return _shutdown || !_work_queue.empty();
      });

      if (_shutdown && _work_queue.empty()) {
        return;
      }

      key = _work_queue.front();
      _work_queue.pop_front();
      _loading_key    = key;
      _loading_active = true;
    }

    V value = Load(key);

    std::vector<std::pair<K, V>> to_discard;
    {
      std::unique_lock<std::mutex> lock(_mutex);

      _loading_active = false;
      _work_set.erase(key);

      _map[key] = value;

      // Evict entries until we are within the size limit, preferring the key
      // that EvictBefore() ranks highest relative to _eviction_center.
      while (static_cast<int>(_map.size()) > _size) {
        auto victim = _map.begin();
        for (auto it = std::next(victim); it != _map.end(); ++it) {
          if (EvictBefore(it->first, victim->first, _eviction_center)) {
            victim = it;
          }
        }
        to_discard.emplace_back(victim->first, victim->second);
        _map.erase(victim);
      }

      _condition.notify_all();
    }

    // Discard evicted entries outside the lock.
    for (auto& entry : to_discard) {
      Discard(entry.first, entry.second);
    }
  }
}

template <typename K, typename V>
bool Cache<K, V>::Contains(const K& key) {
    std::unique_lock<std::mutex> lock(_mutex);
    auto i = _map.find(key);
    return i != _map.end();
}

template <typename K, typename V>
V Cache<K, V>::Get(const K& key, bool refresh) {
  std::unique_lock<std::mutex> lock(_mutex);

  if (refresh) {
    auto it = _map.find(key);
    if (it != _map.end()) {
      Discard(it->first, it->second);
      _map.erase(it);
    }
  }

  for (;;) {
    // Cache check: skipped on the very first iteration when force=true so
    // that a fresh Load() is triggered.  After the first wait, force is
    // cleared and subsequent iterations check normally — this prevents
    // spinning if a spurious wakeup occurs before the worker finishes.
    auto it = _map.find(key);
    if (it != _map.end()) {
      return it->second;
    }

    // Ensure this key is at the FRONT of the work queue (urgent).
    if (_loading_active && _loading_key == key) {
      // Worker is already loading this key — just wait.
    } else if (_work_set.count(key)) {
      // Key was queued by Prepare() at the back — promote it to the front.
      for (auto it = _work_queue.begin(); it != _work_queue.end(); ++it) {
        if (*it == key) {
          _work_queue.erase(it);
          break;
        }
      }
      _work_queue.push_front(key);
      _worker_cv.notify_one();
    } else {
      // Key is not queued at all — add it.
      _work_set.insert(key);
      _work_queue.push_front(key);
      _worker_cv.notify_one();
    }

    _condition.wait(lock);
  }
}

template <typename K, typename V>
void Cache<K, V>::Prepare(const K& key) {
  std::unique_lock<std::mutex> lock(_mutex);

  if (_map.count(key) ||
      _work_set.count(key) ||
      (_loading_active && _loading_key == key)) {
    return;
  }

  _work_set.insert(key);
  _work_queue.push_back(key);
  _worker_cv.notify_one();
}

template <typename K, typename V>
void Cache<K, V>::SetCenter(const K& key) {
  std::unique_lock<std::mutex> lock(_mutex);
  _eviction_center = key;
}

template <typename K, typename V>
void Cache<K, V>::CancelOutsideRange(const K& low, const K& high) {
  std::unique_lock<std::mutex> lock(_mutex);
  for (auto it = _work_queue.begin(); it != _work_queue.end(); ) {
    if (*it < low || *it > high) {
      _work_set.erase(*it);
      it = _work_queue.erase(it);
    } else {
      ++it;
    }
  }
}

template <typename K, typename V>
void Cache<K, V>::Flush() {
  std::vector<std::pair<K, V>> to_discard;
  {
    std::unique_lock<std::mutex> lock(_mutex);

    // Drain the work queue.
    for (const auto& k : _work_queue) {
      _work_set.erase(k);
    }
    _work_queue.clear();

    // Collect all cached entries for discard.
    for (auto& kv : _map) {
      to_discard.emplace_back(kv.first, kv.second);
    }
    _map.clear();

    // Any load currently in progress (_loading_active) is allowed to
    // finish.  Its result will be stored into the now-empty map normally and
    // _condition will be notified.  Any Get waiting will then find it there.
  }

  for (auto& entry : to_discard) {
    Discard(entry.first, entry.second);
  }
}

template <typename K, typename V>
int Cache<K, V>::GetSize() const {
  return _size;
}

template <typename K, typename V>
void Cache<K, V>::Clear() {
  // 1. Drain the pending queue and signal the worker to exit.
  //    Do NOT wait here under the lock — just set shutdown and release.
  //    join() below will wait for any in-progress Load() to finish cleanly
  //    without the risk of waking on a half-committed worker state.
  {
    std::unique_lock<std::mutex> lock(_mutex);
    for (const auto& k : _work_queue) {
      _work_set.erase(k);
    }
    _work_queue.clear();
    _shutdown = true;
  }
  _worker_cv.notify_all();

  if (_worker.joinable()) {
    _worker.join();
  }

  // 2. Discard all remaining cached entries on the calling thread.
  std::vector<std::pair<K, V>> to_discard;
  {
    std::unique_lock<std::mutex> lock(_mutex);
    for (auto& kv : _map) {
      to_discard.emplace_back(kv.first, kv.second);
    }
    _map.clear();
  }

  for (auto& entry : to_discard) {
    Discard(entry.first, entry.second);
  }
}

#endif
