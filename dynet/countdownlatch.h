#ifndef __COUNTDOWNLATCH__
#define __COUNTDOWNLATCH__

#include <inttypes.h>
#include <stdint.h>
#include <atomic>


namespace clatch 
{
    class countdownlatch {
    public:
        countdownlatch(int count) { this->count.store(count); }

        void await() {
           while (1) {
              if (count.load() == 0) return;
           }
        }

        
        void count_down() { count--; }

    private:
        std::atomic<int> count;
        
        countdownlatch() = delete;
        countdownlatch(const countdownlatch& other) = delete;
        countdownlatch& operator=(const countdownlatch& opther) = delete;
    };
}

#endif
