
#ifndef COMMON_HEADER_H
#define COMMON_HEADER_H


// Enable debug output
// #define ENABLE_DEBUG_OUTPUT

#ifdef ENABLE_DEBUG_OUTPUT
#define DEBUG(msg)                                    \
    do {                                              \
        std::cout << msg << std::endl;                \
    } while (false)
#else
#define DEBUG(msg)
#endif

#define MIN(x, y) ((x) <= (y) ? (x) : (y))
#define MAX(x, y) ((x) <= (y) ? (y) : (x))

#include <cinttypes>
#include <vector>
#include <string>

static inline bool Uint16Less (int a, int b) 
{
    if (a < b && a + UINT16_MAX / 2 > b)
        return true;
    else if (a > b && a > b + UINT16_MAX / 2)
        return true;
    else
        return false;
}

static inline bool Uint64Less (uint64_t id1, uint64_t id2) 
{
    uint64_t noWarpSubtract = id2 - id1;
    uint64_t wrapSubtract = id1 - id2;
    return noWarpSubtract < wrapSubtract;
}

static inline void SplitString (const std::string& s, std::vector<std::string>& v, const std::string& c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while (std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));      
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if (pos1 != s.length ())
    v.push_back(s.substr(pos1));
}

#endif /* COMMON_HEADER_H */
