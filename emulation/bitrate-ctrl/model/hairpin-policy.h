#ifndef HAIRPIN_POLICY_H
#define HAIRPIN_POLICY_H

#include "common-header.h"
#include "fec-policy.h"
#include "ns3/object.h"
#include "ns3/nstime.h"
#include <vector>

namespace ns3{

class HairpinPolicy : public FECPolicy {
public:
    HairpinPolicy(uint16_t delayDdl, double_t qoeCoeffPow, bool isRtx, bool isCap);
    HairpinPolicy();
    ~HairpinPolicy();
    static TypeId GetTypeId (void);
private:
    static const std::vector<uint8_t> group_size_list;
    uint16_t k_delayDdl;
    uint8_t k_qoeCoeffPow;
    double_t k_qoeCoeff;
    static const int GROUP_SIZE_ITVL = 5;
    int k_betaArraySize;
    int k_blockArraySize;
    std::string k_paramDir;

    
    uint8_t *m_betaArray;
    uint8_t *m_blockArray;

    bool m_isRtx;
    bool m_isCap;
    bool m_isBlockSizeOpt;
private:
    uint8_t GetFecCnt (double_t loss, uint8_t frameSize, uint16_t remainingTime, uint16_t rtt, uint8_t packet);
    uint8_t GetBlockSize (double_t loss, uint8_t frame_size, uint16_t ddl, uint16_t rtt, double_t rdisp);
public:
    FECParam GetPolicyFECParam(
        Ptr<NetStat> statistic, uint32_t bitrate,
        uint16_t ddl, uint16_t ddl_left,
        bool is_rtx, uint8_t frame_size,
        uint16_t max_group_size, bool fix_group_size
    );
    std::string GetFecName (void);
};  // class HairpinPolicy

}; // namespace ns3

#endif /* HAIRPIN_POLICY_H */