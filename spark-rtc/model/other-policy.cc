#include "other-policy.h"

namespace ns3 {
NS_LOG_COMPONENT_DEFINE("OtherPolicies");


FixedPolicy::FixedPolicy () : FECPolicy(MilliSeconds(1)) {
    k_rate = 0.f;
};

FixedPolicy::FixedPolicy (double_t rate) : FECPolicy(MilliSeconds(1)) {
    k_rate = rate;
};

FixedPolicy::~FixedPolicy () {};

TypeId FixedPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::FixedPolicy")
        .SetParent<FECPolicy> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<FixedPolicy> ()
    ;
    return tid;
};

FECPolicy::FECParam FixedPolicy::GetPolicyFECParam (
    Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
    bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize) {
    if (!isRtx)
        return FECParam (maxGroupSize, k_rate);
    else
        return FECParam (maxGroupSize, 0);
};

std::string FixedPolicy::GetFecName (void) {
    return "FixedPolicy";
}

/* class FixedRtxPolicy */
FixedRtxPolicy::FixedRtxPolicy () : FixedPolicy (0) {};

FixedRtxPolicy::FixedRtxPolicy (double_t rate) : FixedPolicy (rate) {};

FixedRtxPolicy::~FixedRtxPolicy () {};

TypeId FixedRtxPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::FixedRtxPolicy")
        .SetParent<FixedPolicy> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<FixedRtxPolicy> ()
    ;
    return tid;
};

FECPolicy::FECParam FixedRtxPolicy::GetPolicyFECParam (
    Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
    bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize) {
    if (isRtx)
        return FECParam (maxGroupSize, k_rate);
    else
        return FECParam (maxGroupSize, 0);
};

std::string FixedRtxPolicy::GetFecName (void) {
    return "FixedRtxPolicy";
}

/* class TokenRtxPolicy */
TokenRtxPolicy::TokenRtxPolicy () : FixedPolicy (0)
, k_token {0}
, k_addRtx {false} {};

TokenRtxPolicy::~TokenRtxPolicy () {};

TypeId TokenRtxPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::TokenRtxPolicy")
        .SetParent<FixedPolicy> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<TokenRtxPolicy> ()
    ;
    return tid;
};

FECPolicy::FECParam TokenRtxPolicy::GetPolicyFECParam (
    Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
    bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize) {
    if (!isRtx) {
        k_token = frameSize * statistic->curLossRate;
        k_addRtx = bool (std::rand() % 2);
    }
    
    if (!isRtx)
        if (k_addRtx)
            return FECParam (maxGroupSize, 0);
        else
            return FECParam (maxGroupSize, statistic->curLossRate);
    else {
        if (!k_addRtx)
            return FECParam (maxGroupSize, 0);
        else {
            if (k_token >= maxGroupSize) {
                k_token -= maxGroupSize;
                return FECParam (maxGroupSize, 1.f);
            } else {
                double_t fecRate = (MAX (k_token, 0)) / maxGroupSize;
                k_token = 0;
                return FECParam (maxGroupSize, fecRate);
            }
        }
    }
};

std::string TokenRtxPolicy::GetFecName (void) {
    return "TokenRtxPolicy";
}

/* class RtxOnlyPolicy */
RtxOnlyPolicy::RtxOnlyPolicy () : FixedPolicy (0) {};
RtxOnlyPolicy::~RtxOnlyPolicy() {};

TypeId RtxOnlyPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::RtxOnlyPolicy")
        .SetParent<FECPolicy> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<RtxOnlyPolicy> ()
    ;
    return tid;
};

std::string RtxOnlyPolicy::GetFecName (void) {
    return "RtxOnlyPolicy";
}


/* class PtoOnlyPolicy */
PtoOnlyPolicy::PtoOnlyPolicy () : FixedPolicy (0) {};
PtoOnlyPolicy::~PtoOnlyPolicy () {};

TypeId PtoOnlyPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::PtoOnlyPolicy")
        .SetParent<FECPolicy> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<PtoOnlyPolicy> ()
    ;
    return tid;
};

std::string PtoOnlyPolicy::GetFecName (void) {
    return "PtoOnlyPolicy";
}

/* Bolot Policy comes from Bolot et al., INFOCOM 1999 
   We do not implement the per-packet mechanism in Bolot and USF.
   Instead, we calculate the average redundancy rate in each case. 
   Accordingly, we have the following mapping:
   ====================
   Comb | Rate | Reward
   =====|======|=======
    0   | 0    | 1
    1   | 1.0f | 4
    2   | 1.0f | 4
    3   | 2.0f | 8
    4   | 2.0f | 8
    5   | 2.0f | 8
    6   | 2.0f | 8
    7   | 3.0f | 18
    8   | 3.0f | 18
    9   | 4.0f | 18
    ===================

   */
BolotPolicy::BolotPolicy() : FECPolicy(MilliSeconds(1)) {
    k_bolotLow = 0.03f;
    k_bolotHigh = 0.03f;
    k_rewardList = new double_t[10] {1, 4, 4, 8, 8, 8, 8, 18, 18, 18};
    k_rateList = new double_t[10] {0, 1, 1, 2, 2, 2, 2, 3, 3, 4};
    m_lastComb = 0;
};

BolotPolicy::~BolotPolicy() {};

TypeId BolotPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::BolotPolicy")
        .SetParent<FECPolicy> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<BolotPolicy> ()
    ;
    return tid;
};

FECPolicy::FECParam BolotPolicy::GetPolicyFECParam (
    Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
    bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize) {
    if (!isRtx) {
        double_t lossPb = statistic->curLossRate;
        double_t lossPa = lossPb / k_rewardList[m_lastComb];
        if (lossPa > k_bolotHigh) {
            m_lastComb = std::min (10, m_lastComb + 1);
        }
        if (lossPb < k_bolotLow) {
            m_lastComb = std::max (0, m_lastComb - 1);
        }
        return FECParam (maxGroupSize, k_rateList[m_lastComb]);
    }
    else
        return FECParam (maxGroupSize, 0);
};

std::string BolotPolicy::GetFecName (void) {
    return "BolotPolicy";
}

/* USF Policy comes from Padhye et al., INFOCOM 2000 
   We do not implement the per-packet mechanism in Bolot and USF.
   Instead, we calculate the average redundancy rate in each case. 
   Accordingly, we have the following mapping:
   ====================
   Comb | Rate | Reward
   =====|======|=======
    0   | 0    | 1
    1   | 1.0f | 4
    2   | 1.0f | 4
    3   | 2.0f | 8
    4   | 2.0f | 8
    5   | 3.0f | 18
    6   | 3.0f | 18
    7   | 3.0f | 18
    8   | 4.0f | 18
    ===================

   */
UsfPolicy::UsfPolicy() : FECPolicy(MilliSeconds(1)) {
    k_bolotLow = 0.03f;
    k_bolotHigh = 0.03f;
    k_minThresh = 0.03f;
    k_rewardList = new double_t[9] {1, 4, 4, 8, 8, 18, 18, 18, 18};
    k_rateList = new double_t[9] {0, 1, 1, 2, 2, 3, 3, 3, 4};
    m_lastComb = 0;
    m_lossPbPrevious = 0.f;
};

UsfPolicy::~UsfPolicy() {};

TypeId UsfPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::UsfPolocy")
        .SetParent<FECPolicy> ()
        .SetGroupName("spark-rtc")
        .AddConstructor<UsfPolicy> ()
    ;
    return tid;
};

FECPolicy::FECParam UsfPolicy::GetPolicyFECParam (
    Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
    bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize) {
    if (!isRtx) {
        double_t lossPb = statistic->curLossRate;
        double_t lossPa = lossPb / k_rewardList[m_lastComb];
        double_t lossDiff = m_lossPbPrevious - lossPb;
        if (lossPa > k_bolotHigh) {
            m_lastComb = std::min (9, m_lastComb + 1);
        }
        if (lossPb < k_bolotLow && lossDiff > k_minThresh) {
            m_lastComb = std::max (0, m_lastComb - 1);
        }
        m_lossPbPrevious = lossPb;
        if (lossPb < 0.01)
            m_lastComb = 0;
        return FECParam (maxGroupSize, k_rateList[m_lastComb]);
    }
    else 
        return FECParam (maxGroupSize, 0);
};

std::string UsfPolicy::GetFecName (void) {
    return "UsfPolicy";
}

}; // namespace ns3