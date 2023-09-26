#ifndef OTHER_POLICY_H
#define OTHER_POLICY_H

#include "common-header.h"
#include "fec-policy.h"
#include "ns3/object.h"
#include "ns3/nstime.h"
#include <vector>
#include "math.h"

namespace ns3{

class FixedPolicy : public FECPolicy {
public:
    FixedPolicy ();
    FixedPolicy (double_t rate);
    ~FixedPolicy ();
    static TypeId GetTypeId (void);
protected:
    double_t k_rate;
public:
    FECParam GetPolicyFECParam (
        Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
        bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize);
    std::string GetFecName (void);
};  // class FixedPolicy

class FixedRtxPolicy : public FixedPolicy {
public:
    FixedRtxPolicy ();
    FixedRtxPolicy (double_t rate);
    ~FixedRtxPolicy ();
    static TypeId GetTypeId (void);
public:
    FECParam GetPolicyFECParam (
        Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
        bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize);
    std::string GetFecName (void);
};  // class FixedRtxPolicy
// This is a weird baseline required by the reviewers in NSDI'23 Fall.
// It is almost the same as the Hairpin variant with a fixed ratio k.
// The only difference is that it does not use the FEC for init transmission.

class TokenRtxPolicy : public FixedPolicy {
public:
    TokenRtxPolicy ();
    ~TokenRtxPolicy ();
    static TypeId GetTypeId (void);
private:
    uint16_t k_token;   // Total tokens for fec in the current round.
    bool k_addRtx;      // Tokens are going to added over init or rtx in the current round.
public:
    FECParam GetPolicyFECParam (
        Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
        bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize);
    std::string GetFecName (void);
};  // class TokenRtxPolicy


class RtxOnlyPolicy : public FixedPolicy {
public:
    RtxOnlyPolicy ();
    ~RtxOnlyPolicy ();
    static TypeId GetTypeId (void);
    std::string GetFecName (void);
};  // class RtxOnlyPolicy

class PtoOnlyPolicy : public FixedPolicy {
public:
    PtoOnlyPolicy ();
    ~PtoOnlyPolicy ();
    static TypeId GetTypeId (void);
    std::string GetFecName (void);
};  // class PtoOnlyPolicy

class BolotPolicy : public FECPolicy {
public:
    BolotPolicy ();
    ~BolotPolicy ();
    static TypeId GetTypeId (void);
private:
    double_t k_bolotLow;
    double_t k_bolotHigh;
    double_t* k_rewardList;
    double_t* k_rateList;
    int m_lastComb;
public:
    FECParam GetPolicyFECParam (
        Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
        bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize);
    std::string GetFecName (void);
};  // class BolotPolicy

class UsfPolicy : public FECPolicy {
public:
    UsfPolicy();
    ~UsfPolicy();
    static TypeId GetTypeId (void);
private:
    double_t k_bolotLow;
    double_t k_bolotHigh;
    double_t k_minThresh;
    double_t* k_rewardList;
    double_t* k_rateList;
    int m_lastComb;
    double_t m_lossPbPrevious;
public:
    FECParam GetPolicyFECParam (
        Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
        bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize);
    std::string GetFecName (void);
};  // class UsfPolicy

}; // namespace ns3

#endif /* OTHER_POLICY_H */