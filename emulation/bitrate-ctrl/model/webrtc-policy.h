#ifndef WEBRTC_POLICY_H
#define WEBRTC_POLICY_H

#include "fec-policy.h"
#include "ns3/webrtc-adjust-array.h"
#include "ns3/webrtc-fec-array.h"
#include <deque>

namespace ns3 {

class WebRtcLossFilter : public Object {
public:
    WebRtcLossFilter ();
    ~WebRtcLossFilter ();
    static TypeId GetTypeId (void);
private:
    std::deque<std::pair<double_t, Time>> m_longLossList;
    std::deque<std::pair<double_t, Time>> m_shortLossList;
    Time m_longWindow;
    Time m_shortWindow;
public:
    double_t UpdateAndGetLoss (double_t loss, Time now);
};

class WebRTCPolicy : public FECPolicy {
public:
    WebRTCPolicy ();
    ~WebRTCPolicy ();
    static TypeId GetTypeId (void);
private:
    Ptr<WebRtcLossFilter> m_lossFilter;
public:
    FECParam GetPolicyFECParam (
        Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
        bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize);
    std::string GetFecName (void);
};  // class WebRTCPolicy

class WebRTCAdaptivePolicy : public FECPolicy {
public:
    WebRTCAdaptivePolicy ();
    ~WebRTCAdaptivePolicy ();
    static TypeId GetTypeId (void);
private:
    Ptr<WebRtcLossFilter> m_lossFilter;
public:
    FECParam GetPolicyFECParam (
        Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
        bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize);
    std::string GetFecName (void);
};  // class WebRTCAdaptivePolicy

class WebRTCStarPolicy : public FECPolicy {
public:
    WebRTCStarPolicy ();
    WebRTCStarPolicy (int order, double_t coeff);
    ~WebRTCStarPolicy ();
    static TypeId GetTypeId (void);
private:
    int m_order;
    double_t m_coeff;
    double_t LinearFECRate (double_t fecRate, uint16_t ddlLeft, uint16_t rtt);
    double_t QuadraticFECRate (double_t fecRate, uint16_t ddlLeft, uint16_t rtt);
    double_t SqrtFECRate (double_t fecRate, uint16_t ddlLeft, uint16_t rtt);
    Ptr<WebRtcLossFilter> m_lossFilter;
public:
    FECParam GetPolicyFECParam (
        Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
        bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize);
    std::string GetFecName (void);
    void SetOrder (int order);
}; // class WebRTCStarPolicy

}

#endif