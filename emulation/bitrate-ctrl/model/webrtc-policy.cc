#include "webrtc-policy.h"

namespace ns3 {
NS_LOG_COMPONENT_DEFINE("WebRtcPolicy");

/* class WebRtcLossFilter */
/* In WebRtc-based algorithms, WebRtc will take the max loss rate from recent 10 seconds.
   Ref. modules/video_coding/fec_controller_default.cc - external/webrtc - Git at Google
   https://chromium.googlesource.com/external/webrtc/+/6deec38edef45e7505330eb892f9c75b7e6f6ba4/modules/video_coding/fec_controller_default.cc#108
*/

WebRtcLossFilter::WebRtcLossFilter ()
: m_longWindow {Seconds (10)}
, m_shortWindow {Seconds (1)}
{
    m_longLossList.clear ();
    m_shortLossList.clear ();
}

WebRtcLossFilter::~WebRtcLossFilter ()
{
    m_longLossList.clear ();
    m_shortLossList.clear ();
}

TypeId WebRtcLossFilter::GetTypeId (void)
{
    static TypeId tid = TypeId ("ns3::WebRtcLossFilter")
        .SetParent<Object> ()
        .AddConstructor<WebRtcLossFilter> ()
        ;
    return tid;
}

double_t WebRtcLossFilter::UpdateAndGetLoss (double_t loss, Time now)
{
    while (!m_shortLossList.empty () && m_shortLossList.front ().second < now - m_shortWindow) {
        m_shortLossList.pop_front ();
    }
    m_shortLossList.push_back (std::make_pair (loss, now));

    while (!m_longLossList.empty () && m_longLossList.front ().second < now - m_longWindow) {
        m_longLossList.pop_front ();
    }
    if (m_longLossList.empty () || m_longLossList.back ().second < now - m_shortWindow) {
        double_t avgLoss = 0;
        for (auto &loss : m_shortLossList) {
            avgLoss += loss.first;
        }
        avgLoss /= m_shortLossList.size ();
        m_longLossList.push_back (std::make_pair (avgLoss, now));
        m_shortLossList.clear ();
    }

    double_t maxLoss = 0;
    for (auto &loss : m_longLossList) {
        if (loss.first > maxLoss) {
            maxLoss = loss.first;
        }
    }
    return maxLoss;
}

/* class WebRTCPolicy */
WebRTCPolicy::WebRTCPolicy () : FECPolicy (MilliSeconds(1)) {
    m_lossFilter = CreateObject<WebRtcLossFilter> ();
};

WebRTCPolicy::~WebRTCPolicy () {};

TypeId WebRTCPolicy::GetTypeId () {
    static TypeId tid = TypeId ("ns3::WebRTCPolicy")
        .SetParent<FECPolicy> ()
        .SetGroupName ("bitrate-ctrl")
        .AddConstructor<WebRTCPolicy> ()
    ;
    return tid;
};

FECPolicy::FECParam WebRTCPolicy::GetPolicyFECParam (
    Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
    bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize) {
    double_t fecRate = 0;
    if (!isRtx) {
        double_t loss = statistic->curLossRate;
        maxGroupSize = MIN (maxGroupSize, 48U);
        get_fec_rate_webrtc (m_lossFilter->UpdateAndGetLoss (loss, Simulator::Now ()), 
            maxGroupSize, ((double_t) bitrate) / 1000, &fecRate);
    }
    fecRate = MIN (fecRate, 1);
    return FECParam (maxGroupSize, fecRate);
};

std::string WebRTCPolicy::GetFecName (void) {
    return "WebRTCPolicy";
}


/* class WebRTCAdaptivePolicy */
WebRTCAdaptivePolicy::WebRTCAdaptivePolicy () : FECPolicy(MilliSeconds(1)) {
    m_lossFilter = CreateObject<WebRtcLossFilter> ();
};

WebRTCAdaptivePolicy::~WebRTCAdaptivePolicy () {};

TypeId WebRTCAdaptivePolicy::GetTypeId () {
    static TypeId tid = TypeId ("ns3::WebRTCAdaptivePolicy")
        .SetParent<FECPolicy> ()
        .SetGroupName ("bitrate-ctrl")
        .AddConstructor<WebRTCAdaptivePolicy> ()
    ;
    return tid;
};

FECPolicy::FECParam WebRTCAdaptivePolicy::GetPolicyFECParam (
    Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
    bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize) {
    double_t fecRate = 0;
    if (!isRtx) {
        double_t loss = statistic->curLossRate;
        maxGroupSize = MIN (maxGroupSize, 48);
        get_fec_rate_webrtc_rtt (m_lossFilter->UpdateAndGetLoss (loss, Simulator::Now ()), 
            maxGroupSize, (uint16_t) statistic->srtt.GetMilliSeconds (), ((double_t) bitrate) / 1000, &fecRate);
    }
    fecRate = MIN (fecRate, 1);

    return FECParam (maxGroupSize, fecRate);
};

std::string WebRTCAdaptivePolicy::GetFecName (void) {
    return "WebRTCAdaptivePolicy";
}


/* class WebRTCStarPolicy */
WebRTCStarPolicy::WebRTCStarPolicy () 
: FECPolicy (MilliSeconds(1)) {
    m_lossFilter = CreateObject<WebRtcLossFilter> ();
    m_order = 1;
};

WebRTCStarPolicy::WebRTCStarPolicy (int order, double_t coeff) 
: FECPolicy (MilliSeconds(1)) {
    m_lossFilter = CreateObject<WebRtcLossFilter> ();
    m_order = order;
    m_coeff = coeff;
};

WebRTCStarPolicy::~WebRTCStarPolicy() {};

TypeId WebRTCStarPolicy::GetTypeId() {
    static TypeId tid = TypeId ("ns3::WebRTCStarPolicy")
        .SetParent<FECPolicy> ()
        .SetGroupName("bitrate-ctrl")
        .AddConstructor<WebRTCStarPolicy> ()
    ;
    return tid;
};

double_t WebRTCStarPolicy::LinearFECRate (double_t beta, uint16_t ddlLeft, uint16_t rtt) {
    double_t rttToDdlLeft = (double_t) rtt / (double_t) ddlLeft;
    double_t adjustedBeta = std::min (m_coeff * beta * rttToDdlLeft, 1.);
    return adjustedBeta;
};

double_t WebRTCStarPolicy::QuadraticFECRate (double_t beta, uint16_t ddl_left, uint16_t rtt) {
    double_t rtt_to_ddl_left = (double_t) rtt / (double_t) ddl_left;
    double_t new_beta = 4. * beta * rtt_to_ddl_left * rtt_to_ddl_left;
    return new_beta;
};

double_t WebRTCStarPolicy::SqrtFECRate (double_t beta, uint16_t ddl_left, uint16_t rtt){
    double_t rtt_to_ddl_left = (double_t) rtt / (double_t) ddl_left;
    double_t new_beta = beta *  sqrt(2. * rtt_to_ddl_left);
    return new_beta;
};

FECPolicy::FECParam WebRTCStarPolicy::GetPolicyFECParam (
    Ptr<NetStat> statistic, uint32_t bitrate, uint16_t ddl, uint16_t ddlLeft,
    bool isRtx, uint8_t frameSize, uint16_t maxGroupSize, bool fixGroupSize) {
    double_t fecRate;
    double_t loss = statistic->curLossRate;
    uint16_t rtt = statistic->curRtt.GetMilliSeconds ();
    maxGroupSize = MIN (maxGroupSize, 48);
    get_fec_rate_webrtc (m_lossFilter->UpdateAndGetLoss (loss, Simulator::Now ()), 
        maxGroupSize, ((double_t) bitrate) / 1000, &fecRate);

    fecRate = MIN (fecRate, 1);
    if (m_order == 1)
        fecRate = LinearFECRate (fecRate, ddlLeft, rtt);
    else if (m_order == 2)
        fecRate = QuadraticFECRate (fecRate, ddlLeft, rtt);
    else if (m_order == 0)
        fecRate = SqrtFECRate (fecRate, ddlLeft, rtt);     
    else
        NS_LOG_ERROR ("Does not support higher-than-2-order Deadline-aware Multiplier, fallback to the original WebRTC");
    return FECParam (maxGroupSize, fecRate);
};

std::string WebRTCStarPolicy::GetFecName (void) {
    return "WebRTCStarPolicy";
}

void WebRTCStarPolicy::SetOrder (int order){
    m_order = order;
}

}